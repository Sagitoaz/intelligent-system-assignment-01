import { useEffect, useMemo, useState } from 'react'
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native'
import { api } from '../api'
import { colors, shared, spacing } from '../theme'
import type { GraphData, GraphEdge, GraphNode } from '../types'

const labelColors: Record<string, string> = {
  System: colors.teal,
  Model: colors.orange,
  Feature: '#49a59a',
  Target: '#d76850',
  PipelineStep: '#7b75b4',
  Dataset: '#3b6d9a',
  Experiment: '#b07b38',
  Metric: '#73864a',
}

export function KnowledgeGraphScreen() {
  const [graph, setGraph] = useState<GraphData | null>(null)
  const [selectedId, setSelectedId] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  async function loadGraph() {
    setLoading(true)
    setError('')
    try {
      const response = await api.graph()
      setGraph(response)
      const first = response.nodes.find((node) => semanticLabel(node) === 'System') ?? response.nodes[0]
      setSelectedId(first?.id ?? '')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not load the Knowledge Graph.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { void loadGraph() }, [])

  const nodesById = useMemo(
    () => new Map(graph?.nodes.map((node) => [node.id, node]) ?? []),
    [graph],
  )
  const selected = selectedId ? nodesById.get(selectedId) : undefined
  const connections = useMemo(
    () => graph?.edges.filter((edge) => edge.source === selectedId || edge.target === selectedId) ?? [],
    [graph, selectedId],
  )
  const labelCounts = useMemo(() => {
    const counts = new Map<string, number>()
    graph?.nodes.forEach((node) => {
      const label = semanticLabel(node)
      counts.set(label, (counts.get(label) ?? 0) + 1)
    })
    return [...counts.entries()].sort(([a], [b]) => a.localeCompare(b))
  }, [graph])

  return <ScrollView style={shared.screen} contentContainerStyle={shared.content} showsVerticalScrollIndicator={false}>
    <Text style={shared.eyebrow}>Transparent · Model-centric</Text>
    <Text style={shared.title}>Diabetes Knowledge Graph</Text>
    <Text style={shared.intro}>Explore how the fitted model connects its dataset, raw features, pipeline, experiment, metrics and target.</Text>

    {loading ? <View style={styles.loading}>
      <ActivityIndicator color={colors.teal} size="large" />
      <Text style={styles.loadingText}>Loading Neo4j graph…</Text>
    </View> : null}

    {error ? <View accessibilityRole="alert" style={styles.errorCard}>
      <Text style={styles.errorTitle}>Knowledge Graph unavailable</Text>
      <Text style={styles.errorText}>{error}</Text>
      <Text style={styles.errorNote}>Diabetes and House Price predictions remain available.</Text>
      <Pressable accessibilityRole="button" onPress={() => void loadGraph()} style={({ pressed }) => [styles.retry, pressed && styles.pressed]}>
        <Text style={styles.retryText}>Try again</Text>
      </Pressable>
    </View> : null}

    {graph ? <>
      <View style={styles.summaryRow}>
        <Summary value={graph.nodes.length} label="Nodes" />
        <Summary value={graph.edges.length} label="Relationships" />
      </View>

      <View style={shared.card}>
        <Text style={styles.sectionTitle}>Graph schema</Text>
        <View style={styles.legend}>
          {labelCounts.map(([label, count]) => <View key={label} style={styles.legendItem}>
            <View style={[styles.dot, { backgroundColor: labelColors[label] ?? colors.muted }]} />
            <Text style={styles.legendLabel}>{splitLabel(label)}</Text>
            <Text style={styles.legendCount}>{count}</Text>
          </View>)}
        </View>
      </View>

      <Text style={styles.sectionHeading}>Node explorer</Text>
      <ScrollView horizontal contentContainerStyle={styles.nodeStrip} showsHorizontalScrollIndicator={false}>
        {graph.nodes.map((node) => {
          const label = semanticLabel(node)
          const active = node.id === selectedId
          return <Pressable
            accessibilityRole="button"
            accessibilityState={{ selected: active }}
            key={node.id}
            onPress={() => setSelectedId(node.id)}
            style={({ pressed }) => [styles.nodeChip, active && styles.nodeChipActive, pressed && styles.pressed]}
          >
            <View style={[styles.chipDot, { backgroundColor: labelColors[label] ?? colors.muted }]} />
            <Text numberOfLines={1} style={[styles.nodeChipText, active && styles.nodeChipTextActive]}>{nodeName(node)}</Text>
          </Pressable>
        })}
      </ScrollView>

      {selected ? <View style={styles.nodeCard}>
        <View style={styles.nodeHeader}>
          <View style={[styles.nodeIcon, { backgroundColor: labelColors[semanticLabel(selected)] ?? colors.muted }]}>
            <Text style={styles.nodeIconText}>{initials(semanticLabel(selected))}</Text>
          </View>
          <View style={styles.nodeHeadingCopy}>
            <Text style={styles.nodeType}>{splitLabel(semanticLabel(selected))}</Text>
            <Text style={styles.nodeName}>{nodeName(selected)}</Text>
          </View>
        </View>
        <View style={styles.properties}>
          {Object.entries(selected.properties).map(([key, value]) => <View style={styles.propertyRow} key={key}>
            <Text style={styles.propertyKey}>{splitLabel(key)}</Text>
            <Text selectable style={styles.propertyValue}>{formatValue(value)}</Text>
          </View>)}
        </View>
      </View> : null}

      <Text style={styles.sectionHeading}>Connections · {connections.length}</Text>
      <View style={styles.connectionsCard}>
        {connections.length ? connections.map((edge, index) => <Connection
          edge={edge}
          isLast={index === connections.length - 1}
          key={edge.id}
          nodesById={nodesById}
          selectedId={selectedId}
          selectNode={setSelectedId}
        />) : <Text style={styles.empty}>This node has no graph connections.</Text>}
      </View>
      <Text style={shared.disclaimer}>This graph describes the assignment model and its provenance. It is not a medical knowledge base.</Text>
    </> : null}
  </ScrollView>
}

function Summary({ value, label }: { value: number; label: string }) {
  return <View style={styles.summaryCard}><Text style={styles.summaryValue}>{value}</Text><Text style={styles.summaryLabel}>{label}</Text></View>
}

function Connection({ edge, isLast, nodesById, selectedId, selectNode }: {
  edge: GraphEdge
  isLast: boolean
  nodesById: Map<string, GraphNode>
  selectedId: string
  selectNode: (id: string) => void
}) {
  const outgoing = edge.source === selectedId
  const otherId = outgoing ? edge.target : edge.source
  const other = nodesById.get(otherId)
  return <Pressable onPress={() => selectNode(otherId)} style={({ pressed }) => [styles.connection, !isLast && styles.connectionBorder, pressed && styles.pressed]}>
    <View style={styles.direction}><Text style={styles.directionText}>{outgoing ? '→' : '←'}</Text></View>
    <View style={styles.connectionCopy}>
      <Text style={styles.relationship}>{splitLabel(edge.type)}</Text>
      <Text numberOfLines={2} style={styles.connectedNode}>{other ? nodeName(other) : 'Unknown node'}</Text>
    </View>
    <Text style={styles.openArrow}>›</Text>
  </Pressable>
}

function semanticLabel(node: GraphNode) {
  return node.labels.find((label) => label !== 'AssignmentEntity') ?? 'Entity'
}

function nodeName(node: GraphNode) {
  return String(node.properties.name ?? node.properties.key ?? 'Untitled node')
}

function splitLabel(value: string) {
  return value.replaceAll('_', ' ').replace(/([a-z])([A-Z])/g, '$1 $2')
}

function initials(value: string) {
  return splitLabel(value).split(' ').map((part) => part[0]).join('').slice(0, 2).toUpperCase()
}

function formatValue(value: unknown) {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

const styles = StyleSheet.create({
  loading: { alignItems: 'center', paddingVertical: spacing.xxl },
  loadingText: { color: colors.muted, fontSize: 13, marginTop: spacing.sm },
  errorCard: { backgroundColor: '#f8e4df', borderColor: '#ebc2b8', borderWidth: 1, borderRadius: 16, padding: spacing.lg },
  errorTitle: { color: colors.error, fontSize: 17, fontWeight: '700' },
  errorText: { color: colors.error, lineHeight: 20, marginTop: spacing.xs },
  errorNote: { color: colors.muted, fontSize: 12, lineHeight: 18, marginTop: spacing.sm },
  retry: { alignSelf: 'flex-start', backgroundColor: colors.error, borderRadius: 10, paddingHorizontal: spacing.md, paddingVertical: spacing.sm, marginTop: spacing.md },
  retryText: { color: 'white', fontWeight: '700' },
  summaryRow: { flexDirection: 'row', gap: spacing.sm, marginBottom: spacing.md },
  summaryCard: { flex: 1, minHeight: 90, borderRadius: 16, padding: spacing.md, justifyContent: 'center', backgroundColor: colors.teal },
  summaryValue: { color: 'white', fontSize: 30, lineHeight: 34, fontWeight: '800' },
  summaryLabel: { color: '#c2ddd7', fontSize: 12, marginTop: 2 },
  sectionTitle: { color: colors.ink, fontSize: 18, fontWeight: '700', marginBottom: spacing.md },
  legend: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm },
  legendItem: { minWidth: '46%', flexGrow: 1, flexDirection: 'row', alignItems: 'center', borderColor: colors.line, borderWidth: 1, borderRadius: 10, paddingHorizontal: spacing.sm, paddingVertical: 9 },
  dot: { width: 9, height: 9, borderRadius: 5, marginRight: 7 },
  legendLabel: { flex: 1, color: colors.ink, fontSize: 12 },
  legendCount: { color: colors.muted, fontSize: 11, fontWeight: '700' },
  sectionHeading: { color: colors.ink, fontSize: 18, fontWeight: '700', marginTop: spacing.xs, marginBottom: spacing.sm },
  nodeStrip: { gap: spacing.sm, paddingBottom: spacing.md },
  nodeChip: { maxWidth: 190, minHeight: 42, flexDirection: 'row', alignItems: 'center', borderColor: colors.line, borderWidth: 1, borderRadius: 21, paddingHorizontal: 13, backgroundColor: colors.surface },
  nodeChipActive: { borderColor: colors.teal, backgroundColor: colors.paleGreen },
  chipDot: { width: 8, height: 8, borderRadius: 4, marginRight: 7 },
  nodeChipText: { flexShrink: 1, color: colors.muted, fontSize: 12, fontWeight: '600' },
  nodeChipTextActive: { color: colors.teal },
  nodeCard: { backgroundColor: colors.surface, borderColor: colors.line, borderWidth: 1, borderRadius: 18, padding: spacing.lg, marginBottom: spacing.lg },
  nodeHeader: { flexDirection: 'row', alignItems: 'center', marginBottom: spacing.md },
  nodeIcon: { width: 45, height: 45, borderRadius: 14, alignItems: 'center', justifyContent: 'center', marginRight: spacing.sm },
  nodeIconText: { color: 'white', fontSize: 11, fontWeight: '800' },
  nodeHeadingCopy: { flex: 1 },
  nodeType: { color: colors.muted, fontSize: 10, fontWeight: '800', letterSpacing: 1, textTransform: 'uppercase' },
  nodeName: { color: colors.ink, fontSize: 20, lineHeight: 25, fontWeight: '700', marginTop: 2 },
  properties: { borderTopColor: colors.line, borderTopWidth: 1 },
  propertyRow: { paddingVertical: 10, borderBottomColor: colors.line, borderBottomWidth: 1 },
  propertyKey: { color: colors.muted, fontSize: 10, fontWeight: '700', textTransform: 'uppercase', letterSpacing: 0.7 },
  propertyValue: { color: colors.ink, fontSize: 13, lineHeight: 19, marginTop: 3 },
  connectionsCard: { backgroundColor: colors.surface, borderColor: colors.line, borderWidth: 1, borderRadius: 16, marginBottom: spacing.md, overflow: 'hidden' },
  connection: { minHeight: 67, flexDirection: 'row', alignItems: 'center', paddingHorizontal: spacing.md, paddingVertical: spacing.sm },
  connectionBorder: { borderBottomColor: colors.line, borderBottomWidth: 1 },
  direction: { width: 31, height: 31, borderRadius: 10, alignItems: 'center', justifyContent: 'center', backgroundColor: colors.paleGreen, marginRight: spacing.sm },
  directionText: { color: colors.teal, fontSize: 19, fontWeight: '700' },
  connectionCopy: { flex: 1 },
  relationship: { color: colors.orange, fontSize: 10, fontWeight: '800', letterSpacing: 0.7 },
  connectedNode: { color: colors.ink, fontSize: 14, lineHeight: 19, fontWeight: '600', marginTop: 2 },
  openArrow: { color: colors.muted, fontSize: 24, marginLeft: spacing.sm },
  empty: { color: colors.muted, padding: spacing.md },
  pressed: { opacity: 0.7 },
})
