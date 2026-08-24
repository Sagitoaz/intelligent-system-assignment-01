import { useEffect, useRef, useState } from 'react'
import ForceGraph2D, { type ForceGraphMethods } from 'react-force-graph-2d'
import { api } from '../api'
import type { GraphData, GraphEdge, GraphNode } from '../types'

const colors: Record<string, string> = {
  System: '#1d5a54', Model: '#ee9b5a', Feature: '#49a59a', Target: '#d76850',
  PipelineStep: '#7b75b4', Dataset: '#3b6d9a', Experiment: '#b07b38', Metric: '#73864a',
}

export function KnowledgeGraphPage() {
  const [graph, setGraph] = useState<GraphData | null>(null)
  const [selected, setSelected] = useState<GraphNode | null>(null)
  const [hovered, setHovered] = useState<GraphNode | null>(null)
  const [error, setError] = useState('')
  const container = useRef<HTMLDivElement>(null)
  const graphRef = useRef<ForceGraphMethods<GraphNode, GraphEdge>>(undefined)
  const [width, setWidth] = useState(900)
  const graphHeight = width < 680 ? 470 : 620

  useEffect(() => {
    api.graph().then((data) => {
      setGraph(data)
      setSelected(data.nodes.find((node) => node.labels.includes('System')) ?? data.nodes[0] ?? null)
    }).catch((reason: Error) => setError(reason.message))
  }, [])
  useEffect(() => {
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width))
    if (container.current) observer.observe(container.current)
    return () => observer.disconnect()
  }, [])
  useEffect(() => {
    if (!graph || !graphRef.current) return
    graphRef.current.d3Force('charge')?.strength?.(-290)
    graphRef.current.d3Force('link')?.distance?.(105)
    graphRef.current.d3ReheatSimulation()
  }, [graph])

  function fitGraph() {
    graphRef.current?.zoomToFit(450, width < 680 ? 34 : 70)
  }

  function paintNode(node: GraphNode, context: CanvasRenderingContext2D, globalScale: number) {
    if (node.x == null || node.y == null) return
    const semanticLabel = node.labels.find((label) => colors[label])
    const color = semanticLabel ? colors[semanticLabel] : '#65717a'
    const active = node.id === selected?.id
    const isHovered = node.id === hovered?.id
    const radius = active ? 9 : isHovered ? 8 : 7

    context.beginPath()
    context.arc(node.x, node.y, radius, 0, 2 * Math.PI)
    context.fillStyle = color
    context.fill()
    context.lineWidth = (active ? 3 : 1.5) / globalScale
    context.strokeStyle = active ? '#18312c' : '#ffffff'
    context.stroke()

    if (!active && !isHovered) return
    const label = String(node.properties.name ?? semanticLabel ?? 'Node')
    const fontSize = 12 / globalScale
    context.font = `600 ${fontSize}px DM Sans, sans-serif`
    context.textAlign = 'center'
    context.textBaseline = 'middle'
    const textWidth = context.measureText(label).width
    const boxHeight = fontSize * 1.65
    const boxY = node.y + radius + 4 / globalScale
    context.fillStyle = 'rgba(251, 252, 248, 0.95)'
    context.fillRect(node.x - textWidth / 2 - 5 / globalScale, boxY, textWidth + 10 / globalScale, boxHeight)
    context.fillStyle = '#18312c'
    context.fillText(label, node.x, boxY + boxHeight / 2)
  }

  return (
    <div className="page graph-page">
      <section className="page-heading">
        <span className="eyebrow">Transparent, model-centric graph</span>
        <h1>Diabetes Knowledge Graph</h1>
        <p>Inspect how the intelligent system connects its dataset, raw features, fitted pipeline, experiments, metrics and target.</p>
      </section>
      {error && <div className="alert error graph-error"><strong>Knowledge graph unavailable</strong><span>{error}</span><p>Predictions remain available. Start and seed Neo4j to enable this view.</p></div>}
      {!graph && !error && <div className="status-card">Loading Neo4j graph…</div>}
      {graph && (
        <div className="graph-layout">
          <div className="graph-canvas" ref={container}>
            <div className="graph-toolbar"><span>{graph.nodes.length} nodes · {graph.edges.length} relationships</span><button onClick={fitGraph} type="button">Fit graph</button></div>
            <ForceGraph2D
              ref={graphRef}
              width={width}
              height={graphHeight}
              graphData={{ nodes: graph.nodes, links: graph.edges }}
              nodeLabel={(node) => String((node as GraphNode).properties.name ?? '')}
              nodeCanvasObjectMode={() => 'replace'}
              nodeCanvasObject={(node, context, globalScale) => paintNode(node as GraphNode, context, globalScale)}
              nodePointerAreaPaint={(node, paintColor, context) => {
                const graphNode = node as GraphNode
                if (graphNode.x == null || graphNode.y == null) return
                context.beginPath()
                context.arc(graphNode.x, graphNode.y, 13, 0, 2 * Math.PI)
                context.fillStyle = paintColor
                context.fill()
              }}
              nodeRelSize={7}
              linkLabel={(link) => (link as unknown as GraphEdge).type}
              linkColor={() => '#9ca9a4'}
              linkWidth={1.15}
              linkCurvature={0.06}
              linkDirectionalArrowLength={5}
              linkDirectionalArrowRelPos={1}
              d3AlphaDecay={0.025}
              d3VelocityDecay={0.28}
              warmupTicks={80}
              cooldownTicks={220}
              onEngineStop={fitGraph}
              onNodeClick={(node) => setSelected(node as GraphNode)}
              onNodeHover={(node) => setHovered(node as GraphNode | null)}
              showPointerCursor={(object) => Boolean(object && 'labels' in object)}
            />
            <div className="graph-hint">Drag to pan · scroll to zoom · every node is clickable</div>
          </div>
          <aside className="graph-details">
            <h2>Node details</h2>
            {selected ? (
              <><span className="task-pill">{selected.labels.filter((label) => label !== 'AssignmentEntity').join(', ')}</span><h3>{String(selected.properties.name ?? 'Untitled')}</h3>
                <dl>{Object.entries(selected.properties).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{String(value)}</dd></div>)}</dl></>
            ) : <p>Select a node in the graph to inspect its sourced properties.</p>}
            <div className="legend">{Object.entries(colors).map(([label, color]) => <span key={label}><i style={{ background: color }} />{label}</span>)}</div>
          </aside>
        </div>
      )}
    </div>
  )
}
