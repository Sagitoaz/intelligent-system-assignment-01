import { useEffect, useRef, useState } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import { api } from '../api'
import type { GraphData, GraphEdge, GraphNode } from '../types'

const colors: Record<string, string> = {
  System: '#1d5a54', Model: '#ee9b5a', Feature: '#49a59a', Target: '#d76850',
  PipelineStep: '#7b75b4', Dataset: '#3b6d9a', Experiment: '#b07b38', Metric: '#73864a',
}

export function KnowledgeGraphPage() {
  const [graph, setGraph] = useState<GraphData | null>(null)
  const [selected, setSelected] = useState<GraphNode | null>(null)
  const [error, setError] = useState('')
  const container = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(900)

  useEffect(() => {
    api.graph().then(setGraph).catch((reason: Error) => setError(reason.message))
  }, [])
  useEffect(() => {
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width))
    if (container.current) observer.observe(container.current)
    return () => observer.disconnect()
  }, [])

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
            <ForceGraph2D
              width={width}
              height={620}
              graphData={{ nodes: graph.nodes, links: graph.edges }}
              nodeLabel={(node) => String((node as GraphNode).properties.name ?? '')}
              nodeColor={(node) => {
                const semanticLabel = (node as GraphNode).labels.find((label) => colors[label])
                return semanticLabel ? colors[semanticLabel] : '#65717a'
              }}
              nodeRelSize={6}
              linkColor={() => '#9ca9a4'}
              linkDirectionalArrowLength={4}
              linkDirectionalArrowRelPos={1}
              linkCanvasObjectMode={() => 'after'}
              linkCanvasObject={(link, ctx) => {
                const edge = link as unknown as GraphEdge
                const source = edge.source as GraphNode
                const target = edge.target as GraphNode
                if (source.x == null || source.y == null || target.x == null || target.y == null) return
                ctx.font = '3px Inter, sans-serif'
                ctx.fillStyle = '#52615c'
                ctx.textAlign = 'center'
                ctx.fillText(edge.type, (source.x + target.x) / 2, (source.y + target.y) / 2)
              }}
              onNodeClick={(node) => setSelected(node as GraphNode)}
            />
            <div className="graph-hint">Drag to pan · scroll to zoom · click a node for details</div>
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
