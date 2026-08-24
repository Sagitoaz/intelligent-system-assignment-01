export type FieldMetadata = {
  name: string
  label: string
  type: 'number' | 'categorical'
  integer?: boolean
  unit: string | null
  description: string
  required: boolean
  nullable: boolean
  options?: string[]
  example?: string | number
}

export type ModelMetadata = {
  id: 'diabetes' | 'house_price'
  name: string
  task: 'classification' | 'regression'
  dataset: string
  expected_raw_features: string[]
  fields: FieldMetadata[]
  model: { name: string; configuration: Record<string, unknown>; preprocessing: string[] }
  target?: { name: string; unit: string; description: string }
  class_labels?: Record<string, string>
  metrics: Record<string, number>
  disclaimer: string
}

export type DiabetesResult = {
  prediction: number
  label: string
  probability: number | null
  model: string
  disclaimer: string
}

export type HouseResult = {
  predicted_price: number
  unit: string
  formatted: string
  model: string
  disclaimer: string
}

export type GraphNode = {
  id: string
  labels: string[]
  properties: Record<string, string | number | boolean>
  x?: number
  y?: number
}

export type GraphEdge = {
  id: string
  source: string | GraphNode
  target: string | GraphNode
  type: string
  properties: Record<string, string | number | boolean>
}

export type GraphData = { nodes: GraphNode[]; edges: GraphEdge[] }
