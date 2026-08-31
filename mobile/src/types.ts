export type FieldMetadata = {
  name: string
  label: string
  type: 'number' | 'categorical' | 'text'
  multiline?: boolean
  integer?: boolean
  unit: string | null
  description: string
  nullable: boolean
  options?: string[]
  example?: string | number
}

export type ModelMetadata = {
  id: 'diabetes' | 'house_price' | 'ecommerce'
  name: string
  task: 'classification' | 'regression'
  fields: FieldMetadata[]
  model: { name: string; preprocessing: string[] }
  disclaimer: string
}

export type DiabetesResult = { prediction: number; label: string; probability: number | null; model: string; disclaimer: string }
export type HouseResult = { predicted_price: number; unit: string; formatted: string; model: string; disclaimer: string }
export type EcommerceResult = { prediction: number; label: 'Positive' | 'Negative'; probability: number; model: string; disclaimer: string }

export type GraphNode = {
  id: string
  labels: string[]
  properties: Record<string, unknown>
}

export type GraphEdge = {
  id: string
  source: string
  target: string
  type: string
  properties: Record<string, unknown>
}

export type GraphData = { nodes: GraphNode[]; edges: GraphEdge[] }
