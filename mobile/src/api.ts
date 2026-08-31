import type { GraphData, ModelMetadata } from './types'

export const API_BASE_URL = (process.env.EXPO_PUBLIC_API_BASE_URL || '').replace(/\/$/, '')

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  if (!API_BASE_URL) {
    throw new Error('EXPO_PUBLIC_API_BASE_URL is not configured for this device.')
  }
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...init?.headers },
    })
  } catch {
    throw new Error('Cannot reach the prediction service. Check your connection and try again.')
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    const detail = body?.detail
    throw new Error(typeof detail === 'string' ? detail : `API request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

export const api = {
  metadata: (slug: 'diabetes' | 'house-price' | 'ecommerce') => request<ModelMetadata>(`/api/v1/models/${slug}`),
  predict: <T>(path: string, payload: Record<string, unknown>) =>
    request<T>(path, { method: 'POST', body: JSON.stringify(payload) }),
  graph: () => request<GraphData>('/api/v1/diabetes/knowledge-graph'),
}
