import type { GraphData, ModelMetadata } from './types'

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  if (!API_BASE_URL) {
    throw new Error('VITE_API_BASE_URL is not configured for this deployment.')
  }
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...init?.headers },
    })
  } catch {
    throw new Error(`Cannot reach the API at ${API_BASE_URL}. Is FastAPI running?`)
  }

  if (!response.ok) {
    const body = await response.json().catch(() => null)
    const detail = body?.detail
    const message = Array.isArray(detail)
      ? detail.map((item: { loc?: string[]; msg?: string }) => `${item.loc?.at(-1) ?? 'Input'}: ${item.msg}`).join('; ')
      : detail
    throw new Error(message || `Request failed with status ${response.status}`)
  }
  return response.json() as Promise<T>
}

export const api = {
  model: (slug: 'diabetes' | 'house-price' | 'ecommerce') => request<ModelMetadata>(`/api/v1/models/${slug}`),
  predict: <T>(path: string, payload: Record<string, unknown>) =>
    request<T>(path, { method: 'POST', body: JSON.stringify(payload) }),
  graph: () => request<GraphData>('/api/v1/diabetes/knowledge-graph'),
}
