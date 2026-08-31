import { useEffect, useMemo, useState, type FormEvent, type ReactNode } from 'react'
import { api } from '../api'
import { numericInputMinimum, numericInputStep, validateNumericValues } from '../numericValidation'
import type { ModelMetadata } from '../types'

type Props<T> = {
  slug: 'diabetes' | 'house-price' | 'ecommerce'
  endpoint: string
  eyebrow: string
  title: string
  intro: string
  submitLabel: string
  renderResult: (result: T) => ReactNode
}

export function PredictionForm<T>({ slug, endpoint, eyebrow, title, intro, submitLabel, renderResult }: Props<T>) {
  const [metadata, setMetadata] = useState<ModelMetadata | null>(null)
  const [values, setValues] = useState<Record<string, string>>({})
  const [result, setResult] = useState<T | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    api.model(slug)
      .then((data) => {
        setMetadata(data)
        setValues(Object.fromEntries(data.fields.map((field) => [field.name, String(field.example ?? '')])))
      })
      .catch((reason: Error) => setError(reason.message))
  }, [slug])

  const methodology = useMemo(() => metadata?.model.preprocessing.join(' → ') ?? '', [metadata])

  async function submit(event: FormEvent) {
    event.preventDefault()
    if (!metadata) return
    setError('')
    setResult(null)
    const validationError = validateNumericValues(metadata.fields, values)
    if (validationError) {
      setError(validationError)
      return
    }

    setLoading(true)
    const payload = Object.fromEntries(metadata.fields.map((field) => {
      const raw = values[field.name] ?? ''
      if (field.nullable && raw === '') return [field.name, null]
      return [field.name, field.type === 'number' ? Number(raw) : raw]
    }))
    try {
      setResult(await api.predict<T>(endpoint, payload))
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Prediction request failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page prediction-page">
      <section className="page-heading">
        <span className="eyebrow">{eyebrow}</span>
        <h1>{title}</h1>
        <p>{intro}</p>
      </section>

      {!metadata && !error && <div className="status-card">Loading verified model metadata…</div>}
      {metadata && (
        <div className="prediction-layout">
          <form className="form-card" onSubmit={submit}>
            <div className="form-heading">
              <div><span className="step-number">01</span><h2>Raw input</h2></div>
              <span className="field-count">{metadata.fields.length} fields</span>
            </div>
            <div className="form-grid">
              {metadata.fields.map((field) => (
                <label key={field.name}>
                  <span>{field.label}{field.unit ? <small> · {field.unit}</small> : null}</span>
                  {field.type === 'categorical' ? (
                    <select
                      required={!field.nullable}
                      value={values[field.name] ?? ''}
                      onChange={(event) => setValues({ ...values, [field.name]: event.target.value })}
                    >
                      {field.nullable && <option value="">Not provided — pipeline imputes</option>}
                      {field.options?.map((option) => <option key={option} value={option}>{option}</option>)}
                    </select>
                  ) : field.type === 'text' ? (
                    field.multiline ? <textarea
                      required value={values[field.name] ?? ''}
                      maxLength={20000} rows={7}
                      onChange={(event) => setValues({ ...values, [field.name]: event.target.value })}
                    /> : <input
                      type="text" required value={values[field.name] ?? ''} maxLength={500}
                      onChange={(event) => setValues({ ...values, [field.name]: event.target.value })}
                    />
                  ) : (
                    <input
                      type="number"
                      required={!field.nullable}
                      min={numericInputMinimum(field)}
                      step={numericInputStep(field)}
                      value={values[field.name] ?? ''}
                      onChange={(event) => setValues({ ...values, [field.name]: event.target.value })}
                    />
                  )}
                  <small className="field-help">{field.description}</small>
                </label>
              ))}
            </div>
            <button className="primary-button" disabled={loading} type="submit">
              {loading ? 'Running saved pipeline…' : submitLabel}
            </button>
          </form>

          <aside className="result-column">
            <div className="method-card">
              <span className="step-number">02</span>
              <h2>Pipeline</h2>
              <p>{methodology}</p>
              <div className="model-tag">{metadata.model.name}</div>
            </div>
            {error && <div className="alert error" role="alert"><strong>Could not complete request</strong><span>{error}</span></div>}
            {result && <div className="result-card">{renderResult(result)}</div>}
            {!result && !error && (
              <div className="empty-result"><span>03</span><p>Your educational prediction will appear here.</p></div>
            )}
            <p className="disclaimer">{metadata.disclaimer}</p>
          </aside>
        </div>
      )}
    </div>
  )
}
