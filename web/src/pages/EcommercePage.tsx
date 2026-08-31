import { PredictionForm } from '../components/PredictionForm'
import type { EcommerceResult } from '../types'

export function EcommercePage() {
  return <PredictionForm<EcommerceResult>
    slug="ecommerce"
    endpoint="/api/v1/ecommerce/predict"
    eyebrow="System 03 · Text classification"
    title="Customer Preference Prediction"
    intro="Enter a review and its helpfulness counts. Review text → TF-IDF → five engineered numerical features → classifier → preference prediction."
    submitLabel="Predict preference"
    renderResult={(result) => <>
      <span className="result-label">Predicted preference</span>
      <h2>{result.label}</h2>
      <div className="probability"><span>Positive-class probability</span><strong>{(result.probability * 100).toFixed(1)}%</strong></div>
      <small>{result.model}</small>
    </>}
  />
}
