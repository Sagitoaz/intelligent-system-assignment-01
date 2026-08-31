import { PredictionForm } from '../components/PredictionForm'
import type { DiabetesResult } from '../types'

export function DiabetesPage() {
  return (
    <PredictionForm<DiabetesResult>
      slug="diabetes"
      endpoint="/api/v1/diabetes/predict"
      eyebrow="System 01 · Classification"
      title="Diabetes Prediction"
      intro="Raw patient values → median imputation → standardization → Random Forest → educational prediction."
      submitLabel="Generate prediction"
      renderResult={(result) => (
        <>
          <span className="result-label">Prediction</span>
          <h2>{result.label}</h2>
          {result.probability !== null && (
            <div className="probability"><span>Class 1 probability</span><strong>{(result.probability * 100).toFixed(1)}%</strong></div>
          )}
          <small>{result.model}</small>
        </>
      )}
    />
  )
}
