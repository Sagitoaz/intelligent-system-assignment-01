import { PredictionForm } from '../components/PredictionForm'
import type { HouseResult } from '../types'

export function HousePricePage() {
  return (
    <PredictionForm<HouseResult>
      slug="house-price"
      endpoint="/api/v1/house-price/predict"
      eyebrow="System 02 · Regression"
      title="Vietnam House Price Prediction"
      intro="Describe a property using the final eleven-feature representation selected in the notebook experiments."
      submitLabel="Estimate property price"
      renderResult={(result) => (
        <>
          <span className="result-label">Estimated price</span>
          <h2>{result.predicted_price.toFixed(2)}</h2>
          <div className="result-unit">{result.unit}</div>
          <small>{result.model}</small>
        </>
      )}
    />
  )
}
