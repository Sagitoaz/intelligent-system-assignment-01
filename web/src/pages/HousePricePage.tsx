import { PredictionForm } from '../components/PredictionForm'
import type { HouseResult } from '../types'

export function HousePricePage() {
  return (
    <PredictionForm<HouseResult>
      slug="house-price"
      endpoint="/api/v1/house-price/predict"
      eyebrow="System 02 · Regression"
      title="Vietnam House Price Prediction"
      intro="Raw property attributes → missing handling → numerical scaling and categorical one-hot encoding → Random Forest → educational price estimate."
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
