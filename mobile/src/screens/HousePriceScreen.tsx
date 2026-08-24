import { StyleSheet, Text } from 'react-native'
import { PredictionScreen } from '../components/PredictionScreen'
import type { HouseResult } from '../types'

export function HousePriceScreen() {
  return <PredictionScreen<HouseResult> slug="house-price" endpoint="/api/v1/house-price/predict" eyebrow="System 02 · Regression" title="Vietnam House Price" intro="Use the final eleven-feature raw representation selected by training cross-validation." buttonLabel="Estimate property price" renderResult={(result) => <>
    <Text style={styles.label}>Estimated price</Text><Text style={styles.value}>{result.predicted_price.toFixed(2)}</Text><Text style={styles.unit}>{result.unit}</Text><Text style={styles.model}>{result.model}</Text>
  </>} />
}
const styles = StyleSheet.create({ label: { color: '#a9d8ce', textTransform: 'uppercase', letterSpacing: 1.5, fontSize: 11 }, value: { color: 'white', fontSize: 47, fontWeight: '700', marginTop: 10 }, unit: { color: '#c2ddd7', fontSize: 18 }, model: { color: '#a9d8ce', fontSize: 11, marginTop: 18 } })
