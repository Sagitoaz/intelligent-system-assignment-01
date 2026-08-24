import { StyleSheet, Text } from 'react-native'
import { PredictionScreen, type FieldGroup } from '../components/PredictionScreen'
import { colors, spacing } from '../theme'
import type { HouseResult } from '../types'

const fieldGroups: FieldGroup[] = [
  { title: 'Location', fields: ['Province'] },
  { title: 'Property dimensions', fields: ['Area', 'Frontage', 'Access Road', 'Floors'] },
  { title: 'Rooms', fields: ['Bedrooms', 'Bathrooms'] },
  { title: 'Property details', fields: ['House direction', 'Balcony direction', 'Legal status', 'Furniture state'] },
]

export function HousePriceScreen() {
  return <PredictionScreen<HouseResult>
    slug="house-price"
    endpoint="/api/v1/house-price/predict"
    eyebrow="System 02 · Regression"
    title="Vietnam House Price"
    intro="Describe the property to estimate its price with the trained regression pipeline."
    buttonLabel="Estimate property price"
    fieldGroups={fieldGroups}
    renderResult={(result) => <>
      <Text style={styles.label}>Estimated price</Text>
      <Text style={styles.value}>{result.predicted_price.toFixed(2)}</Text>
      <Text style={styles.unit}>{result.unit}</Text>
      <Text style={styles.model}>{result.model}</Text>
      <Text style={styles.disclaimer}>Educational estimate only. Not a professional valuation.</Text>
    </>}
  />
}

const styles = StyleSheet.create({
  label: { color: '#a9d8ce', textTransform: 'uppercase', letterSpacing: 1.3, fontSize: 11, fontWeight: '700' },
  value: { color: 'white', fontSize: 39, lineHeight: 45, fontWeight: '700', letterSpacing: -1, marginTop: spacing.xs },
  unit: { color: '#c2ddd7', fontSize: 17, marginTop: 1 },
  model: { alignSelf: 'flex-start', color: colors.ink, backgroundColor: colors.paleGreen, overflow: 'hidden', borderRadius: 10, paddingHorizontal: 9, paddingVertical: 4, fontSize: 10, fontWeight: '700', marginTop: spacing.md },
  disclaimer: { color: '#c2ddd7', fontSize: 12, lineHeight: 18, marginTop: spacing.md },
})
