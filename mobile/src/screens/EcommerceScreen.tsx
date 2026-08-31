import { StyleSheet, Text, View } from 'react-native'
import { PredictionScreen } from '../components/PredictionScreen'
import { spacing } from '../theme'
import type { EcommerceResult } from '../types'

export function EcommerceScreen() {
  return <PredictionScreen<EcommerceResult>
    slug="ecommerce"
    endpoint="/api/v1/ecommerce/predict"
    eyebrow="System 03 · Text classification"
    title="Customer Preference"
    intro="Review text becomes TF-IDF and joins five engineered numerical features in the saved pipeline."
    buttonLabel="Predict preference"
    renderResult={(result) => <>
      <Text style={styles.label}>Predicted preference</Text>
      <Text style={styles.value}>{result.label}</Text>
      <View style={styles.row}><Text style={styles.muted}>Positive probability</Text><Text style={styles.probability}>{(result.probability * 100).toFixed(1)}%</Text></View>
      <Text style={styles.disclaimer}>Educational rating-derived preference prediction, not actual customer intention.</Text>
    </>}
  />
}

const styles = StyleSheet.create({
  label: { color: '#a9d8ce', textTransform: 'uppercase', letterSpacing: 1.3, fontSize: 11, fontWeight: '700' },
  value: { color: 'white', fontSize: 30, lineHeight: 36, fontWeight: '700', marginTop: spacing.sm, marginBottom: spacing.md },
  row: { flexDirection: 'row', justifyContent: 'space-between', borderTopColor: 'rgba(255,255,255,0.2)', borderTopWidth: 1, paddingTop: 13 },
  muted: { color: '#c2ddd7', fontSize: 14 },
  probability: { color: 'white', fontSize: 16, fontWeight: '700' },
  disclaimer: { color: '#c2ddd7', fontSize: 12, lineHeight: 18, marginTop: spacing.md },
})
