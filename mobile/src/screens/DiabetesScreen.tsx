import { StyleSheet, Text, View } from 'react-native'
import { PredictionScreen } from '../components/PredictionScreen'
import { colors, spacing } from '../theme'
import type { DiabetesResult } from '../types'

export function DiabetesScreen() {
  return <PredictionScreen<DiabetesResult>
    slug="diabetes"
    endpoint="/api/v1/diabetes/predict"
    eyebrow="System 01 · Classification"
    title="Diabetes Prediction"
    intro="Enter six health indicators to run the trained classification pipeline."
    buttonLabel="Generate prediction"
    renderResult={(result) => <>
      <Text style={styles.label}>Prediction</Text>
      <Text style={styles.value}>{result.label}</Text>
      {result.probability !== null ? <View style={styles.row}>
        <Text style={styles.muted}>Diabetic probability</Text>
        <Text style={styles.probability}>{(result.probability * 100).toFixed(1)}%</Text>
      </View> : null}
      <Text style={styles.model}>{result.model}</Text>
      <Text style={styles.disclaimer}>Educational demonstration only. Not a medical diagnosis.</Text>
    </>}
  />
}

const styles = StyleSheet.create({
  label: { color: '#a9d8ce', textTransform: 'uppercase', letterSpacing: 1.3, fontSize: 11, fontWeight: '700' },
  value: { color: 'white', fontSize: 30, lineHeight: 36, fontWeight: '700', marginTop: spacing.sm, marginBottom: spacing.md },
  row: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', borderTopColor: 'rgba(255,255,255,0.2)', borderTopWidth: 1, paddingTop: 13 },
  muted: { color: '#c2ddd7', fontSize: 14 },
  probability: { color: 'white', fontSize: 16, fontWeight: '700' },
  model: { alignSelf: 'flex-start', color: colors.ink, backgroundColor: colors.paleGreen, overflow: 'hidden', borderRadius: 10, paddingHorizontal: 9, paddingVertical: 4, fontSize: 10, fontWeight: '700', marginTop: spacing.md },
  disclaimer: { color: '#c2ddd7', fontSize: 12, lineHeight: 18, marginTop: spacing.md },
})
