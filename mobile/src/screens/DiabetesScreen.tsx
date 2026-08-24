import { StyleSheet, Text, View } from 'react-native'
import { PredictionScreen } from '../components/PredictionScreen'
import type { DiabetesResult } from '../types'

export function DiabetesScreen() {
  return <PredictionScreen<DiabetesResult> slug="diabetes" endpoint="/api/v1/diabetes/predict" eyebrow="System 01 · Classification" title="Diabetes Prediction" intro="Enter six raw values; fitted preprocessing remains inside the saved pipeline." buttonLabel="Generate prediction" renderResult={(result) => <>
    <Text style={styles.label}>Prediction</Text><Text style={styles.value}>{result.label}</Text>{result.probability !== null ? <View style={styles.row}><Text style={styles.muted}>Class 1 probability</Text><Text style={styles.probability}>{(result.probability * 100).toFixed(1)}%</Text></View> : null}<Text style={styles.model}>{result.model}</Text>
  </>} />
}
const styles = StyleSheet.create({ label: { color: '#a9d8ce', textTransform: 'uppercase', letterSpacing: 1.5, fontSize: 11 }, value: { color: 'white', fontSize: 33, fontWeight: '700', marginVertical: 12 }, row: { flexDirection: 'row', justifyContent: 'space-between', borderTopColor: 'rgba(255,255,255,.2)', borderTopWidth: 1, paddingTop: 13 }, muted: { color: '#c2ddd7' }, probability: { color: 'white', fontWeight: '700' }, model: { color: '#a9d8ce', fontSize: 11, marginTop: 18 } })
