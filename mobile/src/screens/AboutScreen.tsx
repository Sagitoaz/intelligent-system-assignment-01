import { ScrollView, StyleSheet, Text, View } from 'react-native'
import { API_BASE_URL } from '../api'
import { colors, shared } from '../theme'

export function AboutScreen() {
  return <ScrollView style={shared.screen} contentContainerStyle={shared.content}><Text style={shared.eyebrow}>System documentation</Text><Text style={shared.title}>About the assignment</Text><Text style={shared.intro}>Training and inference use the same saved scikit-learn Pipelines.</Text>
    {[
      ['Training source', 'The notebooks define representations, controlled experiments, model selection and evaluation.'],
      ['Inference', 'FastAPI loads trusted local joblib files once and never refits them during requests.'],
      ['Knowledge graph', 'Neo4j connects the Diabetes system, model, features, target, preprocessing, experiment and metrics.'],
      ['Limitations', 'Educational use only. No medical diagnosis or professional property valuation.'],
    ].map(([title, body], index) => <View style={shared.card} key={title}><Text style={styles.number}>0{index + 1}</Text><Text style={styles.cardTitle}>{title}</Text><Text style={styles.body}>{body}</Text></View>)}
    <View style={styles.endpoint}><Text style={styles.endpointLabel}>Current API base URL</Text><Text style={styles.endpointValue}>{API_BASE_URL || 'Not configured'}</Text></View>
  </ScrollView>
}
const styles = StyleSheet.create({ number: { color: colors.orange, fontSize: 11, fontWeight: '700' }, cardTitle: { color: colors.ink, fontWeight: '700', fontSize: 20, marginTop: 12 }, body: { color: colors.muted, lineHeight: 22, marginTop: 8 }, endpoint: { backgroundColor: colors.teal, padding: 18, borderRadius: 14 }, endpointLabel: { color: '#a9d8ce', fontSize: 11, textTransform: 'uppercase' }, endpointValue: { color: 'white', marginTop: 7 } })
