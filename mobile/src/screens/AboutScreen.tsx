import { ScrollView, StyleSheet, Text, View } from 'react-native'
import { colors, shared, spacing } from '../theme'

const pipeline = ['Input', 'Representation', 'Preprocessing', 'Model', 'Prediction']

export function AboutScreen() {
  return <ScrollView style={shared.screen} contentContainerStyle={shared.content} showsVerticalScrollIndicator={false}>
    <Text style={shared.eyebrow}>Assignment 01</Text>
    <Text style={shared.title}>Intelligent System Development</Text>
    <Text style={shared.intro}>Two fitted machine-learning pipelines presented through one clear prediction workflow.</Text>

    <View style={styles.systemsCard}>
      <SystemRow number="01" name="Diabetes" task="Classification" />
      <View style={styles.divider} />
      <SystemRow number="02" name="House Price" task="Regression" />
    </View>

    <View style={styles.pipelineCard}>
      <Text style={styles.sectionLabel}>Pipeline</Text>
      {pipeline.map((step, index) => <View style={styles.pipelineRow} key={step}>
        <View style={styles.stepNumber}><Text style={styles.stepNumberText}>{index + 1}</Text></View>
        <Text style={styles.stepText}>{step}</Text>
        {index < pipeline.length - 1 ? <Text style={styles.stepArrow}>↓</Text> : null}
      </View>)}
    </View>

    <View style={styles.note}>
      <Text style={styles.noteTitle}>Educational scope</Text>
      <Text style={styles.noteText}>Predictions demonstrate the assignment pipelines. They are not medical advice or a professional property valuation.</Text>
    </View>
  </ScrollView>
}

function SystemRow({ number, name, task }: { number: string; name: string; task: string }) {
  return <View style={styles.systemRow}>
    <Text style={styles.number}>{number}</Text>
    <View style={styles.systemCopy}><Text style={styles.systemName}>{name}</Text><Text style={styles.systemTask}>{task}</Text></View>
  </View>
}

const styles = StyleSheet.create({
  systemsCard: { backgroundColor: colors.surface, borderColor: colors.line, borderWidth: 1, borderRadius: 16, paddingHorizontal: 18, marginBottom: spacing.md },
  systemRow: { minHeight: 70, flexDirection: 'row', alignItems: 'center' },
  number: { width: 34, color: colors.orange, fontSize: 11, fontWeight: '800', letterSpacing: 0.6 },
  systemCopy: { flex: 1 },
  systemName: { color: colors.ink, fontSize: 17, fontWeight: '700' },
  systemTask: { color: colors.muted, fontSize: 13, marginTop: 2 },
  divider: { height: 1, backgroundColor: colors.line, marginLeft: 34 },
  pipelineCard: { backgroundColor: colors.paleGreen, borderRadius: 16, padding: 18, marginBottom: spacing.md },
  sectionLabel: { color: colors.teal, fontSize: 11, fontWeight: '800', letterSpacing: 1.2, textTransform: 'uppercase', marginBottom: spacing.md },
  pipelineRow: { minHeight: 35, flexDirection: 'row', alignItems: 'center' },
  stepNumber: { width: 24, height: 24, borderRadius: 12, alignItems: 'center', justifyContent: 'center', backgroundColor: 'rgba(255,255,255,0.72)' },
  stepNumberText: { color: colors.teal, fontSize: 10, fontWeight: '800' },
  stepText: { color: colors.ink, fontSize: 14, fontWeight: '600', marginLeft: spacing.sm },
  stepArrow: { position: 'absolute', left: 7, top: 25, color: colors.orange, fontSize: 13 },
  note: { borderLeftColor: colors.orange, borderLeftWidth: 3, paddingLeft: 14, paddingVertical: 3 },
  noteTitle: { color: colors.ink, fontSize: 14, fontWeight: '700' },
  noteText: { color: colors.muted, fontSize: 13, lineHeight: 19, marginTop: 4 },
})
