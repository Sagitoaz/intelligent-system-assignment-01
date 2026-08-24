import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native'
import { colors, shared } from '../theme'

export function HomeScreen({ navigate }: { navigate: (screen: 'diabetes' | 'house' | 'about') => void }) {
  return <ScrollView style={shared.screen} contentContainerStyle={shared.content}>
    <Text style={shared.eyebrow}>Assignment 01 · Intelligent systems</Text><Text style={styles.hero}>Models made{`\n`}transparent.</Text>
    <Text style={shared.intro}>Run both fitted scikit-learn pipelines from one mobile interface connected to the local FastAPI backend.</Text>
    <SystemCard number="01" task="Classification" title="Diabetes Prediction" description="Six raw inputs. Educational prediction only — never a diagnosis." tone="teal" onPress={() => navigate('diabetes')} />
    <SystemCard number="02" task="Regression" title="Vietnam House Price" description="Eleven raw property features. Estimate shown in billion VND." tone="orange" onPress={() => navigate('house')} />
    <Pressable onPress={() => navigate('about')}><Text style={styles.aboutLink}>Read system methodology →</Text></Pressable>
  </ScrollView>
}

function SystemCard({ number, task, title, description, tone, onPress }: { number: string; task: string; title: string; description: string; tone: 'teal' | 'orange'; onPress: () => void }) {
  return <Pressable onPress={onPress} style={({ pressed }) => [styles.systemCard, tone === 'teal' ? styles.teal : styles.orange, pressed && { opacity: .85 }]}>
    <View style={styles.cardTop}><Text style={styles.number}>{number}</Text><Text style={styles.task}>{task}</Text></View><Text style={styles.cardTitle}>{title}</Text><Text style={styles.description}>{description}</Text><Text style={styles.open}>Open system  →</Text>
  </Pressable>
}

const styles = StyleSheet.create({ hero: { color: colors.ink, fontSize: 47, lineHeight: 50, fontWeight: '700', letterSpacing: -1.7, marginTop: 14 }, systemCard: { minHeight: 245, borderRadius: 20, padding: 22, marginBottom: 16 }, teal: { backgroundColor: '#dcebe5' }, orange: { backgroundColor: '#f5e3d1' }, cardTop: { flexDirection: 'row', justifyContent: 'space-between' }, number: { color: colors.muted, fontSize: 12, fontWeight: '700' }, task: { backgroundColor: 'rgba(255,255,255,.65)', borderRadius: 20, paddingHorizontal: 10, paddingVertical: 6, fontSize: 10, textTransform: 'uppercase', letterSpacing: 1 }, cardTitle: { color: colors.ink, fontSize: 29, lineHeight: 34, fontWeight: '700', marginTop: 28 }, description: { color: colors.muted, lineHeight: 21, marginTop: 10 }, open: { color: colors.teal, fontWeight: '700', marginTop: 'auto' }, aboutLink: { color: colors.teal, fontWeight: '700', paddingVertical: 15 } })
