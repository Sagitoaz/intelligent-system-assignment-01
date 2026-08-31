import { Pressable, ScrollView, StyleSheet, Text, useWindowDimensions, View } from 'react-native'
import { colors, shared, spacing } from '../theme'

export function HomeScreen({ navigate }: { navigate: (screen: 'diabetes' | 'house' | 'ecommerce' | 'about') => void }) {
  const { width } = useWindowDimensions()
  const compact = width <= 375

  return <ScrollView
    style={shared.screen}
    contentContainerStyle={[shared.content, styles.content]}
    showsVerticalScrollIndicator={false}
  >
    <View style={styles.heroBlock}>
      <Text style={shared.eyebrow}>Assignment 02</Text>
      <Text style={[styles.hero, compact && styles.heroCompact]}>Intelligent Systems</Text>
      <Text style={styles.intro}>Three trained ML systems, one transparent prediction workflow.</Text>
    </View>
    <SystemCard number="01" task="Classification" title="Diabetes Prediction" description="6 raw features · Random Forest" tone="teal" onPress={() => navigate('diabetes')} />
    <SystemCard number="02" task="Regression" title="Vietnam House Price" description="11 raw features · Random Forest" tone="orange" onPress={() => navigate('house')} />
    <SystemCard number="03" task="Text classification" title="Customer Preference" description="TF-IDF review text + 5 behavioral features" tone="violet" onPress={() => navigate('ecommerce')} />
    <Pressable accessibilityRole="link" onPress={() => navigate('about')} style={({ pressed }) => [styles.aboutLink, pressed && styles.pressed]}>
      <Text style={styles.aboutText}>View assignment methodology</Text><Text style={styles.arrow}>→</Text>
    </Pressable>
  </ScrollView>
}

function SystemCard({ number, task, title, description, tone, onPress }: { number: string; task: string; title: string; description: string; tone: 'teal' | 'orange' | 'violet'; onPress: () => void }) {
  return <Pressable
    accessibilityRole="button"
    accessibilityLabel={`Open ${title}`}
    onPress={onPress}
    style={({ pressed }) => [styles.systemCard, tone === 'teal' ? styles.teal : tone === 'orange' ? styles.orange : styles.violet, pressed && styles.pressed]}
  >
    <View style={styles.cardTop}><Text style={styles.number}>{number}</Text><Text style={styles.task}>{task}</Text></View>
    <Text style={styles.cardTitle}>{title}</Text>
    <Text style={styles.description}>{description}</Text>
    <View style={styles.openRow}><Text style={styles.open}>Open system</Text><Text style={styles.openArrow}>→</Text></View>
  </Pressable>
}

const styles = StyleSheet.create({
  content: { paddingTop: 20 },
  heroBlock: { marginBottom: 22 },
  hero: { color: colors.ink, fontSize: 35, lineHeight: 40, fontWeight: '700', letterSpacing: -1.2, marginTop: 6 },
  heroCompact: { fontSize: 32, lineHeight: 37 },
  intro: { color: colors.muted, fontSize: 16, lineHeight: 23, marginTop: 8, maxWidth: 350 },
  systemCard: { minHeight: 164, borderRadius: 18, padding: 20, marginBottom: 14 },
  teal: { backgroundColor: colors.paleGreen },
  orange: { backgroundColor: colors.palePeach },
  violet: { backgroundColor: '#eae5f2' },
  cardTop: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  number: { color: colors.muted, fontSize: 12, fontWeight: '700', letterSpacing: 0.5 },
  task: { overflow: 'hidden', backgroundColor: 'rgba(255,255,255,0.68)', borderRadius: 14, paddingHorizontal: 10, paddingVertical: 5, color: colors.ink, fontSize: 10, fontWeight: '700', textTransform: 'uppercase', letterSpacing: 0.85 },
  cardTitle: { color: colors.ink, fontSize: 25, lineHeight: 30, fontWeight: '700', letterSpacing: -0.5, marginTop: 15 },
  description: { color: colors.muted, fontSize: 14, lineHeight: 20, marginTop: 4 },
  openRow: { flexDirection: 'row', alignItems: 'center', marginTop: 'auto' },
  open: { color: colors.teal, fontSize: 14, fontWeight: '700' },
  openArrow: { color: colors.orange, fontSize: 18, marginLeft: spacing.xs, marginTop: -1 },
  aboutLink: { minHeight: 44, flexDirection: 'row', alignItems: 'center', justifyContent: 'center' },
  aboutText: { color: colors.teal, fontSize: 14, fontWeight: '700' },
  arrow: { color: colors.orange, fontSize: 18, marginLeft: spacing.xs },
  pressed: { opacity: 0.78 },
})
