import { StyleSheet } from 'react-native'

export const colors = { ink: '#18312c', muted: '#64736e', surface: '#f7f8f4', line: '#d9e0da', teal: '#1d5a54', orange: '#ee9b5a', pale: '#e1eee8', error: '#8f382a' }

export const shared = StyleSheet.create({
  screen: { flex: 1, backgroundColor: '#f2f4ef' },
  content: { paddingHorizontal: 20, paddingTop: 28, paddingBottom: 110 },
  eyebrow: { color: colors.teal, fontSize: 11, fontWeight: '700', letterSpacing: 1.5, textTransform: 'uppercase' },
  title: { color: colors.ink, fontSize: 38, fontWeight: '700', letterSpacing: -1.2, lineHeight: 43, marginTop: 10 },
  intro: { color: colors.muted, fontSize: 16, lineHeight: 24, marginTop: 12, marginBottom: 28 },
  card: { backgroundColor: colors.surface, borderColor: colors.line, borderWidth: 1, borderRadius: 18, padding: 20, marginBottom: 16 },
  cardTitle: { color: colors.ink, fontSize: 20, fontWeight: '700', marginBottom: 16 },
  button: { backgroundColor: colors.teal, borderRadius: 11, paddingVertical: 15, alignItems: 'center', marginTop: 8 },
  buttonText: { color: 'white', fontWeight: '700', fontSize: 15 },
  disclaimer: { color: colors.muted, fontSize: 12, lineHeight: 18, marginTop: 4 },
})
