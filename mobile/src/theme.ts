import { StyleSheet } from 'react-native'

export const colors = {
  ink: '#101828',
  muted: '#3d4c63',
  surface: '#ffffff',
  canvas: '#eaf3fc',
  line: '#b9cde0',
  teal: '#0d9f8f',
  orange: '#f0603a',
  paleGreen: '#d3f3ee',
  palePeach: '#ffe6d9',
  error: '#a8311a',
}

export const spacing = { xs: 6, sm: 10, md: 16, lg: 22, xl: 28, xxl: 36 }

export const shared = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.canvas },
  content: { paddingHorizontal: spacing.lg, paddingTop: spacing.lg, paddingBottom: spacing.xxl },
  eyebrow: { color: colors.teal, fontSize: 12, fontWeight: '700', letterSpacing: 1.35, textTransform: 'uppercase' },
  title: { color: colors.ink, fontSize: 34, fontWeight: '700', letterSpacing: -1, lineHeight: 39, marginTop: 7 },
  intro: { color: colors.muted, fontSize: 15, lineHeight: 22, marginTop: 9, marginBottom: spacing.lg },
  card: { backgroundColor: colors.surface, borderColor: colors.line, borderWidth: 1, borderRadius: 16, padding: spacing.lg, marginBottom: spacing.md },
  cardTitle: { color: colors.ink, fontSize: 19, fontWeight: '700' },
  button: { minHeight: 50, backgroundColor: colors.teal, borderRadius: 12, paddingHorizontal: spacing.md, alignItems: 'center', justifyContent: 'center', marginTop: spacing.xs },
  buttonText: { color: 'white', fontWeight: '700', fontSize: 15 },
  disclaimer: { color: colors.muted, fontSize: 12, lineHeight: 18 },
})
