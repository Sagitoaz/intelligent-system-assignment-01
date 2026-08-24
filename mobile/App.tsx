import { StatusBar } from 'expo-status-bar'
import { useState } from 'react'
import { Pressable, StyleSheet, Text, View } from 'react-native'
import { SafeAreaProvider, useSafeAreaInsets } from 'react-native-safe-area-context'
import { AboutScreen } from './src/screens/AboutScreen'
import { DiabetesScreen } from './src/screens/DiabetesScreen'
import { HomeScreen } from './src/screens/HomeScreen'
import { HousePriceScreen } from './src/screens/HousePriceScreen'
import { colors } from './src/theme'

type Screen = 'home' | 'diabetes' | 'house' | 'about'
const tabs: { key: Screen; label: string; mark: string }[] = [
  { key: 'home', label: 'Home', mark: 'H' },
  { key: 'diabetes', label: 'Diabetes', mark: 'D' },
  { key: 'house', label: 'House price', mark: 'P' },
  { key: 'about', label: 'About', mark: 'i' },
]

export default function App() {
  return <SafeAreaProvider><AppLayout /></SafeAreaProvider>
}

function AppLayout() {
  const [screen, setScreen] = useState<Screen>('home')
  const insets = useSafeAreaInsets()
  const bottomInset = Math.max(insets.bottom, 8)

  return <View style={styles.app}>
    <StatusBar style="dark" backgroundColor={colors.surface} />
    <View style={[styles.header, { paddingTop: insets.top }]}>
      <View style={styles.mark}><Text style={styles.markText}>IS</Text></View>
      <View><Text style={styles.brand}>Intelligent Systems</Text><Text style={styles.assignment}>Assignment 01</Text></View>
    </View>
    <View style={styles.content}>
      {screen === 'home' ? <HomeScreen navigate={setScreen} /> : screen === 'diabetes' ? <DiabetesScreen /> : screen === 'house' ? <HousePriceScreen /> : <AboutScreen />}
    </View>
    <View style={[styles.nav, { height: 56 + bottomInset, paddingBottom: bottomInset }]}>
      {tabs.map((tab) => {
        const active = screen === tab.key
        return <Pressable
          accessibilityRole="tab"
          accessibilityState={{ selected: active }}
          accessibilityLabel={tab.label}
          key={tab.key}
          onPress={() => setScreen(tab.key)}
          style={({ pressed }) => [styles.navItem, pressed && styles.pressed]}
        >
          <View style={[styles.tabMark, active && styles.activeTabMark]}><Text style={[styles.tabMarkText, active && styles.activeTabMarkText]}>{tab.mark}</Text></View>
          <Text numberOfLines={1} style={[styles.navText, active && styles.activeText]}>{tab.label}</Text>
          <View style={[styles.activeIndicator, active && styles.activeIndicatorVisible]} />
        </Pressable>
      })}
    </View>
  </View>
}

const styles = StyleSheet.create({
  app: { flex: 1, backgroundColor: colors.surface },
  header: { minHeight: 54, paddingHorizontal: 22, paddingBottom: 9, flexDirection: 'row', alignItems: 'flex-end', borderBottomColor: colors.line, borderBottomWidth: 1 },
  mark: { width: 34, height: 34, borderRadius: 10, backgroundColor: colors.teal, alignItems: 'center', justifyContent: 'center', marginRight: 10 },
  markText: { color: 'white', fontWeight: '800', fontSize: 11, letterSpacing: 1 },
  brand: { color: colors.ink, fontWeight: '700', fontSize: 13 },
  assignment: { color: colors.muted, fontSize: 10, marginTop: 1 },
  content: { flex: 1, minHeight: 0 },
  nav: { flexDirection: 'row', backgroundColor: '#fbfcf9', borderTopColor: colors.line, borderTopWidth: 1, paddingTop: 6 },
  navItem: { flex: 1, minWidth: 0, alignItems: 'center', justifyContent: 'center', gap: 2 },
  tabMark: { width: 21, height: 21, borderRadius: 7, alignItems: 'center', justifyContent: 'center', borderColor: colors.line, borderWidth: 1 },
  activeTabMark: { backgroundColor: colors.teal, borderColor: colors.teal },
  tabMarkText: { color: colors.muted, fontSize: 9, fontWeight: '800' },
  activeTabMarkText: { color: 'white' },
  navText: { color: colors.muted, fontSize: 10, lineHeight: 14 },
  activeText: { color: colors.teal, fontWeight: '700' },
  activeIndicator: { width: 18, height: 2, borderRadius: 1, backgroundColor: 'transparent', marginTop: 1 },
  activeIndicatorVisible: { backgroundColor: colors.orange },
  pressed: { opacity: 0.65 },
})
