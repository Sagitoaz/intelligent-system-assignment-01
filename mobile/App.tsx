import { useState } from 'react'
import { Pressable, SafeAreaView, StatusBar, StyleSheet, Text, View } from 'react-native'
import { AboutScreen } from './src/screens/AboutScreen'
import { DiabetesScreen } from './src/screens/DiabetesScreen'
import { HomeScreen } from './src/screens/HomeScreen'
import { HousePriceScreen } from './src/screens/HousePriceScreen'
import { colors } from './src/theme'

type Screen = 'home' | 'diabetes' | 'house' | 'about'
const labels: Record<Screen, string> = { home: 'Home', diabetes: 'Diabetes', house: 'House price', about: 'About' }

export default function App() {
  const [screen, setScreen] = useState<Screen>('home')
  return <SafeAreaView style={styles.safe}><StatusBar barStyle="dark-content" backgroundColor="#f7f8f4" />
    <View style={styles.header}><View style={styles.mark}><Text style={styles.markText}>IS</Text></View><View><Text style={styles.brand}>Intelligent Systems</Text><Text style={styles.assignment}>Assignment 01</Text></View></View>
    <View style={styles.content}>{screen === 'home' ? <HomeScreen navigate={setScreen} /> : screen === 'diabetes' ? <DiabetesScreen /> : screen === 'house' ? <HousePriceScreen /> : <AboutScreen />}</View>
    <View style={styles.nav}>{(Object.keys(labels) as Screen[]).map((key) => <Pressable key={key} onPress={() => setScreen(key)} style={styles.navItem}><View style={[styles.dot, screen === key && styles.activeDot]} /><Text style={[styles.navText, screen === key && styles.activeText]}>{labels[key]}</Text></Pressable>)}</View>
  </SafeAreaView>
}
const styles = StyleSheet.create({ safe: { flex: 1, backgroundColor: '#f7f8f4', paddingTop: StatusBar.currentHeight ?? 0 }, header: { height: 70, paddingHorizontal: 20, flexDirection: 'row', alignItems: 'center', borderBottomColor: colors.line, borderBottomWidth: 1 }, mark: { width: 38, height: 38, borderRadius: 11, backgroundColor: colors.teal, alignItems: 'center', justifyContent: 'center', marginRight: 10 }, markText: { color: 'white', fontWeight: '700', fontSize: 12, letterSpacing: 1 }, brand: { color: colors.ink, fontWeight: '700', fontSize: 13 }, assignment: { color: colors.muted, fontSize: 10, marginTop: 2 }, content: { flex: 1 }, nav: { minHeight: 68, backgroundColor: '#fbfcf9', borderTopColor: colors.line, borderTopWidth: 1, flexDirection: 'row', justifyContent: 'space-around', paddingBottom: 6 }, navItem: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 5 }, dot: { width: 5, height: 5, borderRadius: 3, backgroundColor: 'transparent' }, activeDot: { backgroundColor: colors.orange }, navText: { color: colors.muted, fontSize: 10 }, activeText: { color: colors.teal, fontWeight: '700' } })
