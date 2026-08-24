import { Picker } from '@react-native-picker/picker'
import { useEffect, useState, type ReactNode } from 'react'
import { ActivityIndicator, KeyboardAvoidingView, Platform, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native'
import { api } from '../api'
import { colors, shared } from '../theme'
import type { ModelMetadata } from '../types'

type Props<T> = {
  slug: 'diabetes' | 'house-price'
  endpoint: string
  eyebrow: string
  title: string
  intro: string
  buttonLabel: string
  renderResult: (result: T) => ReactNode
}

export function PredictionScreen<T>({ slug, endpoint, eyebrow, title, intro, buttonLabel, renderResult }: Props<T>) {
  const [metadata, setMetadata] = useState<ModelMetadata | null>(null)
  const [values, setValues] = useState<Record<string, string>>({})
  const [result, setResult] = useState<T | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    api.metadata(slug).then((data) => {
      setMetadata(data)
      setValues(Object.fromEntries(data.fields.map((field) => [field.name, String(field.example ?? '')])))
    }).catch((reason: Error) => setError(reason.message))
  }, [slug])

  async function submit() {
    if (!metadata) return
    setLoading(true); setError(''); setResult(null)
    const payload = Object.fromEntries(metadata.fields.map((field) => {
      const raw = values[field.name] ?? ''
      if (field.nullable && raw === '') return [field.name, null]
      return [field.name, field.type === 'number' ? Number(raw) : raw]
    }))
    try { setResult(await api.predict<T>(endpoint, payload)) }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'Prediction failed') }
    finally { setLoading(false) }
  }

  return (
    <KeyboardAvoidingView style={shared.screen} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
      <ScrollView contentContainerStyle={shared.content} keyboardShouldPersistTaps="handled">
        <Text style={shared.eyebrow}>{eyebrow}</Text><Text style={shared.title}>{title}</Text><Text style={shared.intro}>{intro}</Text>
        {!metadata && !error ? <ActivityIndicator color={colors.teal} size="large" /> : null}
        {metadata ? <View style={shared.card}>
          <Text style={shared.cardTitle}>Raw model input · {metadata.fields.length} fields</Text>
          {metadata.fields.map((field) => <View style={styles.field} key={field.name}>
            <Text style={styles.label}>{field.label}{field.unit ? <Text style={styles.unit}> · {field.unit}</Text> : null}</Text>
            {field.type === 'categorical' ? <View style={styles.pickerWrap}><Picker selectedValue={values[field.name] ?? ''} onValueChange={(value) => setValues({ ...values, [field.name]: String(value) })}>
              {field.nullable ? <Picker.Item label="Not provided — pipeline imputes" value="" /> : null}
              {field.options?.map((option) => <Picker.Item key={option} label={option} value={option} />)}
            </Picker></View> : <TextInput style={styles.input} keyboardType="decimal-pad" value={values[field.name] ?? ''} onChangeText={(value) => setValues({ ...values, [field.name]: value })} placeholder={field.nullable ? 'Optional value' : 'Required'} />}
            <Text style={styles.help}>{field.description}</Text>
          </View>)}
          <Pressable accessibilityRole="button" disabled={loading} onPress={submit} style={({ pressed }) => [shared.button, pressed && styles.pressed, loading && styles.disabled]}><Text style={shared.buttonText}>{loading ? 'Running saved pipeline…' : buttonLabel}</Text></Pressable>
        </View> : null}
        {error ? <View style={styles.error}><Text style={styles.errorTitle}>Could not complete request</Text><Text style={styles.errorText}>{error}</Text></View> : null}
        {result ? <View style={styles.result}>{renderResult(result)}</View> : null}
        {metadata ? <Text style={shared.disclaimer}>{metadata.disclaimer}</Text> : null}
      </ScrollView>
    </KeyboardAvoidingView>
  )
}

const styles = StyleSheet.create({
  field: { marginBottom: 18 }, label: { color: colors.ink, fontWeight: '700', marginBottom: 7 }, unit: { color: colors.muted, fontWeight: '400' },
  input: { borderColor: '#cbd5ce', borderWidth: 1, borderRadius: 9, paddingHorizontal: 13, paddingVertical: 12, backgroundColor: 'white', color: colors.ink, fontSize: 16 },
  pickerWrap: { borderColor: '#cbd5ce', borderWidth: 1, borderRadius: 9, overflow: 'hidden', backgroundColor: 'white' }, help: { color: colors.muted, fontSize: 11, marginTop: 5 },
  pressed: { opacity: .85 }, disabled: { opacity: .65 }, error: { backgroundColor: '#f8e4df', borderColor: '#ebc2b8', borderWidth: 1, borderRadius: 13, padding: 16, marginBottom: 16 },
  errorTitle: { color: colors.error, fontWeight: '700', marginBottom: 5 }, errorText: { color: colors.error, lineHeight: 20 }, result: { backgroundColor: colors.teal, borderRadius: 18, padding: 22, marginBottom: 12 },
})
