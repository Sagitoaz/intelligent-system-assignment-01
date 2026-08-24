import { Picker } from '@react-native-picker/picker'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { ActivityIndicator, findNodeHandle, KeyboardAvoidingView, Platform, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native'
import { api } from '../api'
import { colors, shared, spacing } from '../theme'
import type { FieldMetadata, ModelMetadata } from '../types'

export type FieldGroup = { title: string; fields: string[] }

type Props<T> = {
  slug: 'diabetes' | 'house-price'
  endpoint: string
  eyebrow: string
  title: string
  intro: string
  buttonLabel: string
  fieldGroups?: FieldGroup[]
  renderResult: (result: T) => ReactNode
}

export function PredictionScreen<T>({ slug, endpoint, eyebrow, title, intro, buttonLabel, fieldGroups, renderResult }: Props<T>) {
  const [metadata, setMetadata] = useState<ModelMetadata | null>(null)
  const [values, setValues] = useState<Record<string, string>>({})
  const [result, setResult] = useState<T | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const scrollRef = useRef<ScrollView>(null)
  const inputRefs = useRef<Record<string, TextInput | null>>({})

  useEffect(() => {
    api.metadata(slug).then((data) => {
      setMetadata(data)
      setValues(Object.fromEntries(data.fields.map((field) => [field.name, String(field.example ?? '')])))
    }).catch((reason: Error) => setError(reason.message))
  }, [slug])

  async function submit() {
    if (!metadata) return
    setLoading(true)
    setError('')
    setResult(null)
    const payload = Object.fromEntries(metadata.fields.map((field) => {
      const raw = values[field.name] ?? ''
      if (field.nullable && raw === '') return [field.name, null]
      return [field.name, field.type === 'number' ? Number(raw) : raw]
    }))
    try {
      const response = await api.predict<T>(endpoint, payload)
      setResult(response)
      setTimeout(() => scrollRef.current?.scrollToEnd({ animated: true }), 120)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  function keepInputVisible(name: string) {
    setTimeout(() => {
      const handle = findNodeHandle(inputRefs.current[name])
      if (handle) scrollRef.current?.scrollResponderScrollNativeHandleToKeyboard(handle, 116, true)
    }, 180)
  }

  const sections = metadata ? createSections(metadata.fields, fieldGroups) : []

  return <KeyboardAvoidingView style={shared.screen} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
    <ScrollView
      ref={scrollRef}
      style={shared.screen}
      contentContainerStyle={shared.content}
      keyboardDismissMode="on-drag"
      keyboardShouldPersistTaps="handled"
      automaticallyAdjustKeyboardInsets={Platform.OS === 'ios'}
      showsVerticalScrollIndicator={false}
    >
      <Text style={shared.eyebrow}>{eyebrow}</Text>
      <Text style={shared.title}>{title}</Text>
      <Text style={shared.intro}>{intro}</Text>

      {!metadata && !error ? <View style={styles.loading}><ActivityIndicator color={colors.teal} size="large" /><Text style={styles.loadingText}>Loading model fields…</Text></View> : null}

      {metadata ? <>
        <View style={styles.formHeading}>
          <Text style={styles.formTitle}>Model inputs</Text>
          <Text style={styles.fieldCount}>{metadata.fields.length} fields</Text>
        </View>
        {sections.map((section) => <View style={styles.sectionCard} key={section.title}>
          <Text style={styles.sectionTitle}>{section.title}</Text>
          {section.fields.map((field, index) => <Field
            field={field}
            isLast={index === section.fields.length - 1}
            key={field.name}
            value={values[field.name] ?? ''}
            setInputRef={(input) => { inputRefs.current[field.name] = input }}
            onFocus={() => keepInputVisible(field.name)}
            onChange={(value) => setValues((current) => ({ ...current, [field.name]: value }))}
          />)}
        </View>)}
        <Pressable accessibilityRole="button" disabled={loading} onPress={submit} style={({ pressed }) => [shared.button, pressed && styles.pressed, loading && styles.disabled]}>
          {loading ? <View style={styles.buttonLoading}><ActivityIndicator color="white" size="small" /><Text style={shared.buttonText}>Running saved pipeline…</Text></View> : <Text style={shared.buttonText}>{buttonLabel}</Text>}
        </Pressable>
        {!result ? <Text style={[shared.disclaimer, styles.formDisclaimer]}>{metadata.disclaimer}</Text> : null}
      </> : null}

      {error ? <View accessibilityRole="alert" style={styles.error}><Text style={styles.errorTitle}>Could not complete request</Text><Text style={styles.errorText}>{error}</Text></View> : null}
      {result ? <View style={styles.result}>{renderResult(result)}</View> : null}
    </ScrollView>
  </KeyboardAvoidingView>
}

function Field({ field, value, isLast, onChange, onFocus, setInputRef }: {
  field: FieldMetadata
  value: string
  isLast: boolean
  onChange: (value: string) => void
  onFocus: () => void
  setInputRef: (input: TextInput | null) => void
}) {
  return <View style={[styles.field, isLast && styles.lastField]}>
    <View style={styles.labelRow}>
      <Text style={styles.label}>{field.label}{field.unit ? <Text style={styles.unit}> · {field.unit}</Text> : null}</Text>
      {field.nullable ? <Text style={styles.optional}>Optional</Text> : null}
    </View>
    {field.type === 'categorical' ? <View style={styles.pickerWrap}>
      <Picker style={styles.picker} selectedValue={value} onValueChange={(nextValue) => onChange(String(nextValue))}>
        {field.nullable ? <Picker.Item label="Leave blank" value="" color={colors.muted} /> : null}
        {field.options?.map((option) => <Picker.Item key={option} label={option} value={option} />)}
      </Picker>
    </View> : <TextInput
      ref={setInputRef}
      accessibilityLabel={field.label}
      style={styles.input}
      keyboardType={field.integer ? 'number-pad' : 'decimal-pad'}
      value={value}
      onChangeText={onChange}
      onFocus={onFocus}
      placeholder={field.nullable ? 'Leave blank if unknown' : 'Required'}
      placeholderTextColor="#8a9691"
      selectTextOnFocus
    />}
    <Text style={styles.help}>{field.description}</Text>
  </View>
}

function createSections(fields: FieldMetadata[], groups?: FieldGroup[]) {
  if (!groups) return [{ title: 'Health indicators', fields }]
  const fieldsByName = new Map(fields.map((field) => [field.name, field]))
  const groupedNames = new Set(groups.flatMap((group) => group.fields))
  const sections = groups.map((group) => ({ title: group.title, fields: group.fields.map((name) => fieldsByName.get(name)).filter((field): field is FieldMetadata => Boolean(field)) }))
  const remaining = fields.filter((field) => !groupedNames.has(field.name))
  return remaining.length ? [...sections, { title: 'Additional details', fields: remaining }] : sections
}

const styles = StyleSheet.create({
  loading: { alignItems: 'center', paddingVertical: spacing.xl },
  loadingText: { color: colors.muted, fontSize: 13, marginTop: spacing.sm },
  formHeading: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: spacing.sm },
  formTitle: { color: colors.ink, fontSize: 18, fontWeight: '700' },
  fieldCount: { color: colors.muted, fontSize: 12, fontWeight: '600' },
  sectionCard: { backgroundColor: colors.surface, borderColor: colors.line, borderWidth: 1, borderRadius: 16, padding: 18, marginBottom: 12 },
  sectionTitle: { color: colors.teal, fontSize: 12, fontWeight: '800', letterSpacing: 1, textTransform: 'uppercase', marginBottom: spacing.md },
  field: { marginBottom: 16 },
  lastField: { marginBottom: 0 },
  labelRow: { minHeight: 21, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: spacing.xs },
  label: { flex: 1, color: colors.ink, fontSize: 14, fontWeight: '700', paddingRight: spacing.sm },
  unit: { color: colors.muted, fontWeight: '400' },
  optional: { color: colors.teal, backgroundColor: colors.paleGreen, borderRadius: 9, paddingHorizontal: 7, paddingVertical: 2, fontSize: 10, fontWeight: '700', textTransform: 'uppercase' },
  input: { minHeight: 48, borderColor: '#cbd5ce', borderWidth: 1, borderRadius: 10, paddingHorizontal: 13, paddingVertical: 10, backgroundColor: 'white', color: colors.ink, fontSize: 16 },
  pickerWrap: { minHeight: 50, borderColor: '#cbd5ce', borderWidth: 1, borderRadius: 10, overflow: 'hidden', backgroundColor: 'white', justifyContent: 'center' },
  picker: { color: colors.ink, minHeight: 50 },
  help: { color: colors.muted, fontSize: 11, lineHeight: 16, marginTop: 4 },
  buttonLoading: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
  formDisclaimer: { marginTop: spacing.sm, marginBottom: spacing.md },
  pressed: { opacity: 0.82 },
  disabled: { opacity: 0.68 },
  error: { backgroundColor: '#f8e4df', borderColor: '#ebc2b8', borderWidth: 1, borderRadius: 13, padding: 16, marginTop: spacing.md },
  errorTitle: { color: colors.error, fontWeight: '700', marginBottom: 5 },
  errorText: { color: colors.error, lineHeight: 20 },
  result: { backgroundColor: colors.teal, borderRadius: 18, padding: 20, marginTop: spacing.md },
})
