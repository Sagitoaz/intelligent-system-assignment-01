import type { FieldMetadata } from './types'

export function numericInputMinimum(field: FieldMetadata): number | undefined {
  const lowerBounds: number[] = []
  if (field.minimum !== undefined) {
    lowerBounds.push(field.integer ? Math.ceil(field.minimum) : field.minimum)
  }
  if (field.exclusive_minimum !== undefined) {
    lowerBounds.push(
      field.integer ? Math.floor(field.exclusive_minimum) + 1 : field.exclusive_minimum,
    )
  }
  return lowerBounds.length > 0 ? Math.max(...lowerBounds) : undefined
}

export function numericInputStep(field: FieldMetadata): number | 'any' {
  return field.integer ? 1 : 'any'
}

export function validateNumericValues(
  fields: FieldMetadata[],
  values: Record<string, string>,
): string | null {
  for (const field of fields) {
    if (field.type !== 'number') continue
    const raw = values[field.name] ?? ''
    if (raw === '') {
      if (field.nullable) continue
      return `${field.label} is required.`
    }

    const value = Number(raw)
    if (!Number.isFinite(value)) return `${field.label} must be a finite number.`
    if (field.integer && !Number.isInteger(value)) return `${field.label} must be an integer.`
    if (field.minimum !== undefined && value < field.minimum) {
      return `${field.label} must be at least ${field.minimum}.`
    }
    if (field.exclusive_minimum !== undefined && value <= field.exclusive_minimum) {
      return `${field.label} must be greater than ${field.exclusive_minimum}.`
    }
  }
  return null
}
