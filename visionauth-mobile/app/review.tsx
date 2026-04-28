import { router, useLocalSearchParams } from 'expo-router';
import { useEffect, useMemo, useState } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { postKycJson, type ApiError, type ApiResponse } from '@/constants/api';

type DocumentResult = {
  [key: string]: unknown;
};

type EditableValues = Record<string, string>;

type ValidationFieldFailure = {
  field?: unknown;
  message?: unknown;
  similarity?: unknown;
};

type ValidationResponse = ApiResponse & {
  failed_fields?: ValidationFieldFailure[];
};

const requiredFieldsMessage = 'Please complete all required fields before continuing.';
const genericMismatchMessage = 'Some details do not match your ID. Please check and correct them.';
const fieldValidationMessages: Record<string, string> = {
  nationality: 'Nationality must be at least 3 characters long.',
  sex: 'Sex must be M or F.',
  address: 'Address must be at least 8 characters long.',
  valid_from: 'Valid from must be a real date in DD.MM.YY or DD.MM.YYYY format.',
  valid_until: 'Valid until must be a real date in DD.MM.YY or DD.MM.YYYY format.',
};

const preferredFieldOrder = [
  'first_name',
  'last_name',
  'cnp',
  'sex',
  'nationality',
  'series_number',
  'address',
  'valid_from',
  'valid_until',
];
const hiddenResponseFields = new Set([
  'ok',
  'filename',
  'document_path',
  'id_face_path',
  'series_roi_text',
  'series',
  'number',
]);

const fieldLabels: Record<string, string> = {
  first_name: 'First name',
  last_name: 'Last name',
  cnp: 'CNP',
  sex: 'Sex',
  nationality: 'Nationality',
  series_number: 'Series / number',
  address: 'Address',
  valid_from: 'Valid from',
  valid_until: 'Valid until',
};

const friendlyValidationMessages: Record<string, string> = {
  first_name: 'The name you entered is too different from the one on your ID.',
  last_name: 'The name you entered is too different from the one on your ID.',
  cnp: 'CNP does not match the information on your ID.',
  series_number: 'ID series and number do not match the document.',
};

export default function ReviewScreen() {
  const params = useLocalSearchParams<{ documentResult?: string | string[] }>();
  const documentResult = useMemo(() => parseDocumentResult(params.documentResult), [params.documentResult]);
  const editableFields = useMemo(() => buildEditableFields(documentResult), [documentResult]);
  const initialValues = useMemo(() => buildInitialValues(documentResult, editableFields), [documentResult, editableFields]);
  const [values, setValues] = useState<EditableValues>(initialValues);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [formError, setFormError] = useState<string | null>(null);
  const [isValidating, setIsValidating] = useState(false);

  useEffect(() => {
    setValues(initialValues);
    setFieldErrors({});
    setFormError(null);
  }, [initialValues]);

  const updateField = (field: string, value: string) => {
    setFieldErrors((currentErrors) => {
      const nextErrors = { ...currentErrors };
      delete nextErrors[field];
      return nextErrors;
    });
    setFormError(null);

    setValues((currentValues) => ({
      ...currentValues,
      [field]: value,
    }));
  };

  const confirmReview = async () => {
    setIsValidating(true);
    setFieldErrors({});
    setFormError(null);

    const normalizedValues = normalizeReviewValues(values);
    const localValidation = validateReviewForm(normalizedValues, editableFields);

    if (!localValidation.isValid) {
      setFormError(localValidation.formError);
      setFieldErrors(localValidation.fieldErrors);
      setIsValidating(false);
      return;
    }

    try {
      await postKycJson('/kyc/review/validate', normalizedValues);
      router.push('/selfie');
    } catch (error) {
      const validationData = getValidationErrorData(error);
      setFormError(getFriendlyFormError(validationData, error));
      setFieldErrors(buildFieldErrors(validationData?.failed_fields));
    } finally {
      setIsValidating(false);
    }
  };

  return (
    <SafeAreaView style={styles.screen}>
      <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
        <Text style={styles.step}>Step 2 of 5</Text>
        <Text style={styles.title}>Review your data</Text>
        <Text style={styles.subtitle}>
          This information was extracted automatically from your ID. Correct any value before
          continuing.
        </Text>

        {formError && <Text style={styles.formError}>{formError}</Text>}

        <View style={styles.form}>
          {editableFields.map((field) => (
            <View key={field} style={styles.fieldGroup}>
              <Text style={styles.label}>{formatFieldLabel(field)}</Text>
              <TextInput
                autoCapitalize={field === 'cnp' ? 'none' : 'words'}
                keyboardType={field === 'cnp' || field === 'number' ? 'number-pad' : 'default'}
                multiline={field === 'address'}
                numberOfLines={field === 'address' ? 3 : 1}
                onChangeText={(value) => updateField(field, value)}
                placeholder="Not detected"
                placeholderTextColor="#64748B"
                style={[styles.input, fieldErrors[field] && styles.inputError, field === 'address' && styles.addressInput]}
                value={values[field] ?? ''}
              />
              {fieldErrors[field] && <Text style={styles.fieldError}>{fieldErrors[field]}</Text>}
            </View>
          ))}
        </View>

        <Pressable
          style={({ pressed }) => [styles.button, pressed && styles.buttonPressed]}
          onPress={confirmReview}
          disabled={isValidating}>
          <Text style={styles.buttonText}>{isValidating ? 'Validating...' : 'Confirm and continue'}</Text>
        </Pressable>
      </ScrollView>
    </SafeAreaView>
  );
}

function parseDocumentResult(rawParam?: string | string[]) {
  const rawValue = Array.isArray(rawParam) ? rawParam[0] : rawParam;

  if (!rawValue) {
    return {};
  }

  try {
    const parsed = JSON.parse(rawValue) as DocumentResult;
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch (error) {
    console.log('Could not parse document review data:', error);
    return {};
  }
}

function buildEditableFields(documentResult: DocumentResult) {
  const responseFields = Object.keys(documentResult).filter((field) => !hiddenResponseFields.has(field));
  const orderedFields = preferredFieldOrder;
  const extraFields = responseFields.filter((field) => !orderedFields.includes(field));

  return [...orderedFields, ...extraFields];
}

function buildInitialValues(documentResult: DocumentResult, fields: string[]) {
  return fields.reduce<EditableValues>((values, field) => {
    const rawValue = documentResult[field];
    values[field] = rawValue === undefined || rawValue === null ? '' : String(rawValue);
    return values;
  }, {});
}

function getValidationErrorData(error: unknown) {
  const apiError = error as ApiError;
  return apiError?.data as ValidationResponse | undefined;
}

function normalizeReviewValues(values: EditableValues) {
  return Object.entries(values).reduce<EditableValues>((normalizedValues, [field, value]) => {
    const trimmedValue = value.trim();
    normalizedValues[field] = field === 'sex' ? trimmedValue.toUpperCase() : trimmedValue;
    return normalizedValues;
  }, {});
}

function validateReviewForm(values: EditableValues, fields: string[]) {
  const fieldErrors = fields.reduce<Record<string, string>>((errors, field) => {
    const value = values[field]?.trim() ?? '';

    if (!value) {
      errors[field] = requiredFieldsMessage;
      return errors;
    }

    const specificError = validateFieldValue(field, value);
    if (specificError) {
      errors[field] = specificError;
    }

    return errors;
  }, {});

  return {
    isValid: Object.keys(fieldErrors).length === 0,
    fieldErrors,
    formError: Object.values(fieldErrors).some((message) => message === requiredFieldsMessage)
      ? requiredFieldsMessage
      : 'Please review the highlighted fields before continuing.',
  };
}

function validateFieldValue(field: string, value: string) {
  if (field === 'nationality' && value.length < 3) {
    return fieldValidationMessages.nationality;
  }

  if (field === 'sex' && !['M', 'F'].includes(value.toUpperCase())) {
    return fieldValidationMessages.sex;
  }

  if (field === 'address' && value.length < 8) {
    return fieldValidationMessages.address;
  }

  if ((field === 'valid_from' || field === 'valid_until') && !isValidReviewDate(value)) {
    return fieldValidationMessages[field];
  }

  return null;
}

function isValidReviewDate(value: string) {
  const match = value.match(/^(\d{2})\.(\d{2})\.(\d{2}|\d{4})$/);
  if (!match) {
    return false;
  }

  const day = Number(match[1]);
  const month = Number(match[2]);
  const rawYear = match[3];
  const year = rawYear.length === 2 ? Number(`20${rawYear}`) : Number(rawYear);
  const date = new Date(year, month - 1, day);

  return date.getFullYear() === year && date.getMonth() === month - 1 && date.getDate() === day;
}

function getFriendlyFormError(validationData: ValidationResponse | undefined, error: unknown) {
  if (Array.isArray(validationData?.failed_fields) && validationData.failed_fields.length > 0) {
    return 'Please review the highlighted fields before continuing.';
  }

  return error instanceof Error ? error.message : 'Document review validation failed.';
}

function buildFieldErrors(failedFields?: ValidationFieldFailure[]) {
  if (!Array.isArray(failedFields)) {
    return {};
  }

  return failedFields.reduce<Record<string, string>>((errors, failedField) => {
    if (typeof failedField.field !== 'string') {
      return errors;
    }

    errors[failedField.field] =
      friendlyValidationMessages[failedField.field] || genericMismatchMessage;

    return errors;
  }, {});
}

function formatFieldLabel(field: string) {
  if (fieldLabels[field]) {
    return fieldLabels[field];
  }

  return field
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: '#0B1220',
  },
  content: {
    flexGrow: 1,
    padding: 24,
  },
  step: {
    color: '#60A5FA',
    fontSize: 14,
    fontWeight: '700',
    marginBottom: 10,
  },
  title: {
    color: 'white',
    fontSize: 30,
    fontWeight: '800',
    marginBottom: 12,
  },
  subtitle: {
    color: '#C7D2FE',
    fontSize: 16,
    lineHeight: 23,
    marginBottom: 24,
  },
  form: {
    gap: 16,
    marginBottom: 24,
  },
  formError: {
    backgroundColor: '#450A0A',
    borderColor: '#DC2626',
    borderRadius: 8,
    borderWidth: 1,
    color: '#FCA5A5',
    fontSize: 14,
    lineHeight: 20,
    marginBottom: 18,
    padding: 12,
  },
  fieldGroup: {
    gap: 8,
  },
  label: {
    color: '#E5E7EB',
    fontSize: 14,
    fontWeight: '700',
  },
  input: {
    backgroundColor: '#111827',
    borderColor: '#334155',
    borderRadius: 8,
    borderWidth: 1,
    color: 'white',
    fontSize: 16,
    paddingHorizontal: 14,
    paddingVertical: 13,
  },
  inputError: {
    borderColor: '#DC2626',
  },
  fieldError: {
    color: '#FCA5A5',
    fontSize: 13,
    lineHeight: 18,
  },
  addressInput: {
    minHeight: 88,
    textAlignVertical: 'top',
  },
  button: {
    alignItems: 'center',
    backgroundColor: '#2563EB',
    borderRadius: 8,
    paddingVertical: 16,
  },
  buttonPressed: {
    opacity: 0.82,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '700',
  },
});
