import * as ImagePicker from 'expo-image-picker';
import { router } from 'expo-router';
import { useCallback, useEffect, useState } from 'react';
import { Alert, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as ImageManipulator from 'expo-image-manipulator';
import { getKyc, uploadKycFile, type ApiError, type UploadAsset } from '@/constants/api';
import {
  AppBackground,
  ChipRow,
  InfoCard,
  PageHeader,
  PrimaryButton,
  ScannerFrame,
  SecondaryButton,
  StatusChip,
  StepIndicator,
  vaColors,
} from '@/components/visionauth-ui';

type DocumentResponse = {
  ok?: boolean;
  filename?: string;
  document_path?: string;
  id_face_path?: string;
  series_roi_text?: string;
  [key: string]: unknown;
};

type PreparedUploadAsset = {
  uri: string;
  mimeType?: string | null;
  fileName?: string | null;
};

type SessionStatus = {
  id_face_extracted?: boolean;
  session_data?: {
    document_fields?: DocumentResponse | null;
    document_path?: string | null;
    id_face_path?: string | null;
  };
};

const visibleDocumentFields = ['cnp', 'last_name', 'first_name', 'series', 'number', 'address'];

export default function DocumentScreen() {
  const [asset, setAsset] = useState<ImagePicker.ImagePickerAsset | null>(null);
  const [documentResult, setDocumentResult] = useState<DocumentResponse | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  const hydrateDocumentProgress = useCallback(async () => {
    try {
      const status = (await getKyc('/kyc/session/status')) as SessionStatus;
      const documentFields = status.session_data?.document_fields;

      if (status.id_face_extracted && documentFields && !documentResult) {
        setDocumentResult({
          ok: true,
          ...documentFields,
          document_path: status.session_data?.document_path ?? undefined,
          id_face_path: status.session_data?.id_face_path ?? undefined,
        });
      }
    } catch (error) {
      console.log('Document progress restore skipped:', error);
    }
  }, [documentResult]);

  useEffect(() => {
    hydrateDocumentProgress();
  }, [hydrateDocumentProgress]);

  const requestPermissions = async () => {
    const cameraPermission = await ImagePicker.requestCameraPermissionsAsync();
    const mediaPermission = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (!cameraPermission.granted || !mediaPermission.granted) {
      Alert.alert('Permission needed', 'Camera and photo library permissions are required.');
      return false;
    }

    return true;
  };

  const captureDocument = async () => {
    const hasPermission = await requestPermissions();
    if (!hasPermission) {
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ['images'],
      quality: 1,
    });

    if (!result.canceled) {
      setAsset(result.assets[0]);
      setDocumentResult(null);
    }
  };

  const chooseDocument = async () => {
    const hasPermission = await requestPermissions();
    if (!hasPermission) {
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      quality: 1,
    });

    if (!result.canceled) {
      setAsset(result.assets[0]);
      setDocumentResult(null);
    }
  };

  const uploadDocument = async () => {
    if (!asset) {
      Alert.alert('ID photo missing', 'Take or choose a Romanian ID card photo first.');
      return;
    }

    setIsUploading(true);

    try {
      const manipulatedImage = await ImageManipulator.manipulateAsync(
        asset.uri,
        [],
        {
          compress: 0.9,
          format: ImageManipulator.SaveFormat.JPEG,
        }
      );

      const preparedAsset: PreparedUploadAsset = {
        uri: manipulatedImage.uri,
        mimeType: 'image/jpeg',
        fileName: 'document.jpg',
      };

      const data = (await uploadKycFile(
        '/kyc/document',
        preparedAsset as UploadAsset,
        'document.jpg',
        'image/jpeg'
      )) as DocumentResponse;

      setDocumentResult(data);
    } catch (error) {
      console.log('Document upload error:', error);
      Alert.alert('Document check failed', getDocumentUploadErrorMessage(error));
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <AppBackground>
      <SafeAreaView style={styles.screen}>
        <ScrollView contentContainerStyle={styles.content}>
          <StepIndicator currentStep={1} label="Document capture" />
          <PageHeader
            eyebrow="Identity document"
            title="Scan your ID"
            subtitle="Align the Romanian ID inside the scanner frame so OCR and face extraction can run cleanly."
          />

          <InfoCard title="Live document scanner">
            <ScannerFrame imageUri={asset?.uri} placeholder="No ID photo selected" />
            <View style={styles.scanMeta}>
              <StatusChip label={asset ? 'Image ready' : 'Awaiting image'} tone={asset ? 'green' : 'slate'} />
              <StatusChip label="OCR standby" tone={asset ? 'blue' : 'slate'} />
              <StatusChip label="Face crop" tone={documentResult?.id_face_path ? 'green' : 'amber'} />
            </View>
          </InfoCard>

          <InfoCard title="Capture tips" style={styles.tipsCard}>
            <TipRow index="01" text="Keep all corners visible inside the scanner frame." />
            <TipRow index="02" text="Avoid glare, blur, and strong shadows over the text." />
            <TipRow index="03" text="Use good lighting and keep the document flat." />
          </InfoCard>

          <View style={styles.row}>
            <SecondaryButton label="Open camera" onPress={captureDocument} style={styles.rowButton} />
            <SecondaryButton label="Choose photo" onPress={chooseDocument} style={styles.rowButton} />
          </View>

          <PrimaryButton label={isUploading ? 'Analyzing document...' : 'Upload ID card'} onPress={uploadDocument} disabled={isUploading} />

          {documentResult && (
            <InfoCard title="OCR result" style={styles.resultPanel}>
              <ChipRow>
                <StatusChip label="Document uploaded" tone="green" />
                <StatusChip label="Face crop extracted" tone={documentResult.id_face_path ? 'green' : 'amber'} />
                <StatusChip label="OCR data detected" tone="blue" />
              </ChipRow>
              <Text style={styles.resultTitle}>Document accepted</Text>
              <Text style={styles.resultText}>ID face was extracted and OCR data was stored in the session.</Text>
              <View style={styles.fieldGrid}>
                {visibleDocumentFields.map((field) => {
                  const value = documentResult[field];
                  if (!value) {
                    return null;
                  }

                  return (
                    <View key={field} style={styles.fieldPill}>
                      <Text style={styles.fieldLabel}>{field.replaceAll('_', ' ')}</Text>
                      <Text style={styles.fieldText}>{String(value)}</Text>
                    </View>
                  );
                })}
              </View>
              <PrimaryButton
                label="Continue to review"
                onPress={() =>
                  router.push({
                    pathname: '/review',
                    params: { documentResult: JSON.stringify(documentResult) },
                  })
                }
                style={styles.continueButton}
              />
            </InfoCard>
          )}
        </ScrollView>
      </SafeAreaView>
    </AppBackground>
  );
}

function TipRow({ index, text }: { index: string; text: string }) {
  return (
    <View style={styles.tipRow}>
      <Text style={styles.tipBullet}>{index}</Text>
      <Text style={styles.tipText}>{text}</Text>
    </View>
  );
}

function getDocumentUploadErrorMessage(error: unknown) {
  const apiError = error as ApiError | undefined;
  const detail = typeof apiError?.data?.detail === 'object' ? apiError.data.detail : null;

  if (detail?.code === 'INVALID_ROMANIAN_ID_DOCUMENT') {
    return detail.message || 'This photo does not look like a real Romanian ID card. Please scan the front of your Romanian ID clearly.';
  }

  if (apiError?.data?.error) {
    return apiError.data.error;
  }

  const message = error instanceof Error ? error.message : '';

  if (message.includes('JSON Parse error') || message.includes('No face detected on ID')) {
    return 'We could not detect a valid Romanian ID card in this photo. Please scan the front of your ID clearly and try again.';
  }

  return message || 'Could not upload ID card. Please try again with a clear Romanian ID photo.';
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
  },
  content: {
    flexGrow: 1,
    padding: 24,
  },
  scanMeta: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginTop: 14,
  },
  row: {
    flexDirection: 'row',
    gap: 12,
    marginVertical: 14,
  },
  rowButton: {
    flex: 1,
  },
  tipsCard: {
    gap: 12,
    marginTop: 14,
  },
  tipRow: {
    flexDirection: 'row',
    gap: 10,
  },
  tipBullet: {
    color: vaColors.blueSoft,
    fontSize: 12,
    fontWeight: '900',
    width: 24,
  },
  tipText: {
    color: vaColors.subtle,
    flex: 1,
    fontSize: 14,
    lineHeight: 20,
  },
  resultPanel: {
    gap: 10,
    marginTop: 20,
  },
  resultTitle: {
    color: vaColors.text,
    fontSize: 18,
    fontWeight: '900',
  },
  resultText: {
    color: vaColors.subtle,
    fontSize: 14,
    lineHeight: 20,
  },
  fieldText: {
    color: '#E5E7EB',
    fontSize: 14,
    fontWeight: '800',
    marginTop: 4,
  },
  fieldGrid: {
    gap: 8,
  },
  fieldPill: {
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    borderColor: 'rgba(148, 163, 184, 0.14)',
    borderRadius: 14,
    borderWidth: 1,
    padding: 12,
  },
  fieldLabel: {
    color: vaColors.muted,
    fontSize: 11,
    fontWeight: '900',
    textTransform: 'uppercase',
  },
  continueButton: {
    marginTop: 4,
  },
});
