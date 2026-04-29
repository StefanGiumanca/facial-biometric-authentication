import * as ImagePicker from 'expo-image-picker';
import { router } from 'expo-router';
import { useState } from 'react';
import { Alert, Image, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as ImageManipulator from 'expo-image-manipulator';
import { uploadKycFile, type UploadAsset } from '@/constants/api';

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

const visibleDocumentFields = ['cnp', 'last_name', 'first_name', 'series', 'number', 'address'];

export default function DocumentScreen() {
  const [asset, setAsset] = useState<ImagePicker.ImagePickerAsset | null>(null);
  const [documentResult, setDocumentResult] = useState<DocumentResponse | null>(null);
  const [isUploading, setIsUploading] = useState(false);

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
    <SafeAreaView style={styles.screen}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.step}>Step 1 of 5</Text>
        <Text style={styles.title}>Scan your ID</Text>
        <Text style={styles.subtitle}>
          Capture the front of the Romanian ID card clearly so OCR and face extraction can run.
        </Text>

        {asset ? (
          <Image source={{ uri: asset.uri }} style={styles.preview} />
        ) : (
          <View style={styles.placeholder}>
            <Text style={styles.placeholderText}>No ID photo selected</Text>
          </View>
        )}

        <View style={styles.row}>
          <Pressable style={styles.secondaryButton} onPress={captureDocument}>
            <Text style={styles.secondaryButtonText}>Open camera</Text>
          </Pressable>
          <Pressable style={styles.secondaryButton} onPress={chooseDocument}>
            <Text style={styles.secondaryButtonText}>Choose photo</Text>
          </Pressable>
        </View>

        <Pressable
          style={({ pressed }) => [styles.button, pressed && styles.buttonPressed]}
          onPress={uploadDocument}
          disabled={isUploading}>
          <Text style={styles.buttonText}>{isUploading ? 'Uploading...' : 'Upload ID card'}</Text>
        </Pressable>

        {documentResult && (
          <View style={styles.resultPanel}>
            <Text style={styles.resultTitle}>Document accepted</Text>
            <Text style={styles.resultText}>ID face was extracted and OCR data was stored in the session.</Text>
            {visibleDocumentFields.map((field) => {
              const value = documentResult[field];
              if (!value) {
                return null;
              }

              return (
                <Text key={field} style={styles.fieldText}>
                  {field.replaceAll('_', ' ')}: {String(value)}
                </Text>
              );
            })}
            <Pressable
              style={styles.button}
              onPress={() =>
                router.push({
                  pathname: '/review',
                  params: { documentResult: JSON.stringify(documentResult) },
                })
              }>
              <Text style={styles.buttonText}>Continue to review</Text>
            </Pressable>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

function getDocumentUploadErrorMessage(error: unknown) {
  const message = error instanceof Error ? error.message : '';

  if (message.includes('JSON Parse error') || message.includes('No face detected on ID')) {
    return 'We could not detect a valid Romanian ID card in this photo. Please scan the front of your ID clearly and try again.';
  }

  return message || 'Could not upload ID card. Please try again with a clear Romanian ID photo.';
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
  preview: {
    width: '100%',
    aspectRatio: 1.58,
    borderRadius: 8,
    marginBottom: 16,
  },
  placeholder: {
    alignItems: 'center',
    aspectRatio: 1.58,
    backgroundColor: '#111827',
    borderColor: '#334155',
    borderRadius: 8,
    borderWidth: 1,
    justifyContent: 'center',
    marginBottom: 16,
    width: '100%',
  },
  placeholderText: {
    color: '#94A3B8',
    fontSize: 15,
  },
  row: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 12,
  },
  secondaryButton: {
    alignItems: 'center',
    backgroundColor: '#1F2937',
    borderRadius: 8,
    flex: 1,
    paddingVertical: 14,
  },
  secondaryButtonText: {
    color: '#E5E7EB',
    fontSize: 15,
    fontWeight: '700',
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
  resultPanel: {
    backgroundColor: '#111827',
    borderColor: '#1D4ED8',
    borderRadius: 8,
    borderWidth: 1,
    gap: 10,
    marginTop: 20,
    padding: 16,
  },
  resultTitle: {
    color: 'white',
    fontSize: 18,
    fontWeight: '800',
  },
  resultText: {
    color: '#CBD5E1',
    fontSize: 14,
    lineHeight: 20,
  },
  fieldText: {
    color: '#E5E7EB',
    fontSize: 14,
    textTransform: 'capitalize',
  },
});
