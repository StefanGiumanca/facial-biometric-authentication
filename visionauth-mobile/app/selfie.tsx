import * as ImagePicker from 'expo-image-picker';
import { router } from 'expo-router';
import { useState } from 'react';
import { Alert, Image, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { uploadKycFile, type UploadAsset } from '@/constants/api';

export default function SelfieScreen() {
  const [asset, setAsset] = useState<ImagePicker.ImagePickerAsset | null>(null);
  const [facesDetected, setFacesDetected] = useState<number | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  const captureSelfie = async () => {
    const permission = await ImagePicker.requestCameraPermissionsAsync();

    if (!permission.granted) {
      Alert.alert('Permission needed', 'Camera permission is required to take a selfie.');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: false,
      cameraType: ImagePicker.CameraType.front,
      mediaTypes: ['images'],
      quality: 1,
    });

    if (!result.canceled) {
      setAsset(result.assets[0]);
      setFacesDetected(null);
    }
  };

  const uploadSelfie = async () => {
    if (!asset) {
      Alert.alert('Selfie missing', 'Take a selfie before continuing.');
      return;
    }

    setIsUploading(true);

    try {
      const data = await uploadKycFile('/kyc/selfie', asset as UploadAsset, 'selfie.jpg', 'image/jpeg');
      setFacesDetected(typeof data.faces_detected === 'number' ? data.faces_detected : null);
    } catch (error) {
      console.log('Selfie upload error:', error);
      Alert.alert('Selfie check failed', error instanceof Error ? error.message : 'Could not upload selfie.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <SafeAreaView style={styles.screen}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.step}>Step 2 of 4</Text>
        <Text style={styles.title}>Take a selfie</Text>
        <Text style={styles.subtitle}>
          Keep your face centered and well lit. The backend checks that a face is present before the
          liveness step.
        </Text>

        {asset ? (
          <Image source={{ uri: asset.uri }} style={styles.preview} />
        ) : (
          <View style={styles.placeholder}>
            <Text style={styles.placeholderText}>No selfie captured</Text>
          </View>
        )}

        <Pressable style={styles.secondaryButton} onPress={captureSelfie}>
          <Text style={styles.secondaryButtonText}>Open camera</Text>
        </Pressable>

        <Pressable
          style={({ pressed }) => [styles.button, pressed && styles.buttonPressed]}
          onPress={uploadSelfie}
          disabled={isUploading}>
          <Text style={styles.buttonText}>{isUploading ? 'Uploading...' : 'Upload selfie'}</Text>
        </Pressable>

        {facesDetected !== null && (
          <View style={styles.resultPanel}>
            <Text style={styles.resultTitle}>Selfie accepted</Text>
            <Text style={styles.resultText}>Faces detected: {facesDetected}</Text>
            <Pressable style={styles.button} onPress={() => router.push('/liveness')}>
              <Text style={styles.buttonText}>Continue to liveness</Text>
            </Pressable>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
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
    alignSelf: 'center',
    aspectRatio: 0.75,
    borderRadius: 8,
    marginBottom: 16,
    width: '80%',
  },
  placeholder: {
    alignItems: 'center',
    alignSelf: 'center',
    aspectRatio: 0.75,
    backgroundColor: '#111827',
    borderColor: '#334155',
    borderRadius: 8,
    borderWidth: 1,
    justifyContent: 'center',
    marginBottom: 16,
    width: '80%',
  },
  placeholderText: {
    color: '#94A3B8',
    fontSize: 15,
  },
  secondaryButton: {
    alignItems: 'center',
    backgroundColor: '#1F2937',
    borderRadius: 8,
    marginBottom: 12,
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
  },
});
