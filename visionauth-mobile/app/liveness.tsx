import * as ImagePicker from 'expo-image-picker';
import { router, Stack } from 'expo-router';
import { useEffect, useState } from 'react';
import { Alert, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { getKyc, uploadKycFile, type ApiError, type UploadAsset } from '@/constants/api';

type LivenessResponse = {
  ok?: boolean;
  passed?: boolean;
  blink_count?: number;
  required_blinks?: number;
  analyzed_frames?: number;
  [key: string]: unknown;
};

type SessionStatus = {
  liveness_passed?: boolean;
};

export default function LivenessScreen() {
  const [asset, setAsset] = useState<ImagePicker.ImagePickerAsset | null>(null);
  const [result, setResult] = useState<LivenessResponse | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  useEffect(() => {
    hydrateLivenessProgress();
  }, []);

  const hydrateLivenessProgress = async () => {
    try {
      const status = (await getKyc('/kyc/session/status')) as SessionStatus;

      if (status.liveness_passed) {
        setResult({ ok: true, passed: true });
      }
    } catch (error) {
      console.log('Liveness progress restore skipped:', error);
    }
  };

  const recordVideo = async () => {
    const permission = await ImagePicker.requestCameraPermissionsAsync();

    if (!permission.granted) {
      Alert.alert('Permission needed', 'Camera permission is required to record the liveness video.');
      return;
    }

    const video = await ImagePicker.launchCameraAsync({
      cameraType: ImagePicker.CameraType.front,
      mediaTypes: ['videos'],
      quality: 0.8,
      videoMaxDuration: 10,
    });

    if (!video.canceled) {
      setAsset(video.assets[0]);
      setResult(null);
    }
  };

  const uploadVideo = async () => {
    if (!asset) {
      Alert.alert('Video missing', 'Record a short blink video before continuing.');
      return;
    }

    setIsUploading(true);

    try {
      const data = (await uploadKycFile('/kyc/liveness', asset as UploadAsset, 'liveness.mp4', 'video/mp4')) as LivenessResponse;
      setResult(data);

      if (data.passed) {
        router.push('/result');
      }
    } catch (error) {
      console.log('Liveness upload error:', error);
      const locked = isSessionLockedError(error);
      const message = getLivenessErrorMessage(error);

      if (locked) {
        Alert.alert('Session rejected', message, [
          {
            text: 'View result',
            onPress: () =>
              router.replace({
                pathname: '/result',
                params: {
                  decision: 'REJECTED',
                  reason: 'TOO_MANY_FAILED_SECURITY_CHECKS',
                },
              }),
          },
        ]);
        return;
      }

      Alert.alert('Liveness check failed', message);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <SafeAreaView style={styles.screen}>
      <Stack.Screen options={{ headerShown: false, title: '' }} />
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.step}>Step 4 of 5</Text>
        <Text style={styles.title}>Blink liveness</Text>
        <Text style={styles.subtitle}>
          Record a short video with your face visible and blink at least three times.
        </Text>

        <View style={styles.instructions}>
          <Text style={styles.instruction}>Keep the phone steady.</Text>
          <Text style={styles.instruction}>Use good lighting.</Text>
          <Text style={styles.instruction}>Blink clearly during the recording.</Text>
        </View>

        {asset && <Text style={styles.videoReady}>Video ready to upload.</Text>}

        <Pressable style={styles.secondaryButton} onPress={recordVideo}>
          <Text style={styles.secondaryButtonText}>Record liveness video</Text>
        </Pressable>

        <Pressable
          style={({ pressed }) => [styles.button, pressed && styles.buttonPressed]}
          onPress={result?.passed ? () => router.push('/result') : uploadVideo}
          disabled={isUploading}>
          <Text style={styles.buttonText}>
            {isUploading ? 'Analyzing...' : result?.passed ? 'Continue to result' : 'Upload liveness video'}
          </Text>
        </Pressable>

        {result && !result.passed && (
          <View style={styles.resultPanel}>
            <Text style={styles.resultTitle}>Liveness not passed</Text>
            <Text style={styles.resultText}>
              Blinks detected: {String(result.blink_count ?? 0)} / {String(result.required_blinks ?? 3)}
            </Text>
            <Pressable style={styles.button} onPress={() => router.push('/result')}>
              <Text style={styles.buttonText}>View result</Text>
            </Pressable>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

function isSessionLockedError(error: unknown) {
  const data = (error as ApiError | undefined)?.data;
  const detail = typeof data?.detail === 'object' ? data.detail : null;

  return (
    data?.session_locked === true ||
    detail?.code === 'SESSION_LOCKED' ||
    detail?.reason === 'TOO_MANY_FAILED_SECURITY_CHECKS'
  );
}

function getLivenessErrorMessage(error: unknown) {
  const apiError = error as ApiError | undefined;
  const detail = typeof apiError?.data?.detail === 'object' ? apiError.data.detail : null;

  if (detail?.code === 'SESSION_LOCKED' || detail?.reason === 'TOO_MANY_FAILED_SECURITY_CHECKS') {
    return detail.message || 'This session has been rejected after too many failed security checks.';
  }

  if (apiError?.data?.error) {
    return apiError.data.error;
  }

  if (typeof apiError?.data?.detail === 'string') {
    return apiError.data.detail;
  }

  return error instanceof Error ? error.message : 'Could not upload liveness video.';
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: '#0B1220',
  },
  content: {
    flexGrow: 1,
    justifyContent: 'center',
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
  instructions: {
    gap: 10,
    marginBottom: 24,
  },
  instruction: {
    color: '#E5E7EB',
    fontSize: 15,
  },
  videoReady: {
    color: '#86EFAC',
    fontSize: 15,
    fontWeight: '700',
    marginBottom: 14,
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
    borderColor: '#991B1B',
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
