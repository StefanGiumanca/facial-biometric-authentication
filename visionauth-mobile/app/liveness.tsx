import * as ImagePicker from 'expo-image-picker';
import { router, Stack } from 'expo-router';
import { useCallback, useEffect, useState } from 'react';
import { Alert, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { getKyc, postKyc, uploadKycFile, type ApiError, type UploadAsset } from '@/constants/api';

type LivenessResponse = {
  ok?: boolean;
  passed?: boolean;
  challenge_id?: string;
  challenge_type?: string;
  instruction?: string;
  message?: string;
  error?: string;
  details?: {
    challenge?: {
      blink_count?: number;
      required_blinks?: number;
      consecutive_hit_frames?: number;
      required_hit_frames?: number;
      max_finger_count?: number;
    };
  };
  [key: string]: unknown;
};

type LivenessChallenge = {
  ok?: boolean;
  challenge_id: string;
  challenge_type: string;
  instruction: string;
  required_action?: Record<string, unknown>;
};

type SessionStatus = {
  liveness_passed?: boolean;
  liveness_challenge?: LivenessChallenge | null;
};

export default function LivenessScreen() {
  const [asset, setAsset] = useState<ImagePicker.ImagePickerAsset | null>(null);
  const [challenge, setChallenge] = useState<LivenessChallenge | null>(null);
  const [result, setResult] = useState<LivenessResponse | null>(null);
  const [isLoadingChallenge, setIsLoadingChallenge] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  const requestChallenge = useCallback(async () => {
    setIsLoadingChallenge(true);
    setAsset(null);
    setResult(null);

    try {
      const data = (await postKyc('/kyc/liveness/challenge')) as LivenessChallenge;

      if (data.ok === false) {
        Alert.alert('Challenge unavailable', 'Please complete the selfie step before starting liveness.');
        return;
      }

      setChallenge(data);
    } catch (error) {
      console.log('Liveness challenge error:', error);
      Alert.alert('Challenge unavailable', getLivenessErrorMessage(error));
    } finally {
      setIsLoadingChallenge(false);
    }
  }, []);

  const hydrateLivenessProgress = useCallback(async () => {
    try {
      const status = (await getKyc('/kyc/session/status')) as SessionStatus;

      if (status.liveness_passed) {
        setResult({ ok: true, passed: true });
        return;
      }

      if (status.liveness_challenge?.challenge_id) {
        setChallenge(status.liveness_challenge);
        return;
      }

      await requestChallenge();
    } catch (error) {
      console.log('Liveness progress restore skipped:', error);
      await requestChallenge();
    }
  }, [requestChallenge]);

  useEffect(() => {
    hydrateLivenessProgress();
  }, [hydrateLivenessProgress]);

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
      Alert.alert('Video missing', 'Record a short liveness challenge video before continuing.');
      return;
    }

    if (!challenge) {
      Alert.alert('Challenge missing', 'Please generate a liveness challenge before recording.');
      return;
    }

    setIsUploading(true);

    try {
      const data = (await uploadKycFile('/kyc/liveness', asset as UploadAsset, 'liveness.mp4', 'video/mp4')) as LivenessResponse;
      setResult(data);

      if (data.passed) {
        router.push('/result');
      } else {
        Alert.alert('Challenge not completed', data.message || data.error || 'Please try again with a new challenge.');
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
        <Text style={styles.title}>Liveness challenge</Text>
        <Text style={styles.subtitle}>
          Record a short video with your face visible and complete the randomized action.
        </Text>

        <View style={styles.challengeCard}>
          <Text style={styles.challengeLabel}>Your challenge</Text>
          <Text style={styles.challengeText}>
            {isLoadingChallenge ? 'Generating challenge...' : challenge?.instruction || 'No challenge loaded'}
          </Text>
          {challenge?.challenge_type && <Text style={styles.challengeType}>{challenge.challenge_type.replace(/_/g, ' ')}</Text>}
        </View>

        <View style={styles.instructions}>
          <Text style={styles.instruction}>Keep the phone steady.</Text>
          <Text style={styles.instruction}>Use good lighting.</Text>
          <Text style={styles.instruction}>Keep only your face in frame.</Text>
          <Text style={styles.instruction}>Start neutral for one second, then perform the action.</Text>
          <Text style={styles.instruction}>Make the requested action clear for at least a few seconds.</Text>
        </View>

        {asset && <Text style={styles.videoReady}>Video ready to upload.</Text>}

        <Pressable style={styles.secondaryButton} onPress={recordVideo} disabled={isLoadingChallenge || !challenge}>
          <Text style={styles.secondaryButtonText}>Record liveness video</Text>
        </Pressable>

        <Pressable
          style={({ pressed }) => [styles.button, pressed && styles.buttonPressed]}
          onPress={result?.passed ? () => router.push('/result') : uploadVideo}
          disabled={isUploading || isLoadingChallenge || !challenge}>
          <Text style={styles.buttonText}>
            {isUploading ? 'Analyzing challenge...' : result?.passed ? 'Continue to result' : 'Upload challenge video'}
          </Text>
        </Pressable>

        <Pressable style={styles.ghostButton} onPress={requestChallenge} disabled={isUploading || isLoadingChallenge}>
          <Text style={styles.ghostButtonText}>Try a new challenge</Text>
        </Pressable>

        {result && !result.passed && (
          <View style={styles.resultPanel}>
            <Text style={styles.resultTitle}>Challenge not completed</Text>
            <Text style={styles.resultText}>{result.message || result.error || 'Please try again.'}</Text>
            <Text style={styles.resultText}>{formatChallengeDetails(result)}</Text>
            <Pressable style={styles.button} onPress={requestChallenge}>
              <Text style={styles.buttonText}>Retry with new challenge</Text>
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

function formatChallengeDetails(result: LivenessResponse) {
  const details = result.details?.challenge;

  if (!details) {
    return 'Tip: keep your face centered, avoid other people in frame, and make the action clear.';
  }

  if (typeof details.blink_count === 'number') {
    return `Blinks detected: ${details.blink_count} / ${details.required_blinks ?? 3}`;
  }

  if (typeof details.max_finger_count === 'number' && details.max_finger_count > 0) {
    return `Fingers detected: up to ${details.max_finger_count}.`;
  }

  return `Detected action frames: ${details.consecutive_hit_frames ?? 0} / ${details.required_hit_frames ?? 0}`;
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
  challengeCard: {
    backgroundColor: '#111827',
    borderColor: '#2563EB',
    borderRadius: 8,
    borderWidth: 1,
    marginBottom: 20,
    padding: 16,
  },
  challengeLabel: {
    color: '#93C5FD',
    fontSize: 13,
    fontWeight: '800',
    marginBottom: 8,
    textTransform: 'uppercase',
  },
  challengeText: {
    color: 'white',
    fontSize: 24,
    fontWeight: '900',
    lineHeight: 30,
  },
  challengeType: {
    color: '#CBD5E1',
    fontSize: 13,
    fontWeight: '700',
    marginTop: 8,
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
  ghostButton: {
    alignItems: 'center',
    marginTop: 12,
    paddingVertical: 12,
  },
  ghostButtonText: {
    color: '#93C5FD',
    fontSize: 15,
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
