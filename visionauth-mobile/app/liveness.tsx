import * as ImagePicker from 'expo-image-picker';
import { router, Stack } from 'expo-router';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { Alert, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { getKyc, postKyc, uploadKycFile, type ApiError, type UploadAsset } from '@/constants/api';
import {
  ChipRow,
  InfoCard,
  PageHeader,
  PrimaryButton,
  RadarVisual,
  SecondaryButton,
  StatusChip,
  StepIndicator,
  vaColors,
} from '@/components/visionauth-ui';

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

  const state = useMemo(
    () => getLivenessVisualState({ assetReady: Boolean(asset), challengeReady: Boolean(challenge), isLoadingChallenge, isUploading, result }),
    [asset, challenge, isLoadingChallenge, isUploading, result],
  );

  return (
    <SafeAreaView style={styles.screen}>
      <Stack.Screen options={{ headerShown: false, title: '' }} />
      <ScrollView contentContainerStyle={styles.content}>
        <StepIndicator currentStep={4} label="Liveness challenge" />
        <PageHeader
          title="Challenge response"
          subtitle="Record a short video with your face visible and complete the randomized action."
        />

        <InfoCard style={styles.radarCard}>
          <RadarVisual active={state.active} passed={result?.passed === true} failed={result?.passed === false} />
          <Text style={styles.stateLabel}>{state.label}</Text>
          <Text style={styles.stateText}>{state.text}</Text>
          <ChipRow>
            <StatusChip label={challenge?.challenge_type?.replace(/_/g, ' ') || 'Waiting'} tone={challenge ? 'blue' : 'slate'} />
            <StatusChip label={asset ? 'Video ready' : 'No video'} tone={asset ? 'green' : 'slate'} />
          </ChipRow>
        </InfoCard>

        <InfoCard title="Your challenge" style={styles.challengeCard}>
          <Text style={styles.challengeText}>
            {isLoadingChallenge ? 'Generating challenge...' : challenge?.instruction || 'Waiting for challenge'}
          </Text>
          <Text style={styles.challengeHint}>Start neutral for one second, then perform the requested action clearly.</Text>
        </InfoCard>

        <InfoCard title="Recording guidance" style={styles.instructions}>
          <InstructionRow text="Keep the phone steady and use good lighting." />
          <InstructionRow text="Keep only your face in frame." />
          <InstructionRow text="Make the requested action clear for at least a few seconds." />
        </InfoCard>

        <View style={styles.actions}>
          <SecondaryButton label={asset ? 'Record again' : 'Record liveness video'} onPress={recordVideo} disabled={isLoadingChallenge || !challenge} />
          <PrimaryButton
            label={isUploading ? 'Analyzing challenge...' : result?.passed ? 'Continue to result' : 'Upload challenge video'}
            onPress={result?.passed ? () => router.push('/result') : uploadVideo}
            disabled={isUploading || isLoadingChallenge || !challenge}
          />
        </View>

        <Pressable style={styles.ghostButton} onPress={requestChallenge} disabled={isUploading || isLoadingChallenge}>
          <Text style={styles.ghostButtonText}>Try a new challenge</Text>
        </Pressable>

        {result && !result.passed && (
          <InfoCard title="Challenge not completed" style={styles.resultPanel}>
            <Text style={styles.resultText}>{result.message || result.error || 'Please try again.'}</Text>
            <Text style={styles.resultText}>{formatChallengeDetails(result)}</Text>
            <PrimaryButton label="Retry with new challenge" onPress={requestChallenge} />
          </InfoCard>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

function InstructionRow({ text }: { text: string }) {
  return (
    <View style={styles.instructionRow}>
      <View style={styles.instructionDot} />
      <Text style={styles.instruction}>{text}</Text>
    </View>
  );
}

function getLivenessVisualState({
  assetReady,
  challengeReady,
  isLoadingChallenge,
  isUploading,
  result,
}: {
  assetReady: boolean;
  challengeReady: boolean;
  isLoadingChallenge: boolean;
  isUploading: boolean;
  result: LivenessResponse | null;
}) {
  if (isUploading) {
    return {
      active: true,
      label: 'Analyzing challenge',
      text: 'The backend is checking motion, face visibility, and identity binding.',
    };
  }

  if (result?.passed) {
    return {
      active: false,
      label: 'Liveness passed',
      text: 'Challenge response and selfie binding passed for this session.',
    };
  }

  if (result && !result.passed) {
    return {
      active: false,
      label: 'Challenge failed',
      text: 'Record a clearer attempt with the requested action visible.',
    };
  }

  if (isLoadingChallenge || !challengeReady) {
    return {
      active: true,
      label: 'Waiting for challenge',
      text: 'VisionAuth is preparing a randomized action.',
    };
  }

  if (assetReady) {
    return {
      active: true,
      label: 'Ready to analyze',
      text: 'Upload the recorded video to verify the challenge.',
    };
  }

  return {
    active: false,
    label: 'Ready to record',
    text: 'Record the requested action with your face visible.',
  };
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
    backgroundColor: vaColors.background,
  },
  content: {
    flexGrow: 1,
    padding: 24,
  },
  radarCard: {
    alignItems: 'center',
    marginBottom: 14,
  },
  stateLabel: {
    color: vaColors.text,
    fontSize: 20,
    fontWeight: '900',
    marginBottom: 6,
    textAlign: 'center',
  },
  stateText: {
    color: vaColors.subtle,
    fontSize: 14,
    lineHeight: 20,
    marginBottom: 14,
    textAlign: 'center',
  },
  challengeCard: {
    gap: 8,
    marginBottom: 14,
  },
  challengeText: {
    color: vaColors.text,
    fontSize: 26,
    fontWeight: '900',
    lineHeight: 32,
  },
  challengeHint: {
    color: vaColors.muted,
    fontSize: 14,
    lineHeight: 20,
  },
  instructions: {
    gap: 10,
    marginBottom: 14,
  },
  instructionRow: {
    flexDirection: 'row',
    gap: 10,
  },
  instructionDot: {
    backgroundColor: vaColors.blueSoft,
    borderRadius: 999,
    height: 8,
    marginTop: 6,
    width: 8,
  },
  instruction: {
    color: '#E5E7EB',
    flex: 1,
    fontSize: 15,
    lineHeight: 21,
  },
  actions: {
    gap: 12,
  },
  ghostButton: {
    alignItems: 'center',
    marginTop: 12,
    paddingVertical: 12,
  },
  ghostButtonText: {
    color: '#93C5FD',
    fontSize: 15,
    fontWeight: '800',
  },
  resultPanel: {
    borderColor: 'rgba(239, 68, 68, 0.28)',
    gap: 10,
    marginTop: 14,
  },
  resultText: {
    color: vaColors.subtle,
    fontSize: 14,
    lineHeight: 20,
  },
});
