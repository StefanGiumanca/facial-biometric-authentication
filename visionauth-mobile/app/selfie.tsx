import * as ImagePicker from 'expo-image-picker';
import { router } from 'expo-router';
import { useEffect, useState } from 'react';
import { Alert, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { getKyc, uploadKycFile, type ApiError, type UploadAsset } from '@/constants/api';
import {
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

type SessionStatus = {
  selfie_uploaded?: boolean;
};

export default function SelfieScreen() {
  const [asset, setAsset] = useState<ImagePicker.ImagePickerAsset | null>(null);
  const [facesDetected, setFacesDetected] = useState<number | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  useEffect(() => {
    hydrateSelfieProgress();
  }, []);

  const hydrateSelfieProgress = async () => {
    try {
      const status = (await getKyc('/kyc/session/status')) as SessionStatus;

      if (status.selfie_uploaded) {
        setFacesDetected(1);
      }
    } catch (error) {
      console.log('Selfie progress restore skipped:', error);
    }
  };

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
      const locked = isSessionLockedError(error);
      const message = getSelfieErrorMessage(error);

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

      Alert.alert('Selfie check failed', message);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <SafeAreaView style={styles.screen}>
      <ScrollView contentContainerStyle={styles.content}>
        <StepIndicator currentStep={3} label="Biometric capture" />
        <PageHeader
          title="Take a selfie"
          subtitle="Keep your face centered and well lit. This selfie will be compared with the ID portrait."
        />

        <InfoCard title="Face capture">
          <ScannerFrame imageUri={asset?.uri} placeholder="No selfie captured" variant="face" />
          <ChipRow>
            <StatusChip label="Face required" />
            <StatusChip label="Compared with ID" tone="green" />
          </ChipRow>
        </InfoCard>

        <InfoCard title="Capture guidance" style={styles.guidanceCard}>
          <View style={styles.guidanceRow}>
            <Text style={styles.guidanceIndex}>01</Text>
            <Text style={styles.guidanceText}>Face the camera directly and keep your head centered.</Text>
          </View>
          <View style={styles.guidanceRow}>
            <Text style={styles.guidanceIndex}>02</Text>
            <Text style={styles.guidanceText}>Use good lighting without strong backlight.</Text>
          </View>
          <View style={styles.guidanceRow}>
            <Text style={styles.guidanceIndex}>03</Text>
            <Text style={styles.guidanceText}>Keep only one person visible in the frame.</Text>
          </View>
        </InfoCard>

        <View style={styles.actions}>
          <SecondaryButton label="Open camera" onPress={captureSelfie} />
          <PrimaryButton label={isUploading ? 'Checking selfie...' : 'Upload selfie'} onPress={uploadSelfie} disabled={isUploading} />
        </View>

        {facesDetected !== null && (
          <InfoCard title="Selfie accepted" style={styles.resultPanel}>
            <ChipRow>
              <StatusChip label="Face detected" tone="green" />
              <StatusChip label={`${facesDetected} face${facesDetected === 1 ? '' : 's'}`} tone={facesDetected === 1 ? 'green' : 'amber'} />
            </ChipRow>
            <Text style={styles.resultTitle}>Selfie accepted</Text>
            <Text style={styles.resultText}>Faces detected: {facesDetected}</Text>
            <PrimaryButton label="Continue to liveness" onPress={() => router.push('/liveness')} />
          </InfoCard>
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

function getSelfieErrorMessage(error: unknown) {
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

  return error instanceof Error ? error.message : 'Could not upload selfie.';
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
  guidanceCard: {
    gap: 12,
    marginTop: 14,
  },
  guidanceRow: {
    flexDirection: 'row',
    gap: 10,
  },
  guidanceIndex: {
    color: vaColors.blueSoft,
    fontSize: 12,
    fontWeight: '900',
    width: 24,
  },
  guidanceText: {
    color: vaColors.subtle,
    flex: 1,
    fontSize: 14,
    lineHeight: 20,
  },
  actions: {
    gap: 12,
    marginTop: 14,
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
  },
});
