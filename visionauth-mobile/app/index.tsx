import { router } from 'expo-router';
import { useEffect, useState } from 'react';
import { Alert, ScrollView, StyleSheet, Text, View } from 'react-native';
import Animated, { Easing, useAnimatedStyle, useSharedValue, withTiming } from 'react-native-reanimated';
import { SafeAreaView } from 'react-native-safe-area-context';

import { API_BASE_URL, postKyc } from '@/constants/api';
import {
  AppBackground,
  BiometricPulse,
  ChipRow,
  FeatureChip,
  InfoCard,
  PrimaryButton,
  SecondaryButton,
  vaColors,
} from '@/components/visionauth-ui';

export default function StartScreen() {
  const [isStarting, setIsStarting] = useState(false);
  const reveal = useSharedValue(0);

  useEffect(() => {
    reveal.value = withTiming(1, { duration: 720, easing: Easing.out(Easing.cubic) });
  }, [reveal]);

  const heroStyle = useAnimatedStyle(() => ({
    opacity: reveal.value,
    transform: [{ translateY: 18 - reveal.value * 18 }],
  }));

  const handleStartVerification = async () => {
    setIsStarting(true);

    try {
      await postKyc('/kyc/session/start');
      router.push('/document');
    } catch (error) {
      console.log('Start verification error:', error);
      Alert.alert('Could not start verification', `Check that the backend is reachable at ${API_BASE_URL}.`);
    } finally {
      setIsStarting(false);
    }
  };

  return (
    <AppBackground>
      <SafeAreaView style={styles.screen}>
        <ScrollView contentContainerStyle={styles.content}>
          <Animated.View style={[styles.hero, heroStyle]}>
            <View style={styles.brandRow}>
              <View style={styles.brandMark}>
                <Text style={styles.brandMarkText}>VA</Text>
              </View>
              <View>
                <Text style={styles.brand}>VisionAuth</Text>
                <Text style={styles.brandSubline}>Biometric eKYC suite</Text>
              </View>
            </View>

            <BiometricPulse />

            <Text style={styles.title}>Secure identity verification</Text>
            <Text style={styles.subtitle}>
              OCR, face match, liveness, and audit-ready evidence in one guided biometric flow.
            </Text>

            <ChipRow>
              <FeatureChip label="OCR" />
              <FeatureChip label="Face Match" tone="green" />
              <FeatureChip label="Liveness" tone="amber" />
              <FeatureChip label="Audit Trail" tone="slate" />
            </ChipRow>
          </Animated.View>

          <InfoCard title="Verification stack" style={styles.heroCard}>
            <FeatureRow title="ID intelligence" text="Romanian ID OCR with document face extraction." />
            <FeatureRow title="Biometric binding" text="Selfie comparison against the extracted ID portrait." />
            <FeatureRow title="Challenge-response" text="Randomized liveness video checks for presentation attacks." />
            <FeatureRow title="Operator evidence" text="Every session remains traceable for admin review." last />
          </InfoCard>

          <InfoCard title="How it works" style={styles.howItWorksCard}>
            <View style={styles.previewSteps}>
              <PreviewStep number="01" title="Scan ID" text="Capture a clear document image." />
              <PreviewStep number="02" title="Review data" text="Confirm extracted identity fields." />
              <PreviewStep number="03" title="Face check" text="Take a selfie and complete liveness." />
              <PreviewStep number="04" title="Ticket" text="Receive the final verification decision." />
            </View>
          </InfoCard>

          <View style={styles.actions}>
            <PrimaryButton label={isStarting ? 'Starting verification...' : 'Start verification'} onPress={handleStartVerification} disabled={isStarting} />
            <SecondaryButton label="Admin logs" onPress={() => router.push('/admin')} />
          </View>
        </ScrollView>
      </SafeAreaView>
    </AppBackground>
  );
}

function FeatureRow({ title, text, last = false }: { title: string; text: string; last?: boolean }) {
  return (
    <View style={[styles.featureRow, last && styles.featureRowLast]}>
      <View style={styles.featureDot} />
      <View style={styles.featureCopy}>
        <Text style={styles.featureTitle}>{title}</Text>
        <Text style={styles.featureText}>{text}</Text>
      </View>
    </View>
  );
}

function PreviewStep({ number, title, text }: { number: string; title: string; text: string }) {
  return (
    <View style={styles.previewStep}>
      <Text style={styles.previewNumber}>{number}</Text>
      <Text style={styles.previewTitle}>{title}</Text>
      <Text style={styles.previewText}>{text}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
  },
  content: {
    flexGrow: 1,
    justifyContent: 'center',
    padding: 24,
  },
  hero: {
    alignItems: 'center',
    gap: 18,
  },
  brandRow: {
    alignItems: 'center',
    alignSelf: 'stretch',
    flexDirection: 'row',
    gap: 12,
    marginBottom: 4,
  },
  brandMark: {
    alignItems: 'center',
    backgroundColor: vaColors.blue,
    borderColor: 'rgba(191, 219, 254, 0.36)',
    borderRadius: 16,
    borderWidth: 1,
    height: 46,
    justifyContent: 'center',
    width: 46,
  },
  brandMarkText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '900',
  },
  brand: {
    color: vaColors.text,
    fontSize: 20,
    fontWeight: '900',
  },
  brandSubline: {
    color: vaColors.muted,
    fontSize: 12,
    fontWeight: '800',
    marginTop: 2,
    textTransform: 'uppercase',
  },
  title: {
    color: vaColors.text,
    fontSize: 38,
    fontWeight: '900',
    lineHeight: 43,
    textAlign: 'center',
  },
  subtitle: {
    color: vaColors.subtle,
    fontSize: 16,
    lineHeight: 24,
    textAlign: 'center',
  },
  heroCard: {
    marginTop: 24,
  },
  howItWorksCard: {
    marginTop: 14,
  },
  featureRow: {
    borderBottomColor: 'rgba(148, 163, 184, 0.12)',
    borderBottomWidth: 1,
    flexDirection: 'row',
    gap: 12,
    paddingBottom: 12,
    marginBottom: 12,
  },
  featureRowLast: {
    borderBottomWidth: 0,
    marginBottom: 0,
    paddingBottom: 0,
  },
  featureDot: {
    backgroundColor: vaColors.blueSoft,
    borderRadius: 999,
    height: 9,
    marginTop: 5,
    width: 9,
  },
  featureCopy: {
    flex: 1,
  },
  featureTitle: {
    color: vaColors.text,
    fontSize: 14,
    fontWeight: '900',
    marginBottom: 4,
  },
  featureText: {
    color: vaColors.muted,
    fontSize: 13,
    lineHeight: 19,
  },
  previewSteps: {
    gap: 12,
  },
  previewStep: {
    backgroundColor: 'rgba(30, 41, 59, 0.46)',
    borderColor: 'rgba(148, 163, 184, 0.12)',
    borderRadius: 18,
    borderWidth: 1,
    padding: 13,
  },
  previewNumber: {
    color: vaColors.blueSoft,
    fontSize: 11,
    fontWeight: '900',
    marginBottom: 5,
  },
  previewTitle: {
    color: vaColors.text,
    fontSize: 15,
    fontWeight: '900',
    marginBottom: 4,
  },
  previewText: {
    color: vaColors.muted,
    fontSize: 13,
    lineHeight: 18,
  },
  actions: {
    gap: 12,
    marginTop: 18,
  },
});
