import { router } from 'expo-router';
import { useState } from 'react';
import { Alert, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { API_BASE_URL, postKyc } from '@/constants/api';
import {
  ChipRow,
  InfoCard,
  PrimaryButton,
  SecondaryButton,
  StatusChip,
  vaColors,
} from '@/components/visionauth-ui';

export default function StartScreen() {
  const [isStarting, setIsStarting] = useState(false);

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
    <SafeAreaView style={styles.screen}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.heroMark}>
          <View style={styles.heroRingOuter}>
            <View style={styles.heroRingInner}>
              <Text style={styles.heroMarkText}>VA</Text>
            </View>
          </View>
        </View>

        <Text style={styles.brand}>VisionAuth</Text>
        <Text style={styles.title}>Biometric identity verification</Text>
        <Text style={styles.subtitle}>
          A premium eKYC flow for Romanian ID OCR, selfie matching, randomized liveness, and secure audit review.
        </Text>

        <ChipRow>
          <StatusChip label="OCR" />
          <StatusChip label="Face Match" tone="green" />
          <StatusChip label="Liveness" tone="amber" />
          <StatusChip label="Audit Logs" tone="slate" />
        </ChipRow>

        <InfoCard title="Verification stack" style={styles.heroCard}>
          <FeatureRow title="ID OCR" text="Extracts identity fields and document face crop." />
          <FeatureRow title="Face match" text="Compares the selfie against the ID portrait." />
          <FeatureRow title="Liveness challenge" text="Uses randomized challenge-response video checks." />
          <FeatureRow title="Secure audit trail" text="Stores review evidence and operator decisions." last />
        </InfoCard>

        <InfoCard title="How it works" style={styles.howItWorksCard}>
          <View style={styles.previewSteps}>
            <PreviewStep number="01" title="Scan ID" text="Capture the front of the Romanian document." />
            <PreviewStep number="02" title="Verify face" text="Take a selfie and complete liveness." />
            <PreviewStep number="03" title="Decision" text="Receive a digital verification ticket." />
          </View>
        </InfoCard>

        <View style={styles.actions}>
          <PrimaryButton label={isStarting ? 'Starting verification...' : 'Start verification'} onPress={handleStartVerification} disabled={isStarting} />
          <SecondaryButton label="Admin logs" onPress={() => router.push('/admin')} />
        </View>
      </ScrollView>
    </SafeAreaView>
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
    backgroundColor: vaColors.background,
  },
  content: {
    flexGrow: 1,
    justifyContent: 'center',
    padding: 24,
  },
  heroMark: {
    alignItems: 'center',
    marginBottom: 22,
  },
  heroRingOuter: {
    alignItems: 'center',
    backgroundColor: 'rgba(37, 99, 235, 0.13)',
    borderColor: 'rgba(96, 165, 250, 0.22)',
    borderRadius: 999,
    borderWidth: 1,
    height: 104,
    justifyContent: 'center',
    width: 104,
  },
  heroRingInner: {
    alignItems: 'center',
    backgroundColor: vaColors.blue,
    borderRadius: 28,
    height: 64,
    justifyContent: 'center',
    shadowColor: vaColors.blue,
    shadowOffset: { width: 0, height: 14 },
    shadowOpacity: 0.35,
    shadowRadius: 22,
    width: 64,
  },
  heroMarkText: {
    color: 'white',
    fontSize: 18,
    fontWeight: '900',
  },
  brand: {
    color: vaColors.blueSoft,
    fontSize: 15,
    fontWeight: '900',
    letterSpacing: 1.2,
    marginBottom: 8,
    textAlign: 'center',
    textTransform: 'uppercase',
  },
  title: {
    color: vaColors.text,
    fontSize: 37,
    fontWeight: '900',
    lineHeight: 43,
    marginBottom: 14,
    textAlign: 'center',
  },
  subtitle: {
    color: vaColors.subtle,
    fontSize: 16,
    lineHeight: 24,
    marginBottom: 20,
    textAlign: 'center',
  },
  heroCard: {
    marginTop: 22,
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
    backgroundColor: 'rgba(30, 41, 59, 0.55)',
    borderRadius: 14,
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
