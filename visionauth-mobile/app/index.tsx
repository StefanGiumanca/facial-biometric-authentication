import { router } from 'expo-router';
import { useState } from 'react';
import { Alert, Pressable, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { API_BASE_URL, postKyc } from '@/constants/api';

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
      <View style={styles.content}>
        <Text style={styles.eyebrow}>VisionAuth</Text>
        <Text style={styles.title}>Identity verification</Text>
        <Text style={styles.subtitle}>
          Scan your Romanian ID card, take a selfie, complete the blink check, and compare your face
          with the ID photo.
        </Text>

        <View style={styles.steps}>
          <Text style={styles.step}>1. Romanian ID OCR</Text>
          <Text style={styles.step}>2. Selfie capture</Text>
          <Text style={styles.step}>3. Blink liveness</Text>
          <Text style={styles.step}>4. Face match result</Text>
        </View>

        <Pressable
          style={({ pressed }) => [styles.button, pressed && styles.buttonPressed]}
          onPress={handleStartVerification}
          disabled={isStarting}>
          <Text style={styles.buttonText}>{isStarting ? 'Starting...' : 'Start verification'}</Text>
        </Pressable>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: '#0B1220',
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: 24,
  },
  eyebrow: {
    color: '#60A5FA',
    fontSize: 16,
    fontWeight: '700',
    marginBottom: 12,
  },
  title: {
    color: 'white',
    fontSize: 34,
    fontWeight: '800',
    marginBottom: 16,
  },
  subtitle: {
    color: '#C7D2FE',
    fontSize: 16,
    lineHeight: 24,
    marginBottom: 28,
  },
  steps: {
    gap: 10,
    marginBottom: 32,
  },
  step: {
    color: '#E5E7EB',
    fontSize: 15,
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
