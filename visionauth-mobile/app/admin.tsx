import React, { useState } from 'react';
import { Alert, Pressable, StyleSheet, Text, TextInput, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { router } from 'expo-router';
import { API_BASE_URL } from '@/constants/api';

export default function AdminAuthScreen() {
  const [adminKey, setAdminKey] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleLogin = async () => {
    if (!adminKey.trim()) {
      Alert.alert('Error', 'Please enter the admin key');
      return;
    }

    setIsLoading(true);
    try {
      // Test the key by making a minimal request to sessions endpoint
      const response = await fetch(`${API_BASE_URL}/admin/sessions?limit=1`, {
        headers: {
          'X-Admin-Key': adminKey,
        },
      });

      if (response.status === 403) {
        Alert.alert('Error', 'Invalid admin key');
        setIsLoading(false);
        return;
      }

      if (!response.ok) {
        Alert.alert('Error', 'Failed to authenticate');
        setIsLoading(false);
        return;
      }

      // Key is valid, navigate to sessions list
      // Pass adminKey via navigation params
      router.push({
        pathname: '/admin/sessions',
        params: { adminKey },
      });
    } catch (error) {
      console.error('Auth error:', error);
      Alert.alert('Error', 'Could not connect to backend');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.screen}>
      <View style={styles.container}>
        <Text style={styles.title}>Admin Access</Text>
        <Text style={styles.subtitle}>Enter admin key to view audit logs and sessions</Text>

        <View style={styles.form}>
          <Text style={styles.label}>Admin Key</Text>
          <TextInput
            style={styles.input}
            placeholder="Enter admin key"
            placeholderTextColor="#94A3B8"
            secureTextEntry
            value={adminKey}
            onChangeText={setAdminKey}
            editable={!isLoading}
          />

          <Pressable
            style={({ pressed }) => [styles.button, pressed && styles.buttonPressed, isLoading && styles.buttonDisabled]}
            onPress={handleLogin}
            disabled={isLoading}>
            <Text style={styles.buttonText}>{isLoading ? 'Authenticating...' : 'Access Admin'}</Text>
          </Pressable>

          <Pressable
            style={({ pressed }) => [styles.backButton, pressed && styles.backButtonPressed]}
            onPress={() => router.back()}
            disabled={isLoading}>
            <Text style={styles.backButtonText}>Back</Text>
          </Pressable>
        </View>

        <View style={styles.info}>
          <Text style={styles.infoText}>This section is for admin access only.</Text>
          <Text style={styles.infoText}>You can inspect KYC sessions and audit logs.</Text>
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: '#0B1220',
  },
  container: {
    flex: 1,
    paddingHorizontal: 24,
    paddingVertical: 20,
    justifyContent: 'center',
  },
  title: {
    color: 'white',
    fontSize: 32,
    fontWeight: '800',
    marginBottom: 8,
  },
  subtitle: {
    color: '#C7D2FE',
    fontSize: 16,
    lineHeight: 24,
    marginBottom: 32,
  },
  form: {
    marginBottom: 40,
  },
  label: {
    color: '#94A3B8',
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 8,
  },
  input: {
    backgroundColor: '#1E293B',
    borderWidth: 1,
    borderColor: '#334155',
    borderRadius: 8,
    paddingHorizontal: 16,
    paddingVertical: 12,
    color: 'white',
    fontSize: 16,
    marginBottom: 20,
  },
  button: {
    alignItems: 'center',
    backgroundColor: '#2563EB',
    borderRadius: 8,
    paddingVertical: 14,
    marginBottom: 12,
  },
  buttonPressed: {
    opacity: 0.82,
  },
  buttonDisabled: {
    opacity: 0.6,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '700',
  },
  backButton: {
    alignItems: 'center',
    backgroundColor: '#475569',
    borderRadius: 8,
    paddingVertical: 12,
  },
  backButtonPressed: {
    opacity: 0.82,
  },
  backButtonText: {
    color: '#E2E8F0',
    fontSize: 16,
    fontWeight: '600',
  },
  info: {
    backgroundColor: '#1E293B',
    borderRadius: 8,
    padding: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#60A5FA',
  },
  infoText: {
    color: '#CBD5E1',
    fontSize: 13,
    lineHeight: 20,
    marginBottom: 4,
  },
});
