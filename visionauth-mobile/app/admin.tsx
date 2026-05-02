import { router } from 'expo-router';
import { Linking, Pressable, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { LOCAL_WEB_ADMIN_URL, WEB_ADMIN_URL } from '@/constants/api';

export default function AdminDashboardInfoScreen() {
  const openDashboard = async () => {
    await Linking.openURL(WEB_ADMIN_URL);
  };

  return (
    <SafeAreaView style={styles.screen}>
      <View style={styles.container}>
        <Text style={styles.eyebrow}>Admin review</Text>
        <Text style={styles.title}>Use the web dashboard for manual decisions</Text>
        <Text style={styles.subtitle}>
          The mobile app is reserved for the applicant KYC flow. Audit inspection, image comparison,
          and approve/reject decisions are handled from the VisionAuth web admin dashboard.
        </Text>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Open it on your PC</Text>
          <Text style={styles.cardText}>
            Start the web admin app, then open this address in your desktop browser:
          </Text>
          <Text style={styles.urlText}>{LOCAL_WEB_ADMIN_URL}</Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Open it from this phone</Text>
          <Text style={styles.cardText}>
            If the phone and PC are on the same Wi-Fi, this opens the dashboard in the phone browser:
          </Text>
          <Text style={styles.urlText}>{WEB_ADMIN_URL}</Text>
          <Pressable style={({ pressed }) => [styles.button, pressed && styles.buttonPressed]} onPress={openDashboard}>
            <Text style={styles.buttonText}>Open dashboard in browser</Text>
          </Pressable>
        </View>

        <View style={styles.notice}>
          <Text style={styles.noticeText}>
            Sessions appear after a KYC flow starts and stores data in PostgreSQL. If the dashboard is
            empty, complete a verification first or call the session start endpoint for a test record.
          </Text>
        </View>

        <Pressable style={({ pressed }) => [styles.backButton, pressed && styles.buttonPressed]} onPress={() => router.back()}>
          <Text style={styles.backButtonText}>Back</Text>
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
  container: {
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: 24,
    paddingVertical: 20,
  },
  eyebrow: {
    color: '#60A5FA',
    fontSize: 15,
    fontWeight: '800',
    letterSpacing: 0.8,
    marginBottom: 10,
    textTransform: 'uppercase',
  },
  title: {
    color: 'white',
    fontSize: 30,
    fontWeight: '800',
    lineHeight: 37,
    marginBottom: 14,
  },
  subtitle: {
    color: '#C7D2FE',
    fontSize: 16,
    lineHeight: 24,
    marginBottom: 22,
  },
  card: {
    backgroundColor: '#1E293B',
    borderColor: '#334155',
    borderRadius: 8,
    borderWidth: 1,
    marginBottom: 12,
    padding: 16,
  },
  cardTitle: {
    color: 'white',
    fontSize: 16,
    fontWeight: '800',
    marginBottom: 8,
  },
  cardText: {
    color: '#CBD5E1',
    fontSize: 14,
    lineHeight: 20,
    marginBottom: 10,
  },
  urlText: {
    color: '#93C5FD',
    fontSize: 14,
    fontWeight: '700',
    marginBottom: 12,
  },
  button: {
    alignItems: 'center',
    backgroundColor: '#2563EB',
    borderRadius: 8,
    paddingVertical: 13,
  },
  buttonPressed: {
    opacity: 0.82,
  },
  buttonText: {
    color: 'white',
    fontSize: 15,
    fontWeight: '800',
  },
  notice: {
    backgroundColor: '#111827',
    borderLeftColor: '#F59E0B',
    borderLeftWidth: 4,
    borderRadius: 8,
    marginBottom: 12,
    padding: 14,
  },
  noticeText: {
    color: '#E5E7EB',
    fontSize: 13,
    lineHeight: 20,
  },
  backButton: {
    alignItems: 'center',
    backgroundColor: '#334155',
    borderRadius: 8,
    paddingVertical: 12,
  },
  backButtonText: {
    color: '#E2E8F0',
    fontSize: 15,
    fontWeight: '700',
  },
});
