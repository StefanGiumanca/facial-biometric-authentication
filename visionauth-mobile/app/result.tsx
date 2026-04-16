import { router } from 'expo-router';
import { useEffect, useRef, useState } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { getKyc, postKyc } from '@/constants/api';

type SessionStatus = {
  ok?: boolean;
  session_id?: string;
  document_uploaded?: boolean;
  id_face_extracted?: boolean;
  selfie_uploaded?: boolean;
  liveness_passed?: boolean;
  ready_for_face_match?: boolean;
  [key: string]: unknown;
};

type FaceMatchResult = {
  ok?: boolean;
  decision?: string;
  distance?: number;
  accept_threshold?: number;
  review_threshold?: number;
  error?: string;
  [key: string]: unknown;
};

export default function ResultScreen() {
  const hasLoaded = useRef(false);
  const [status, setStatus] = useState<SessionStatus | null>(null);
  const [matchResult, setMatchResult] = useState<FaceMatchResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (hasLoaded.current) {
      return;
    }

    hasLoaded.current = true;
    loadResult();
  }, []);

  const loadResult = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const currentStatus = (await getKyc('/kyc/session/status')) as SessionStatus;
      setStatus(currentStatus);

      if (!currentStatus.ready_for_face_match) {
        setMatchResult(null);
        return;
      }

      const faceMatch = (await postKyc('/kyc/face-match')) as FaceMatchResult;
      setMatchResult(faceMatch);
    } catch (loadError) {
      console.log('Result load error:', loadError);
      setError(loadError instanceof Error ? loadError.message : 'Could not load verification result.');
    } finally {
      setIsLoading(false);
    }
  };

  const decision = matchResult?.decision;
  const isAccepted = decision === 'ACCEPTED';
  const needsReview = decision === 'MANUAL_REVIEW';
  const isRejected = decision === 'REJECTED';

  return (
    <SafeAreaView style={styles.screen}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.step}>Step 4 of 4</Text>
        <Text style={styles.title}>Verification result</Text>

        {isLoading && <Text style={styles.subtitle}>Checking session status and face match result...</Text>}

        {!isLoading && error && (
          <View style={styles.resultPanel}>
            <Text style={styles.resultTitle}>Result unavailable</Text>
            <Text style={styles.resultText}>{error}</Text>
          </View>
        )}

        {!isLoading && !error && status && !status.ready_for_face_match && (
          <View style={styles.resultPanel}>
            <Text style={styles.resultTitle}>Verification incomplete</Text>
            <Text style={styles.resultText}>
              Complete document upload, selfie upload, and liveness before face matching.
            </Text>
            <StatusRow label="Document uploaded" value={status.document_uploaded} />
            <StatusRow label="ID face extracted" value={status.id_face_extracted} />
            <StatusRow label="Selfie uploaded" value={status.selfie_uploaded} />
            <StatusRow label="Liveness passed" value={status.liveness_passed} />
          </View>
        )}

        {!isLoading && !error && matchResult && (
          <View
            style={[
              styles.resultPanel,
              isAccepted && styles.acceptedPanel,
              needsReview && styles.reviewPanel,
              isRejected && styles.rejectedPanel,
            ]}>
            <Text style={styles.resultTitle}>{decisionLabel(decision)}</Text>
            <Text style={styles.resultText}>
              Face distance: {typeof matchResult.distance === 'number' ? matchResult.distance.toFixed(4) : 'n/a'}
            </Text>
            <Text style={styles.resultText}>
              Accept threshold: {String(matchResult.accept_threshold ?? 'n/a')}
            </Text>
            <Text style={styles.resultText}>
              Review threshold: {String(matchResult.review_threshold ?? 'n/a')}
            </Text>
          </View>
        )}

        <View style={styles.actions}>
          <Pressable style={styles.secondaryButton} onPress={loadResult}>
            <Text style={styles.secondaryButtonText}>Refresh status</Text>
          </Pressable>
          <Pressable style={styles.button} onPress={() => router.replace('/')}>
            <Text style={styles.buttonText}>Start new session</Text>
          </Pressable>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

function StatusRow({ label, value }: { label: string; value?: boolean }) {
  return (
    <View style={styles.statusRow}>
      <Text style={styles.statusLabel}>{label}</Text>
      <Text style={[styles.statusValue, value ? styles.statusPassed : styles.statusMissing]}>
        {value ? 'Yes' : 'No'}
      </Text>
    </View>
  );
}

function decisionLabel(decision?: string) {
  if (decision === 'ACCEPTED') {
    return 'Identity verified';
  }

  if (decision === 'MANUAL_REVIEW') {
    return 'Manual review needed';
  }

  if (decision === 'REJECTED') {
    return 'Identity rejected';
  }

  return 'Face match complete';
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
  resultPanel: {
    backgroundColor: '#111827',
    borderColor: '#334155',
    borderRadius: 8,
    borderWidth: 1,
    gap: 10,
    marginTop: 12,
    padding: 16,
  },
  acceptedPanel: {
    borderColor: '#16A34A',
  },
  reviewPanel: {
    borderColor: '#F59E0B',
  },
  rejectedPanel: {
    borderColor: '#DC2626',
  },
  resultTitle: {
    color: 'white',
    fontSize: 22,
    fontWeight: '800',
  },
  resultText: {
    color: '#CBD5E1',
    fontSize: 15,
    lineHeight: 22,
  },
  statusRow: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  statusLabel: {
    color: '#CBD5E1',
    fontSize: 15,
  },
  statusValue: {
    fontSize: 15,
    fontWeight: '800',
  },
  statusPassed: {
    color: '#86EFAC',
  },
  statusMissing: {
    color: '#FCA5A5',
  },
  actions: {
    gap: 12,
    marginTop: 24,
  },
  secondaryButton: {
    alignItems: 'center',
    backgroundColor: '#1F2937',
    borderRadius: 8,
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
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '700',
  },
});
