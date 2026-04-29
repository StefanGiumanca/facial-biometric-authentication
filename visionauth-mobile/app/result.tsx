import { router, useLocalSearchParams } from 'expo-router';
import { useEffect, useRef, useState } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { getKyc, postKyc } from '@/constants/api';

type SessionStatus = {
  ok?: boolean;
  session_id?: string;
  first_name?: string;
  last_name?: string;
  session_data?: {
    reviewed_document_fields?: {
      first_name?: string | null;
      last_name?: string | null;
    } | null;
    document_fields?: {
      first_name?: string | null;
      last_name?: string | null;
    } | null;
  };
  document_uploaded?: boolean;
  id_face_extracted?: boolean;
  selfie_uploaded?: boolean;
  liveness_passed?: boolean;
  ready_for_face_match?: boolean;
  [key: string]: unknown;
};

type FaceMatchResult = {
  ok?: boolean;
  passed?: boolean;
  decision?: string;
  session_status?: string;
  reason?: string;
  distance?: number;
  accept_threshold?: number;
  review_threshold?: number;
  error?: string;
  [key: string]: unknown;
};

export default function ResultScreen() {
  const params = useLocalSearchParams<{
    first_name?: string;
    last_name?: string;
    name?: string;
    decision?: string;
    reason?: string;
  }>();
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

  const decision = getBackendDecision(matchResult, getParamValue(params.decision));
  const ticket = getTicketState({ decision, error, isLoading, matchResult, status });
  const fullName = getDisplayName(status, params);
  const faceDistance = getFaceDistance(matchResult);

  return (
    <SafeAreaView style={styles.screen}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.step}>Step 5 of 5</Text>
        <Text style={styles.title}>Verification pass</Text>
        <Text style={styles.subtitle}>Generated from the current eKYC session.</Text>

        <View style={[styles.ticketCard, { borderColor: ticket.color }]}>
          <View style={styles.ticketHeader}>
            <Text style={styles.brand}>VisionAuth</Text>
            <Text style={styles.ticketType}>Digital ticket</Text>
          </View>

          <View style={[styles.badge, { backgroundColor: ticket.badgeBackground, borderColor: ticket.color }]}>
            <Text style={[styles.badgeText, { color: ticket.color }]}>{ticket.badge}</Text>
          </View>

          <Text style={styles.ticketTitle}>{ticket.title}</Text>
          <Text style={styles.ticketDescription}>{ticket.description}</Text>

          <View style={styles.divider} />

          <DetailRow label="Name" value={fullName} />
          <DetailRow label="Session ID" value={String(status?.session_id ?? 'Unavailable')} />
          <DetailRow label="Timestamp" value={new Date().toLocaleString()} />
          <DetailRow label="Backend decision" value={decision ?? ticket.badge} />
          <DetailRow label="Face match distance" value={faceDistance} />
        </View>

        <View style={styles.actions}>
          <Pressable style={styles.button} onPress={returnToStart}>
            <Text style={styles.buttonText}>Start new verification</Text>
          </Pressable>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

function returnToStart() {
  if (router.canDismiss()) {
    router.dismissAll();
  }

  router.replace('/');
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.detailRow}>
      <Text style={styles.detailLabel}>{label}</Text>
      <Text style={styles.detailValue}>{value}</Text>
    </View>
  );
}

function getBackendDecision(matchResult: FaceMatchResult | null, fallbackDecision?: string) {
  return String(matchResult?.session_status || matchResult?.decision || matchResult?.final_decision || fallbackDecision || '').toUpperCase() || null;
}

function getTicketState({
  decision,
  error,
  isLoading,
  matchResult,
  status,
}: {
  decision: string | null;
  error: string | null;
  isLoading: boolean;
  matchResult: FaceMatchResult | null;
  status: SessionStatus | null;
}) {
  if (isLoading && !decision) {
    return {
      badge: 'CHECKING',
      title: 'Checking verification...',
      description: 'Please wait while we process your identity.',
      color: '#60A5FA',
      badgeBackground: '#172554',
    };
  }

  if (decision === 'ACCEPTED' || decision === 'VERIFIED' || decision === 'APPROVED') {
    return {
      badge: 'VERIFIED',
      title: 'Identity verified',
      description: 'The identity verification passed successfully.',
      color: '#22C55E',
      badgeBackground: '#123820',
    };
  }

  if (decision === 'REJECTED') {
    return {
      badge: 'REJECTED',
      title: 'Identity rejected',
      description: 'The selfie did not meet the backend face-match threshold for this session.',
      color: '#EF4444',
      badgeBackground: '#3F1218',
    };
  }

  if (decision === 'MANUAL_REVIEW') {
    return {
      badge: 'MANUAL REVIEW',
      title: 'Manual review required',
      description: 'The result requires manual review because the confidence is borderline.',
      color: '#F59E0B',
      badgeBackground: '#3A2708',
    };
  }

  if (error || !status?.ready_for_face_match || !matchResult) {
    return {
      badge: 'UNAVAILABLE',
      title: 'Result unavailable',
      description: 'Error processing face-match result.',
      color: '#EF4444',
      badgeBackground: '#3F1218',
    };
  }

  return {
    badge: 'UNAVAILABLE',
    title: 'Result unavailable',
    description: 'Error processing face-match result.',
    color: '#EF4444',
    badgeBackground: '#3F1218',
  };
}

function getDisplayName(
  status: SessionStatus | null,
  params: { first_name?: string; last_name?: string; name?: string }
) {
  const paramName = getParamValue(params.name);
  if (paramName) {
    return paramName;
  }

  const reviewedFields = status?.session_data?.reviewed_document_fields;
  const documentFields = status?.session_data?.document_fields;
  const firstName = String(
    reviewedFields?.first_name || documentFields?.first_name || status?.first_name || getParamValue(params.first_name) || ''
  ).trim();
  const lastName = String(
    reviewedFields?.last_name || documentFields?.last_name || status?.last_name || getParamValue(params.last_name) || ''
  ).trim();
  const fullName = `${firstName} ${lastName}`.trim();
  return fullName || 'Unavailable';
}

function getParamValue(value?: string | string[]) {
  return Array.isArray(value) ? value[0] : value;
}

function getFaceDistance(matchResult: FaceMatchResult | null) {
  const distance = matchResult?.distance ?? matchResult?.face_match_distance ?? matchResult?.final_face_match_distance;
  return typeof distance === 'number' ? distance.toFixed(4) : 'Unavailable';
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: '#0B1220',
  },
  content: {
    flexGrow: 1,
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
  ticketCard: {
    backgroundColor: '#111827',
    borderRadius: 18,
    borderWidth: 1.5,
    padding: 20,
  },
  ticketHeader: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 24,
  },
  brand: {
    color: 'white',
    fontSize: 18,
    fontWeight: '900',
    letterSpacing: 0.5,
  },
  ticketType: {
    color: '#94A3B8',
    fontSize: 13,
    fontWeight: '800',
    letterSpacing: 0.8,
    textTransform: 'uppercase',
  },
  badge: {
    alignSelf: 'flex-start',
    borderRadius: 999,
    borderWidth: 1,
    marginBottom: 18,
    paddingHorizontal: 16,
    paddingVertical: 8,
  },
  badgeText: {
    fontSize: 13,
    fontWeight: '900',
    letterSpacing: 1,
  },
  ticketTitle: {
    color: 'white',
    fontSize: 30,
    fontWeight: '900',
    marginBottom: 10,
  },
  ticketDescription: {
    color: '#CBD5E1',
    fontSize: 15,
    lineHeight: 22,
  },
  divider: {
    backgroundColor: '#334155',
    height: 1,
    marginVertical: 18,
  },
  detailRow: {
    gap: 6,
    marginBottom: 13,
  },
  detailLabel: {
    color: '#94A3B8',
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 0.7,
    textTransform: 'uppercase',
  },
  detailValue: {
    color: '#E2E8F0',
    fontSize: 15,
    fontWeight: '700',
  },
  actions: {
    marginTop: 24,
  },
  button: {
    alignItems: 'center',
    backgroundColor: '#2563EB',
    borderRadius: 12,
    paddingVertical: 16,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '700',
  },
});
