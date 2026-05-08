import { router, useLocalSearchParams } from 'expo-router';
import { useEffect, useRef, useState } from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { API_BASE_URL, getKyc } from '@/constants/api';
import {
  ChipRow,
  ConfidenceMeter,
  InfoCard,
  PageHeader,
  PrimaryButton,
  StatusChip,
  StepIndicator,
  vaColors,
  type ChipTone,
} from '@/components/visionauth-ui';

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
  session_locked?: boolean;
  reject_reason?: string;
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

      const faceMatch = await postFaceMatchResult();
      setMatchResult(faceMatch);
    } catch (loadError) {
      console.log('Result load error:', loadError);
      setError(loadError instanceof Error ? loadError.message : 'Could not load verification result.');
    } finally {
      setIsLoading(false);
    }
  };

  const decision = isLoading ? null : getBackendDecision(matchResult, getParamValue(params.decision));
  const ticket = getTicketState({ decision, error, isLoading, matchResult, status });
  const fullName = getDisplayName(status, params);
  const faceDistance = getFaceDistance(matchResult);
  const confidence = getConfidenceScore(matchResult);

  return (
    <SafeAreaView style={styles.screen}>
      <ScrollView contentContainerStyle={styles.content}>
        <StepIndicator currentStep={5} label="Verification result" />
        <PageHeader title="Digital ticket" subtitle="Generated from the current VisionAuth eKYC session." />

        <View style={[styles.ticketCard, { borderColor: ticket.color }]}>
          <View style={styles.ticketHeader}>
            <View>
              <Text style={styles.brand}>VisionAuth</Text>
              <Text style={styles.ticketType}>Secure verification pass</Text>
            </View>
            <VerificationCode sessionId={status?.session_id} />
          </View>

          <View style={[styles.badge, { backgroundColor: ticket.badgeBackground, borderColor: ticket.color }]}>
            <Text style={[styles.badgeText, { color: ticket.color }]}>{ticket.badge}</Text>
          </View>

          <Text style={styles.ticketTitle}>{ticket.title}</Text>
          <Text style={styles.ticketDescription}>{ticket.description}</Text>

          <View style={styles.meterCard}>
            <ConfidenceMeter value={confidence} label="Face match confidence" tone={ticket.tone} />
            <Text style={styles.distanceText}>Distance: {isLoading ? 'Checking...' : faceDistance}</Text>
          </View>

          <ChipRow>
            <StatusChip label={status?.document_uploaded ? 'ID captured' : 'ID pending'} tone={status?.document_uploaded ? 'green' : 'slate'} />
            <StatusChip label={status?.selfie_uploaded ? 'Selfie captured' : 'Selfie pending'} tone={status?.selfie_uploaded ? 'green' : 'slate'} />
            <StatusChip label={status?.liveness_passed ? 'Liveness passed' : 'Liveness pending'} tone={status?.liveness_passed ? 'green' : 'amber'} />
          </ChipRow>
        </View>

        <InfoCard title="Identity" style={styles.section}>
          <DetailRow label="Name" value={fullName} />
          <DetailRow label="Backend decision" value={isLoading ? 'Checking...' : decision ?? ticket.badge} />
        </InfoCard>

        <InfoCard title="Session" style={styles.section}>
          <DetailRow label="Session ID" value={isLoading ? 'Checking...' : String(status?.session_id ?? 'Unavailable')} />
          <DetailRow label="Status" value={String(matchResult?.session_status || decision || 'Checking')} />
        </InfoCard>

        <InfoCard title="Biometric result" style={styles.section}>
          <DetailRow label="Face match distance" value={isLoading ? 'Checking...' : faceDistance} />
          <DetailRow label="Liveness" value={status?.liveness_passed ? 'Passed' : isLoading ? 'Checking...' : 'Pending'} />
        </InfoCard>

        <InfoCard title="Timestamp" style={styles.section}>
          <DetailRow label="Generated" value={new Date().toLocaleString()} />
        </InfoCard>

        {error ? <Text style={styles.errorText}>{error}</Text> : null}

        <View style={styles.actions}>
          <PrimaryButton label="Start new verification" onPress={returnToStart} />
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

async function postFaceMatchResult() {
  const response = await fetch(`${API_BASE_URL}/kyc/face-match`, {
    method: 'POST',
  });
  const data = (await response.json()) as FaceMatchResult;

  if (!response.ok || (data.ok === false && !data.session_locked && !data.reject_reason)) {
    throw new Error(data.error || 'Could not load verification result.');
  }

  return data;
}

function VerificationCode({ sessionId }: { sessionId?: string }) {
  const code = String(sessionId || 'VISIONAUTH').replaceAll('-', '').slice(0, 12).toUpperCase();
  return (
    <View style={styles.codeBlock}>
      {Array.from({ length: 16 }).map((_, index) => (
        <View key={index} style={[styles.codePixel, code.charCodeAt(index % code.length) % 2 === 0 && styles.codePixelActive]} />
      ))}
    </View>
  );
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
  const rawDecision = String(
    matchResult?.session_status || matchResult?.decision || matchResult?.final_decision || fallbackDecision || ''
  ).toUpperCase();

  if (rawDecision) {
    return rawDecision;
  }

  if (matchResult?.session_locked || matchResult?.reject_reason) {
    return 'REJECTED';
  }

  if (matchResult?.passed === true) {
    return 'ACCEPTED';
  }

  return null;
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
      description: 'Please wait while VisionAuth processes your identity.',
      color: '#60A5FA',
      badgeBackground: '#172554',
      tone: 'blue' as ChipTone,
    };
  }

  if (decision === 'ACCEPTED' || decision === 'VERIFIED' || decision === 'APPROVED') {
    return {
      badge: 'VERIFIED',
      title: 'Identity verified',
      description: 'Document, selfie, liveness, and face match checks passed successfully.',
      color: '#22C55E',
      badgeBackground: '#123820',
      tone: 'green' as ChipTone,
    };
  }

  if (decision === 'REJECTED') {
    return {
      badge: 'REJECTED',
      title: 'Identity rejected',
      description: 'The session did not meet the configured biometric verification thresholds.',
      color: '#EF4444',
      badgeBackground: '#3F1218',
      tone: 'red' as ChipTone,
    };
  }

  if (decision === 'MANUAL_REVIEW') {
    return {
      badge: 'MANUAL REVIEW',
      title: 'Manual review required',
      description: 'The result requires operator review because the confidence is borderline.',
      color: '#F59E0B',
      badgeBackground: '#3A2708',
      tone: 'amber' as ChipTone,
    };
  }

  if (!status?.ready_for_face_match && !error) {
    return {
      badge: 'CHECKING',
      title: 'Checking verification...',
      description: 'Please wait while VisionAuth processes your identity.',
      color: '#60A5FA',
      badgeBackground: '#172554',
      tone: 'blue' as ChipTone,
    };
  }

  if (error) {
    return {
      badge: 'UNAVAILABLE',
      title: 'Result unavailable',
      description: 'The backend request failed before a verification decision could be loaded.',
      color: '#EF4444',
      badgeBackground: '#3F1218',
      tone: 'red' as ChipTone,
    };
  }

  return {
    badge: matchResult?.passed ? 'VERIFIED' : 'CHECKING',
    title: 'Checking verification...',
    description: 'Please wait while VisionAuth processes your identity.',
    color: '#60A5FA',
    badgeBackground: '#172554',
    tone: 'blue' as ChipTone,
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

function getConfidenceScore(matchResult: FaceMatchResult | null) {
  const distance = matchResult?.distance ?? matchResult?.face_match_distance ?? matchResult?.final_face_match_distance;
  const reviewThreshold = matchResult?.review_threshold ?? 0.60;

  if (typeof distance !== 'number' || !Number.isFinite(distance)) {
    return null;
  }

  return Math.max(0, Math.min(1, 1 - distance / reviewThreshold));
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
  ticketCard: {
    backgroundColor: 'rgba(17, 24, 39, 0.98)',
    borderRadius: 24,
    borderWidth: 1.5,
    gap: 16,
    padding: 20,
    shadowColor: '#020617',
    shadowOffset: { width: 0, height: 18 },
    shadowOpacity: 0.28,
    shadowRadius: 26,
  },
  ticketHeader: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  brand: {
    color: vaColors.text,
    fontSize: 20,
    fontWeight: '900',
    letterSpacing: 0.4,
  },
  ticketType: {
    color: vaColors.muted,
    fontSize: 12,
    fontWeight: '900',
    marginTop: 4,
    textTransform: 'uppercase',
  },
  codeBlock: {
    backgroundColor: '#E2E8F0',
    borderRadius: 8,
    flexDirection: 'row',
    flexWrap: 'wrap',
    height: 46,
    padding: 5,
    width: 46,
  },
  codePixel: {
    backgroundColor: 'transparent',
    borderRadius: 1,
    height: 8,
    margin: 1,
    width: 8,
  },
  codePixelActive: {
    backgroundColor: '#0F172A',
  },
  badge: {
    alignSelf: 'flex-start',
    borderRadius: 999,
    borderWidth: 1,
    paddingHorizontal: 18,
    paddingVertical: 9,
  },
  badgeText: {
    fontSize: 13,
    fontWeight: '900',
    letterSpacing: 1,
  },
  ticketTitle: {
    color: vaColors.text,
    fontSize: 32,
    fontWeight: '900',
    lineHeight: 38,
  },
  ticketDescription: {
    color: vaColors.subtle,
    fontSize: 15,
    lineHeight: 22,
  },
  meterCard: {
    backgroundColor: 'rgba(15, 23, 42, 0.9)',
    borderColor: 'rgba(148, 163, 184, 0.12)',
    borderRadius: 16,
    borderWidth: 1,
    gap: 10,
    padding: 14,
  },
  distanceText: {
    color: vaColors.muted,
    fontSize: 13,
    fontWeight: '700',
  },
  section: {
    marginTop: 14,
  },
  detailRow: {
    borderBottomColor: 'rgba(148, 163, 184, 0.10)',
    borderBottomWidth: 1,
    gap: 6,
    paddingBottom: 12,
    marginBottom: 12,
  },
  detailLabel: {
    color: vaColors.muted,
    fontSize: 12,
    fontWeight: '900',
    letterSpacing: 0.7,
    textTransform: 'uppercase',
  },
  detailValue: {
    color: '#E2E8F0',
    fontSize: 15,
    fontWeight: '800',
  },
  errorText: {
    color: '#FCA5A5',
    fontSize: 13,
    lineHeight: 19,
    marginTop: 14,
  },
  actions: {
    marginTop: 24,
  },
});
