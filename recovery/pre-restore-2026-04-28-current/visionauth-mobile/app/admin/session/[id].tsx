import React, { useCallback, useEffect, useState } from 'react';
import { Alert, ScrollView, StyleSheet, Text, View, Pressable } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { router, useLocalSearchParams } from 'expo-router';
import { getAdminSessionDetail, getAdminSessionLogs, SessionDetail, AuditLogEntry } from '@/constants/api';

export default function AdminSessionDetailScreen() {
  const { id: sessionId, adminKey } = useLocalSearchParams<{ id: string; adminKey: string }>();
  const [session, setSession] = useState<SessionDetail | null>(null);
  const [logs, setLogs] = useState<AuditLogEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const loadSessionData = useCallback(async () => {
    if (!sessionId || !adminKey) {
      Alert.alert('Error', 'Missing session ID or admin key');
      router.back();
      return;
    }

    try {
      const [sessionRes, logsRes] = await Promise.all([
        getAdminSessionDetail(adminKey, sessionId),
        getAdminSessionLogs(adminKey, sessionId),
      ]);

      if (sessionRes.ok && sessionRes.session) {
        setSession(sessionRes.session);
      } else {
        Alert.alert('Error', 'Failed to load session details');
      }

      if (logsRes.ok && logsRes.logs) {
        setLogs(logsRes.logs);
      }
    } catch (error) {
      console.error('Error loading session data:', error);
      Alert.alert('Error', 'Failed to load session data');
    } finally {
      setIsLoading(false);
    }
  }, [adminKey, sessionId]);

  useEffect(() => {
    loadSessionData();
  }, [loadSessionData]);

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'VERIFIED':
      case 'ACCEPTED':
        return '#10B981';
      case 'REJECTED':
        return '#EF4444';
      case 'MANUAL_REVIEW':
        return '#F59E0B';
      default:
        return '#60A5FA';
    }
  };

  if (isLoading || !session) {
    return (
      <SafeAreaView style={styles.screen}>
        <View style={styles.loadingContainer}>
          <Text style={styles.loadingText}>Loading session details...</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.screen}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Pressable
            style={({ pressed }) => [styles.backBtn, pressed && styles.backBtnPressed]}
            onPress={() => router.back()}>
            <Text style={styles.backBtnText}>← Back</Text>
          </Pressable>
        </View>

        {/* Identity Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Identity Information</Text>
          <View style={styles.card}>
            <DetailRow label="First Name" value={session.first_name || 'N/A'} />
            <DetailRow label="Last Name" value={session.last_name || 'N/A'} />
            <DetailRow label="CNP" value={session.cnp || 'N/A'} />
            <DetailRow label="Series/Number" value={session.series_number || 'N/A'} />
          </View>
        </View>

        {/* Verification Results Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Verification Results</Text>
          <View style={styles.card}>
            <DetailRow
              label="Status"
              value={session.final_decision || 'PENDING'}
              valueColor={getStatusColor(session.final_decision || 'PENDING')}
            />
            <DetailRow
              label="Liveness"
              value={session.liveness_passed ? 'Passed' : session.liveness_passed === false ? 'Failed' : 'Not Done'}
              valueColor={session.liveness_passed ? '#10B981' : session.liveness_passed === false ? '#EF4444' : '#60A5FA'}
            />
            <DetailRow label="Face Match Decision" value={session.face_match_decision || 'N/A'} />
            <DetailRow
              label="Face Match Distance"
              value={session.face_match_distance !== null ? session.face_match_distance.toFixed(3) : 'N/A'}
            />
            <DetailRow label="Session Status" value={session.status} />
          </View>
        </View>

        {/* Media Paths Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Media & Files</Text>
          <View style={styles.card}>
            <PathRow label="Document" path={session.document_path} />
            <PathRow label="ID Face" path={session.id_face_path} />
            <PathRow label="Selfie" path={session.selfie_path} />
            <PathRow label="Liveness Video" path={session.liveness_video_path} />
          </View>
        </View>

        {/* OCR Data Section */}
        {session.raw_ocr_text && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>OCR Raw Text</Text>
            <View style={styles.card}>
              <Text style={styles.ocrText}>{session.raw_ocr_text}</Text>
            </View>
          </View>
        )}

        {/* Embeddings Section */}
        {session.embeddings && session.embeddings.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Embeddings Metadata</Text>
            {session.embeddings.map((emb) => (
              <View key={emb.id} style={styles.card}>
                <DetailRow label="Type" value={emb.embedding_type} />
                <DetailRow label="Vector Length" value={emb.vector_length?.toString() || 'Unknown'} />
                {emb.vector_preview && (
                  <View style={styles.previewRow}>
                    <Text style={styles.previewLabel}>Preview (first 5):</Text>
                    <Text style={styles.previewValue}>
                      [{emb.vector_preview.map((v) => v.toFixed(3)).join(', ')}...]
                    </Text>
                  </View>
                )}
                <DetailRow label="Created" value={formatDate(emb.created_at)} />
              </View>
            ))}
          </View>
        )}

        {/* Audit Log Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Audit Log Timeline</Text>
          {logs.length === 0 ? (
            <Text style={styles.noLogsText}>No audit logs available</Text>
          ) : (
            logs.map((log, index) => (
              <View key={log.id} style={[styles.logEntry, index === logs.length - 1 && { borderLeftWidth: 0 }]}>
                <View style={styles.logTime}>
                  <Text style={styles.logTimestamp}>{formatDate(log.created_at)}</Text>
                </View>
                <View style={styles.logContent}>
                  <Text style={styles.logEventType}>{log.event_type}</Text>
                  <Text style={styles.logMessage}>{log.message}</Text>
                </View>
              </View>
            ))
          )}
        </View>

        {/* Session Info Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Session Info</Text>
          <View style={styles.card}>
            <DetailRow label="Session ID" value={session.session_id} />
            <DetailRow label="Created" value={formatDate(session.created_at)} />
            <DetailRow label="Updated" value={formatDate(session.updated_at)} />
          </View>
        </View>

        <View style={styles.spacer} />
      </ScrollView>
    </SafeAreaView>
  );
}

function DetailRow({
  label,
  value,
  valueColor = '#E2E8F0',
}: {
  label: string;
  value: string;
  valueColor?: string;
}) {
  return (
    <View style={styles.detailRow}>
      <Text style={styles.detailLabel}>{label}</Text>
      <Text style={[styles.detailValue, { color: valueColor }]} numberOfLines={2}>
        {value}
      </Text>
    </View>
  );
}

function PathRow({ label, path }: { label: string; path: string | null }) {
  const filename = path ? path.split('/').pop() : null;
  return (
    <View style={styles.detailRow}>
      <Text style={styles.detailLabel}>{label}</Text>
      <Text style={styles.pathValue} numberOfLines={1}>
        {filename || 'N/A'}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: '#0B1220',
  },
  scrollContent: {
    paddingBottom: 40,
  },
  header: {
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  backBtn: {
    alignSelf: 'flex-start',
    backgroundColor: '#1E293B',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 6,
  },
  backBtnPressed: {
    opacity: 0.7,
  },
  backBtnText: {
    color: '#60A5FA',
    fontSize: 14,
    fontWeight: '600',
  },
  section: {
    marginHorizontal: 16,
    marginTop: 20,
  },
  sectionTitle: {
    color: 'white',
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 12,
  },
  card: {
    backgroundColor: '#1E293B',
    borderRadius: 8,
    padding: 16,
    borderWidth: 1,
    borderColor: '#334155',
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#334155',
  },
  detailLabel: {
    color: '#94A3B8',
    fontSize: 13,
    fontWeight: '600',
    flex: 0.4,
  },
  detailValue: {
    color: '#E2E8F0',
    fontSize: 13,
    fontWeight: '500',
    flex: 0.6,
    textAlign: 'right',
  },
  pathValue: {
    color: '#60A5FA',
    fontSize: 13,
    fontWeight: '500',
    flex: 0.6,
    textAlign: 'right',
  },
  previewRow: {
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#334155',
  },
  previewLabel: {
    color: '#94A3B8',
    fontSize: 12,
    fontWeight: '600',
    marginBottom: 4,
  },
  previewValue: {
    color: '#60A5FA',
    fontSize: 12,
    fontFamily: 'monospace',
  },
  ocrText: {
    color: '#CBD5E1',
    fontSize: 12,
    lineHeight: 18,
    fontFamily: 'monospace',
  },
  logEntry: {
    marginBottom: 16,
    paddingLeft: 12,
    borderLeftWidth: 2,
    borderLeftColor: '#60A5FA',
  },
  logTime: {
    marginBottom: 4,
  },
  logTimestamp: {
    color: '#94A3B8',
    fontSize: 11,
    fontWeight: '600',
  },
  logContent: {
    backgroundColor: '#0F172A',
    borderRadius: 6,
    padding: 10,
  },
  logEventType: {
    color: '#60A5FA',
    fontSize: 13,
    fontWeight: '700',
    marginBottom: 2,
  },
  logMessage: {
    color: '#CBD5E1',
    fontSize: 12,
    lineHeight: 16,
  },
  noLogsText: {
    color: '#64748B',
    fontSize: 14,
    fontStyle: 'italic',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    color: '#94A3B8',
    fontSize: 16,
  },
  spacer: {
    height: 20,
  },
});
