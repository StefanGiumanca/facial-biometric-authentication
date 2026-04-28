import React, { useCallback, useEffect, useState } from 'react';
import { Alert, FlatList, Pressable, RefreshControl, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { router, useLocalSearchParams } from 'expo-router';
import { getAdminSessions, AdminSession } from '@/constants/api';

export default function AdminSessionsScreen() {
  const { adminKey } = useLocalSearchParams<{ adminKey: string }>();
  const [sessions, setSessions] = useState<AdminSession[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const loadSessions = useCallback(async () => {
    if (!adminKey) {
      Alert.alert('Error', 'Admin key missing');
      router.back();
      return;
    }

    try {
      const response = await getAdminSessions(adminKey, 50, 0);
      if (response.ok && response.sessions) {
        setSessions(response.sessions);
      } else {
        Alert.alert('Error', 'Failed to load sessions');
      }
    } catch (error) {
      console.error('Error loading sessions:', error);
      Alert.alert('Error', 'Failed to load sessions');
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  }, [adminKey]);

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'VERIFIED':
      case 'ACCEPTED':
        return '#10B981'; // green
      case 'REJECTED':
        return '#EF4444'; // red
      case 'MANUAL_REVIEW':
        return '#F59E0B'; // amber
      default:
        return '#60A5FA'; // blue
    }
  };

  const getStatusBadge = (status: string) => {
    const color = getStatusColor(status);
    return (
      <View style={[styles.statusBadge, { backgroundColor: color }]}>
        <Text style={styles.statusText}>{status}</Text>
      </View>
    );
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const shortSessionId = (id: string) => id.substring(0, 8) + '...';

  const handleSessionPress = (sessionId: string) => {
    router.push({
      pathname: '/admin/session/[id]',
      params: { id: sessionId, adminKey },
    });
  };

  const renderSessionCard = ({ item }: { item: AdminSession }) => {
    const userName = item.first_name && item.last_name ? `${item.first_name} ${item.last_name}` : 'Unknown';
    const updatedTime = formatDate(item.updated_at);

    return (
      <Pressable
        style={({ pressed }) => [styles.card, pressed && styles.cardPressed]}
        onPress={() => handleSessionPress(item.session_id)}>
        <View style={styles.cardHeader}>
          <View style={styles.userInfo}>
            <Text style={styles.userName}>{userName}</Text>
            <Text style={styles.sessionId}>{shortSessionId(item.session_id)}</Text>
          </View>
          {getStatusBadge(item.final_decision || 'IN_PROGRESS')}
        </View>

        <View style={styles.cardDetails}>
          <Text style={styles.detailLabel}>Updated:</Text>
          <Text style={styles.detailValue}>{updatedTime}</Text>
        </View>

        {item.face_match_decision && (
          <View style={styles.cardDetails}>
            <Text style={styles.detailLabel}>Face Match:</Text>
            <Text style={styles.detailValue}>{item.face_match_decision}</Text>
          </View>
        )}

        <View style={styles.cardDetails}>
          <Text style={styles.detailLabel}>Liveness:</Text>
          <Text style={[styles.detailValue, { color: item.liveness_passed ? '#10B981' : '#EF4444' }]}>
            {item.liveness_passed ? 'Passed' : 'Failed'}
          </Text>
        </View>
      </Pressable>
    );
  };

  if (isLoading) {
    return (
      <SafeAreaView style={styles.screen}>
        <View style={styles.loadingContainer}>
          <Text style={styles.loadingText}>Loading sessions...</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.screen}>
      <View style={styles.header}>
        <View>
          <Text style={styles.title}>Admin Sessions</Text>
          <Text style={styles.subtitle}>Total: {sessions.length} sessions</Text>
        </View>
        <Pressable
          style={({ pressed }) => [styles.backBtn, pressed && styles.backBtnPressed]}
          onPress={() => router.push('/')}>
          <Text style={styles.backBtnText}>Back</Text>
        </Pressable>
      </View>

      {sessions.length === 0 ? (
        <View style={styles.emptyContainer}>
          <Text style={styles.emptyText}>No sessions found</Text>
        </View>
      ) : (
        <FlatList
          data={sessions}
          keyExtractor={(item) => item.session_id}
          renderItem={renderSessionCard}
          contentContainerStyle={styles.list}
          refreshControl={<RefreshControl refreshing={isRefreshing} onRefresh={loadSessions} />}
        />
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: '#0B1220',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 24,
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#1E293B',
  },
  title: {
    color: 'white',
    fontSize: 28,
    fontWeight: '800',
  },
  subtitle: {
    color: '#94A3B8',
    fontSize: 14,
    marginTop: 4,
  },
  backBtn: {
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
  list: {
    padding: 16,
    gap: 12,
  },
  card: {
    backgroundColor: '#1E293B',
    borderRadius: 8,
    padding: 16,
    borderWidth: 1,
    borderColor: '#334155',
  },
  cardPressed: {
    opacity: 0.8,
    backgroundColor: '#334155',
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  userInfo: {
    flex: 1,
  },
  userName: {
    color: 'white',
    fontSize: 16,
    fontWeight: '700',
  },
  sessionId: {
    color: '#94A3B8',
    fontSize: 12,
    marginTop: 4,
  },
  statusBadge: {
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 4,
  },
  statusText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
  },
  cardDetails: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  detailLabel: {
    color: '#94A3B8',
    fontSize: 12,
  },
  detailValue: {
    color: '#E2E8F0',
    fontSize: 12,
    fontWeight: '500',
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
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  emptyText: {
    color: '#64748B',
    fontSize: 16,
  },
});
