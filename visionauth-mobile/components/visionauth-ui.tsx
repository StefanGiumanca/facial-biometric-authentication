import type { ReactNode } from 'react';
import { Image, ImageSourcePropType, Pressable, StyleProp, StyleSheet, Text, View, ViewStyle } from 'react-native';

export const vaColors = {
  background: '#0B1220',
  panel: '#111827',
  panelStrong: '#1E293B',
  border: '#334155',
  borderSoft: '#243244',
  blue: '#2563EB',
  blueSoft: '#60A5FA',
  text: '#F8FAFC',
  muted: '#94A3B8',
  subtle: '#CBD5E1',
  green: '#22C55E',
  red: '#EF4444',
  amber: '#F59E0B',
  cyan: '#38BDF8',
};

export type ChipTone = 'blue' | 'green' | 'red' | 'amber' | 'slate';

const toneStyles: Record<ChipTone, { backgroundColor: string; borderColor: string; color: string }> = {
  blue: { backgroundColor: 'rgba(37, 99, 235, 0.16)', borderColor: 'rgba(96, 165, 250, 0.26)', color: '#BFDBFE' },
  green: { backgroundColor: 'rgba(34, 197, 94, 0.14)', borderColor: 'rgba(34, 197, 94, 0.28)', color: '#BBF7D0' },
  red: { backgroundColor: 'rgba(239, 68, 68, 0.14)', borderColor: 'rgba(239, 68, 68, 0.28)', color: '#FECACA' },
  amber: { backgroundColor: 'rgba(245, 158, 11, 0.14)', borderColor: 'rgba(245, 158, 11, 0.28)', color: '#FDE68A' },
  slate: { backgroundColor: 'rgba(148, 163, 184, 0.10)', borderColor: 'rgba(148, 163, 184, 0.18)', color: '#CBD5E1' },
};

export function ScreenShell({ children, centered = false }: { children: ReactNode; centered?: boolean }) {
  return <View style={[styles.shell, centered && styles.shellCentered]}>{children}</View>;
}

export function StepIndicator({
  currentStep,
  totalSteps = 5,
  label,
}: {
  currentStep: number;
  totalSteps?: number;
  label: string;
}) {
  return (
    <View style={styles.stepWrap}>
      <View style={styles.stepTopRow}>
        <Text style={styles.stepLabel}>{label}</Text>
        <Text style={styles.stepCount}>Step {currentStep} of {totalSteps}</Text>
      </View>
      <View style={styles.segmentRow}>
        {Array.from({ length: totalSteps }).map((_, index) => {
          const isActive = index + 1 <= currentStep;
          return <View key={index} style={[styles.segment, isActive && styles.segmentActive]} />;
        })}
      </View>
    </View>
  );
}

export function PageHeader({
  eyebrow,
  title,
  subtitle,
}: {
  eyebrow?: string;
  title: string;
  subtitle: string;
}) {
  return (
    <View style={styles.pageHeader}>
      {eyebrow ? <Text style={styles.eyebrow}>{eyebrow}</Text> : null}
      <Text style={styles.pageTitle}>{title}</Text>
      <Text style={styles.pageSubtitle}>{subtitle}</Text>
    </View>
  );
}

export function InfoCard({
  title,
  children,
  style,
}: {
  title?: string;
  children: ReactNode;
  style?: StyleProp<ViewStyle>;
}) {
  return (
    <View style={[styles.infoCard, style]}>
      {title ? <Text style={styles.infoTitle}>{title}</Text> : null}
      {children}
    </View>
  );
}

export function StatusChip({ label, tone = 'blue' }: { label: string; tone?: ChipTone }) {
  const toneStyle = toneStyles[tone];
  return (
    <View style={[styles.chip, { backgroundColor: toneStyle.backgroundColor, borderColor: toneStyle.borderColor }]}>
      <Text style={[styles.chipText, { color: toneStyle.color }]}>{label}</Text>
    </View>
  );
}

export function ChipRow({ children }: { children: ReactNode }) {
  return <View style={styles.chipRow}>{children}</View>;
}

export function PrimaryButton({
  label,
  onPress,
  disabled,
  style,
}: {
  label: string;
  onPress: () => void;
  disabled?: boolean;
  style?: StyleProp<ViewStyle>;
}) {
  return (
    <Pressable
      style={({ pressed }) => [styles.primaryButton, disabled && styles.buttonDisabled, pressed && !disabled && styles.buttonPressed, style]}
      onPress={onPress}
      disabled={disabled}>
      <Text style={styles.primaryButtonText}>{label}</Text>
    </Pressable>
  );
}

export function SecondaryButton({
  label,
  onPress,
  disabled,
  style,
}: {
  label: string;
  onPress: () => void;
  disabled?: boolean;
  style?: StyleProp<ViewStyle>;
}) {
  return (
    <Pressable
      style={({ pressed }) => [styles.secondaryButton, disabled && styles.buttonDisabled, pressed && !disabled && styles.buttonPressed, style]}
      onPress={onPress}
      disabled={disabled}>
      <Text style={styles.secondaryButtonText}>{label}</Text>
    </Pressable>
  );
}

export function ScannerFrame({
  imageUri,
  placeholder,
  variant = 'document',
}: {
  imageUri?: string | null;
  placeholder: string;
  variant?: 'document' | 'face';
}) {
  const isFace = variant === 'face';
  return (
    <View style={[styles.scannerFrame, isFace && styles.scannerFrameFace]}>
      <View style={[styles.corner, styles.cornerTopLeft]} />
      <View style={[styles.corner, styles.cornerTopRight]} />
      <View style={[styles.corner, styles.cornerBottomLeft]} />
      <View style={[styles.corner, styles.cornerBottomRight]} />
      {imageUri ? (
        <Image source={{ uri: imageUri } as ImageSourcePropType} style={[styles.scannerImage, isFace && styles.scannerImageFace]} />
      ) : (
        <View style={[styles.scannerPlaceholder, isFace && styles.scannerPlaceholderFace]}>
          <View style={[styles.placeholderMark, isFace && styles.placeholderMarkFace]} />
          <Text style={styles.placeholderText}>{placeholder}</Text>
        </View>
      )}
    </View>
  );
}

export function ConfidenceMeter({
  value,
  label,
  tone = 'blue',
}: {
  value: number | null;
  label: string;
  tone?: ChipTone;
}) {
  const safeValue = typeof value === 'number' && Number.isFinite(value) ? Math.max(0, Math.min(value, 1)) : 0;
  const toneStyle = toneStyles[tone];

  return (
    <View style={styles.meterWrap}>
      <View style={styles.meterHeader}>
        <Text style={styles.meterLabel}>{label}</Text>
        <Text style={[styles.meterValue, { color: toneStyle.color }]}>{value === null ? 'Pending' : `${Math.round(safeValue * 100)}%`}</Text>
      </View>
      <View style={styles.meterTrack}>
        <View style={[styles.meterFill, { width: `${Math.max(safeValue * 100, value === null ? 0 : 6)}%`, backgroundColor: toneStyle.color }]} />
      </View>
    </View>
  );
}

export function RadarVisual({ active = false, passed = false, failed = false }: { active?: boolean; passed?: boolean; failed?: boolean }) {
  const tone = passed ? vaColors.green : failed ? vaColors.red : active ? vaColors.blueSoft : vaColors.border;
  return (
    <View style={styles.radarOuter}>
      <View style={[styles.radarRingLarge, { borderColor: tone }]} />
      <View style={[styles.radarRingMedium, { borderColor: tone }]} />
      <View style={[styles.radarRingSmall, { borderColor: tone, backgroundColor: active ? 'rgba(96, 165, 250, 0.14)' : 'transparent' }]} />
      <View style={[styles.radarBeam, { backgroundColor: tone }]} />
    </View>
  );
}

const styles = StyleSheet.create({
  shell: {
    backgroundColor: vaColors.background,
    flexGrow: 1,
    padding: 24,
  },
  shellCentered: {
    justifyContent: 'center',
  },
  stepWrap: {
    gap: 10,
    marginBottom: 18,
  },
  stepTopRow: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  stepLabel: {
    color: vaColors.blueSoft,
    fontSize: 12,
    fontWeight: '900',
    letterSpacing: 1,
    textTransform: 'uppercase',
  },
  stepCount: {
    color: vaColors.muted,
    fontSize: 12,
    fontWeight: '800',
  },
  segmentRow: {
    flexDirection: 'row',
    gap: 6,
  },
  segment: {
    backgroundColor: '#1E293B',
    borderRadius: 999,
    flex: 1,
    height: 5,
  },
  segmentActive: {
    backgroundColor: vaColors.blue,
  },
  pageHeader: {
    marginBottom: 22,
  },
  eyebrow: {
    color: vaColors.blueSoft,
    fontSize: 13,
    fontWeight: '900',
    letterSpacing: 1,
    marginBottom: 8,
    textTransform: 'uppercase',
  },
  pageTitle: {
    color: vaColors.text,
    fontSize: 32,
    fontWeight: '900',
    lineHeight: 38,
    marginBottom: 10,
  },
  pageSubtitle: {
    color: vaColors.subtle,
    fontSize: 16,
    lineHeight: 24,
  },
  infoCard: {
    backgroundColor: 'rgba(17, 24, 39, 0.96)',
    borderColor: 'rgba(96, 165, 250, 0.18)',
    borderRadius: 18,
    borderWidth: 1,
    padding: 16,
    shadowColor: '#020617',
    shadowOffset: { width: 0, height: 16 },
    shadowOpacity: 0.24,
    shadowRadius: 24,
  },
  infoTitle: {
    color: vaColors.text,
    fontSize: 17,
    fontWeight: '900',
    marginBottom: 12,
  },
  chipRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  chip: {
    alignItems: 'center',
    borderRadius: 999,
    borderWidth: 1,
    paddingHorizontal: 11,
    paddingVertical: 7,
  },
  chipText: {
    fontSize: 12,
    fontWeight: '900',
    letterSpacing: 0.3,
    textTransform: 'uppercase',
  },
  primaryButton: {
    alignItems: 'center',
    backgroundColor: vaColors.blue,
    borderRadius: 14,
    paddingVertical: 16,
    shadowColor: vaColors.blue,
    shadowOffset: { width: 0, height: 12 },
    shadowOpacity: 0.28,
    shadowRadius: 20,
  },
  primaryButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '900',
  },
  secondaryButton: {
    alignItems: 'center',
    backgroundColor: 'rgba(30, 41, 59, 0.92)',
    borderColor: 'rgba(148, 163, 184, 0.18)',
    borderRadius: 14,
    borderWidth: 1,
    paddingVertical: 14,
  },
  secondaryButtonText: {
    color: '#E2E8F0',
    fontSize: 15,
    fontWeight: '800',
  },
  buttonPressed: {
    opacity: 0.82,
    transform: [{ scale: 0.992 }],
  },
  buttonDisabled: {
    opacity: 0.52,
  },
  scannerFrame: {
    alignItems: 'center',
    aspectRatio: 1.58,
    backgroundColor: 'rgba(2, 6, 23, 0.36)',
    borderColor: 'rgba(96, 165, 250, 0.26)',
    borderRadius: 22,
    borderWidth: 1,
    justifyContent: 'center',
    overflow: 'hidden',
    position: 'relative',
    width: '100%',
  },
  scannerFrameFace: {
    alignSelf: 'center',
    aspectRatio: 1,
    borderRadius: 999,
    width: '82%',
  },
  scannerImage: {
    borderRadius: 16,
    height: '88%',
    width: '91%',
  },
  scannerImageFace: {
    borderRadius: 999,
    height: '90%',
    width: '90%',
  },
  scannerPlaceholder: {
    alignItems: 'center',
    height: '88%',
    justifyContent: 'center',
    width: '91%',
  },
  scannerPlaceholderFace: {
    height: '90%',
    width: '90%',
  },
  placeholderMark: {
    borderColor: 'rgba(96, 165, 250, 0.55)',
    borderRadius: 18,
    borderWidth: 2,
    height: 72,
    marginBottom: 12,
    width: 112,
  },
  placeholderMarkFace: {
    borderRadius: 999,
    height: 104,
    width: 104,
  },
  placeholderText: {
    color: vaColors.muted,
    fontSize: 14,
    fontWeight: '700',
  },
  corner: {
    borderColor: vaColors.blueSoft,
    height: 30,
    position: 'absolute',
    width: 30,
    zIndex: 2,
  },
  cornerTopLeft: {
    borderLeftWidth: 3,
    borderTopWidth: 3,
    left: 14,
    top: 14,
  },
  cornerTopRight: {
    borderRightWidth: 3,
    borderTopWidth: 3,
    right: 14,
    top: 14,
  },
  cornerBottomLeft: {
    borderBottomWidth: 3,
    borderLeftWidth: 3,
    bottom: 14,
    left: 14,
  },
  cornerBottomRight: {
    borderBottomWidth: 3,
    borderRightWidth: 3,
    bottom: 14,
    right: 14,
  },
  meterWrap: {
    gap: 9,
  },
  meterHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  meterLabel: {
    color: vaColors.muted,
    fontSize: 12,
    fontWeight: '900',
    letterSpacing: 0.6,
    textTransform: 'uppercase',
  },
  meterValue: {
    fontSize: 13,
    fontWeight: '900',
  },
  meterTrack: {
    backgroundColor: '#1E293B',
    borderRadius: 999,
    height: 9,
    overflow: 'hidden',
  },
  meterFill: {
    borderRadius: 999,
    height: '100%',
  },
  radarOuter: {
    alignItems: 'center',
    alignSelf: 'center',
    height: 190,
    justifyContent: 'center',
    marginBottom: 18,
    position: 'relative',
    width: 190,
  },
  radarRingLarge: {
    borderRadius: 999,
    borderWidth: 1,
    height: 180,
    opacity: 0.3,
    position: 'absolute',
    width: 180,
  },
  radarRingMedium: {
    borderRadius: 999,
    borderWidth: 1,
    height: 128,
    opacity: 0.5,
    position: 'absolute',
    width: 128,
  },
  radarRingSmall: {
    borderRadius: 999,
    borderWidth: 1.5,
    height: 72,
    position: 'absolute',
    width: 72,
  },
  radarBeam: {
    borderRadius: 999,
    height: 3,
    opacity: 0.68,
    position: 'absolute',
    transform: [{ rotate: '-24deg' }],
    width: 82,
  },
});
