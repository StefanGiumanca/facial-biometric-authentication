import { useEffect, type ReactNode } from 'react';
import { Image, Pressable, StyleProp, StyleSheet, Text, View, ViewStyle } from 'react-native';
import Animated, {
  Easing,
  useAnimatedStyle,
  useSharedValue,
  withDelay,
  withRepeat,
  withTiming,
} from 'react-native-reanimated';

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

const toneStyles: Record<ChipTone, { backgroundColor: string; borderColor: string; color: string; glow: string }> = {
  blue: { backgroundColor: 'rgba(37, 99, 235, 0.16)', borderColor: 'rgba(96, 165, 250, 0.32)', color: '#BFDBFE', glow: '#2563EB' },
  green: { backgroundColor: 'rgba(34, 197, 94, 0.14)', borderColor: 'rgba(34, 197, 94, 0.30)', color: '#BBF7D0', glow: '#22C55E' },
  red: { backgroundColor: 'rgba(239, 68, 68, 0.14)', borderColor: 'rgba(239, 68, 68, 0.30)', color: '#FECACA', glow: '#EF4444' },
  amber: { backgroundColor: 'rgba(245, 158, 11, 0.14)', borderColor: 'rgba(245, 158, 11, 0.32)', color: '#FDE68A', glow: '#F59E0B' },
  slate: { backgroundColor: 'rgba(148, 163, 184, 0.10)', borderColor: 'rgba(148, 163, 184, 0.20)', color: '#CBD5E1', glow: '#64748B' },
};

export function AppBackground({ children, solid = false }: { children: ReactNode; solid?: boolean }) {
  const drift = useSharedValue(0);

  useEffect(() => {
    drift.value = withRepeat(withTiming(1, { duration: 7800, easing: Easing.inOut(Easing.quad) }), -1, true);
  }, [drift]);

  const blobOneStyle = useAnimatedStyle(() => ({
    opacity: 0.05 + drift.value * 0.025,
    transform: [{ translateX: drift.value * 8 }, { translateY: drift.value * 6 }],
  }));

  const blobTwoStyle = useAnimatedStyle(() => ({
    opacity: 0.045 + drift.value * 0.02,
    transform: [{ translateX: -drift.value * 6 }, { translateY: -drift.value * 8 }],
  }));

  return (
    <View style={styles.appBackground}>
      {!solid && (
        <>
          <Animated.View pointerEvents="none" style={[styles.glowBlob, styles.glowBlobOne, blobOneStyle]} />
          <Animated.View pointerEvents="none" style={[styles.glowBlob, styles.glowBlobTwo, blobTwoStyle]} />
          <View pointerEvents="none" style={styles.gridLayer}>
            {Array.from({ length: 13 }).map((_, index) => (
              <View key={`h-${index}`} style={[styles.gridLine, { top: `${index * 8}%` }]} />
            ))}
            {Array.from({ length: 7 }).map((_, index) => (
              <View key={`v-${index}`} style={[styles.gridLineVertical, { left: `${index * 16}%` }]} />
            ))}
          </View>
        </>
      )}
      <View style={styles.backgroundContent}>{children}</View>
    </View>
  );
}

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
      <View style={styles.stepRail}>
        <View style={styles.stepConnector} />
        {Array.from({ length: totalSteps }).map((_, index) => {
          const stepNumber = index + 1;
          const isCompleted = stepNumber < currentStep;
          const isActive = stepNumber === currentStep;
          return (
            <View
              key={index}
              style={[
                styles.stepDot,
                isCompleted && styles.stepDotCompleted,
                isActive && styles.stepDotActive,
              ]}>
              <Text style={[styles.stepDotText, (isCompleted || isActive) && styles.stepDotTextActive]}>{stepNumber}</Text>
            </View>
          );
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

export function GlassCard({
  title,
  children,
  style,
}: {
  title?: string;
  children: ReactNode;
  style?: StyleProp<ViewStyle>;
}) {
  return (
    <View style={[styles.glassCard, style]}>
      <View pointerEvents="none" style={styles.cardGlow} />
      {title ? <Text style={styles.infoTitle}>{title}</Text> : null}
      {children}
    </View>
  );
}

export function InfoCard(props: { title?: string; children: ReactNode; style?: StyleProp<ViewStyle> }) {
  return <GlassCard {...props} />;
}

export function StatusChip({ label, tone = 'blue' }: { label: string; tone?: ChipTone }) {
  const toneStyle = toneStyles[tone];
  return (
    <View style={[styles.chip, { backgroundColor: toneStyle.backgroundColor, borderColor: toneStyle.borderColor }]}>
      <View style={[styles.chipDot, { backgroundColor: toneStyle.color }]} />
      <Text style={[styles.chipText, { color: toneStyle.color }]}>{label}</Text>
    </View>
  );
}

export function StatusBadge({ label, tone = 'blue' }: { label: string; tone?: ChipTone }) {
  const toneStyle = toneStyles[tone];
  return (
    <View style={[styles.statusBadge, { borderColor: toneStyle.borderColor, backgroundColor: toneStyle.backgroundColor }]}>
      <Text style={[styles.statusBadgeText, { color: toneStyle.color }]}>{label}</Text>
    </View>
  );
}

export function FeatureChip({ label, tone = 'blue' }: { label: string; tone?: ChipTone }) {
  return <StatusChip label={label} tone={tone} />;
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
      <View pointerEvents="none" style={styles.primaryButtonTopLight} />
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

export function BiometricPulse({ label = 'VA', size = 170 }: { label?: string; size?: number }) {
  const pulse = useSharedValue(0);
  const rotate = useSharedValue(0);

  useEffect(() => {
    pulse.value = withRepeat(withTiming(1, { duration: 2400, easing: Easing.inOut(Easing.quad) }), -1, true);
    rotate.value = withRepeat(withTiming(1, { duration: 9000, easing: Easing.linear }), -1, false);
  }, [pulse, rotate]);

  const ringOne = useAnimatedStyle(() => ({
    opacity: 0.72 - pulse.value * 0.34,
    transform: [{ scale: 0.84 + pulse.value * 0.16 }],
  }));
  const ringTwo = useAnimatedStyle(() => ({
    opacity: 0.35 - pulse.value * 0.14,
    transform: [{ scale: 0.64 + pulse.value * 0.28 }],
  }));
  const beam = useAnimatedStyle(() => ({
    opacity: 0.10,
    transform: [{ rotate: `${rotate.value * 360}deg` }],
  }));

  return (
    <View style={[styles.pulseOuter, { height: size, width: size }]}>
      <Animated.View style={[styles.pulseRing, { height: size, width: size, borderRadius: size / 2 }, ringOne]} />
      <Animated.View style={[styles.pulseRingSoft, { height: size * 0.72, width: size * 0.72, borderRadius: size / 2 }, ringTwo]} />
      <Animated.View style={[styles.pulseBeam, { height: size * 0.78 }, beam]} />
      <View style={[styles.pulseCore, { height: size * 0.48, width: size * 0.48, borderRadius: size / 4 }]}>
        <Text style={styles.pulseLabel}>{label}</Text>
      </View>
    </View>
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
  const scan = useSharedValue(0);

  useEffect(() => {
    scan.value = withRepeat(
      withDelay(350, withTiming(1, { duration: 2300, easing: Easing.inOut(Easing.cubic) })),
      -1,
      false,
    );
  }, [scan]);

  const scanStyle = useAnimatedStyle(() => ({
    transform: [{ translateY: isFace ? -82 + scan.value * 164 : -92 + scan.value * 184 }],
  }));

  return (
    <View style={[styles.scannerFrame, isFace && styles.scannerFrameFace]}>
      <View style={[styles.corner, styles.cornerTopLeft]} />
      <View style={[styles.corner, styles.cornerTopRight]} />
      <View style={[styles.corner, styles.cornerBottomLeft]} />
      <View style={[styles.corner, styles.cornerBottomRight]} />
      <View pointerEvents="none" style={styles.scannerGrid} />
      {imageUri ? (
        <Image source={{ uri: imageUri }} style={[styles.scannerImage, isFace && styles.scannerImageFace]} />
      ) : (
        <View style={[styles.scannerPlaceholder, isFace && styles.scannerPlaceholderFace]}>
          <View style={[styles.placeholderMark, isFace && styles.placeholderMarkFace]} />
          <Text style={styles.placeholderText}>{placeholder}</Text>
        </View>
      )}
      <Animated.View pointerEvents="none" style={[styles.scannerLine, isFace && styles.scannerLineFace, scanStyle]} />
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
  const rotate = useSharedValue(0);
  const pulse = useSharedValue(0);

  useEffect(() => {
    rotate.value = withRepeat(withTiming(1, { duration: 2400, easing: Easing.linear }), -1, false);
    pulse.value = withRepeat(withTiming(1, { duration: 1800, easing: Easing.inOut(Easing.quad) }), -1, true);
  }, [pulse, rotate]);

  const beamStyle = useAnimatedStyle(() => ({
    opacity: active ? 0.72 : 0.42,
    transform: [{ rotate: `${rotate.value * 360}deg` }],
  }));
  const activeRing = useAnimatedStyle(() => ({
    transform: [{ scale: 0.94 + pulse.value * 0.05 }],
  }));

  return (
    <View style={styles.radarOuter}>
      <Animated.View style={[styles.radarRingLarge, { borderColor: tone }, activeRing]} />
      <View style={[styles.radarRingMedium, { borderColor: tone }]} />
      <View style={[styles.radarRingSmall, { borderColor: tone, backgroundColor: active ? 'rgba(96, 165, 250, 0.14)' : 'transparent' }]} />
      <Animated.View style={[styles.radarBeam, { backgroundColor: tone }, beamStyle]} />
      <View style={[styles.radarCore, { backgroundColor: tone }]} />
    </View>
  );
}

const styles = StyleSheet.create({
  appBackground: {
    backgroundColor: vaColors.background,
    flex: 1,
    overflow: 'hidden',
  },
  backgroundContent: {
    flex: 1,
  },
  glowBlob: {
    borderRadius: 999,
    position: 'absolute',
  },
  glowBlobOne: {
    backgroundColor: 'rgba(37, 99, 235, 0.18)',
    height: 190,
    right: -118,
    top: 118,
    width: 190,
  },
  glowBlobTwo: {
    backgroundColor: 'rgba(56, 189, 248, 0.12)',
    bottom: 120,
    height: 210,
    left: -150,
    width: 210,
  },
  gridLayer: {
    bottom: 0,
    left: 0,
    opacity: 0.10,
    position: 'absolute',
    right: 0,
    top: 0,
  },
  gridLine: {
    backgroundColor: 'rgba(96, 165, 250, 0.08)',
    height: 1,
    left: 0,
    position: 'absolute',
    right: 0,
  },
  gridLineVertical: {
    backgroundColor: 'rgba(96, 165, 250, 0.055)',
    bottom: 0,
    position: 'absolute',
    top: 0,
    width: 1,
  },
  shell: {
    flexGrow: 1,
    padding: 24,
  },
  shellCentered: {
    justifyContent: 'center',
  },
  stepWrap: {
    backgroundColor: 'rgba(15, 23, 42, 0.54)',
    borderColor: 'rgba(148, 163, 184, 0.14)',
    borderRadius: 18,
    borderWidth: 1,
    gap: 12,
    marginBottom: 18,
    padding: 14,
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
  stepRail: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
    position: 'relative',
  },
  stepConnector: {
    backgroundColor: 'rgba(148, 163, 184, 0.18)',
    height: 1,
    left: 14,
    position: 'absolute',
    right: 14,
  },
  stepDot: {
    alignItems: 'center',
    backgroundColor: 'rgba(15, 23, 42, 0.94)',
    borderColor: 'rgba(148, 163, 184, 0.24)',
    borderRadius: 999,
    borderWidth: 1,
    height: 24,
    justifyContent: 'center',
    width: 24,
  },
  stepDotCompleted: {
    backgroundColor: 'rgba(37, 99, 235, 0.45)',
    borderColor: 'rgba(96, 165, 250, 0.45)',
    shadowColor: vaColors.blueSoft,
    shadowOpacity: 0.18,
    shadowRadius: 7,
  },
  stepDotActive: {
    backgroundColor: vaColors.blue,
    borderColor: 'rgba(191, 219, 254, 0.52)',
    height: 30,
    shadowColor: vaColors.blue,
    shadowOpacity: 0.22,
    shadowRadius: 10,
    width: 30,
  },
  stepDotText: {
    color: vaColors.muted,
    fontSize: 10,
    fontWeight: '900',
  },
  stepDotTextActive: {
    color: '#FFFFFF',
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
    fontSize: 33,
    fontWeight: '900',
    lineHeight: 39,
    marginBottom: 10,
  },
  pageSubtitle: {
    color: vaColors.subtle,
    fontSize: 16,
    lineHeight: 24,
  },
  glassCard: {
    backgroundColor: 'rgba(15, 23, 42, 0.70)',
    borderColor: 'rgba(148, 163, 184, 0.16)',
    borderRadius: 24,
    borderWidth: 1,
    overflow: 'hidden',
    padding: 16,
    shadowColor: '#020617',
    shadowOffset: { width: 0, height: 18 },
    shadowOpacity: 0.20,
    shadowRadius: 22,
  },
  cardGlow: {
    backgroundColor: 'rgba(96, 165, 250, 0.035)',
    borderRadius: 999,
    height: 130,
    position: 'absolute',
    right: -52,
    top: -70,
    width: 130,
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
    flexDirection: 'row',
    gap: 7,
    paddingHorizontal: 11,
    paddingVertical: 7,
  },
  chipDot: {
    borderRadius: 999,
    height: 6,
    width: 6,
  },
  chipText: {
    fontSize: 11,
    fontWeight: '900',
    letterSpacing: 0.4,
    textTransform: 'uppercase',
  },
  statusBadge: {
    alignSelf: 'flex-start',
    borderRadius: 999,
    borderWidth: 1,
    paddingHorizontal: 16,
    paddingVertical: 9,
  },
  statusBadgeText: {
    fontSize: 13,
    fontWeight: '900',
    letterSpacing: 1,
    textTransform: 'uppercase',
  },
  primaryButton: {
    alignItems: 'center',
    backgroundColor: vaColors.blue,
    borderColor: 'rgba(191, 219, 254, 0.20)',
    borderRadius: 16,
    borderWidth: 1,
    justifyContent: 'center',
    overflow: 'hidden',
    paddingVertical: 16,
    shadowColor: vaColors.blue,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.18,
    shadowRadius: 14,
  },
  primaryButtonTopLight: {
    backgroundColor: 'rgba(255, 255, 255, 0.10)',
    height: 1,
    left: 0,
    position: 'absolute',
    top: 0,
    width: '100%',
  },
  primaryButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '900',
  },
  secondaryButton: {
    alignItems: 'center',
    backgroundColor: 'rgba(15, 23, 42, 0.68)',
    borderColor: 'rgba(148, 163, 184, 0.18)',
    borderRadius: 16,
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
  pulseOuter: {
    alignItems: 'center',
    alignSelf: 'center',
    justifyContent: 'center',
    position: 'relative',
  },
  pulseRing: {
    borderColor: 'rgba(96, 165, 250, 0.22)',
    borderWidth: 1,
    position: 'absolute',
  },
  pulseRingSoft: {
    backgroundColor: 'rgba(37, 99, 235, 0.035)',
    borderColor: 'rgba(56, 189, 248, 0.16)',
    borderWidth: 1,
    position: 'absolute',
  },
  pulseBeam: {
    backgroundColor: 'rgba(96, 165, 250, 0.16)',
    borderRadius: 999,
    position: 'absolute',
    width: 2,
  },
  pulseCore: {
    alignItems: 'center',
    backgroundColor: 'rgba(37, 99, 235, 0.95)',
    borderColor: 'rgba(191, 219, 254, 0.42)',
    borderWidth: 1,
    justifyContent: 'center',
    shadowColor: vaColors.blue,
    shadowOpacity: 0.22,
    shadowRadius: 16,
  },
  pulseLabel: {
    color: '#FFFFFF',
    fontSize: 19,
    fontWeight: '900',
    letterSpacing: 1,
  },
  scannerFrame: {
    alignItems: 'center',
    aspectRatio: 1.58,
    backgroundColor: 'rgba(2, 6, 23, 0.46)',
    borderColor: 'rgba(96, 165, 250, 0.26)',
    borderRadius: 26,
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
    shadowColor: vaColors.blue,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.16,
    shadowRadius: 18,
    width: '82%',
  },
  scannerGrid: {
    borderColor: 'rgba(96, 165, 250, 0.07)',
    borderRadius: 22,
    borderWidth: 1,
    bottom: 22,
    left: 22,
    position: 'absolute',
    right: 22,
    top: 22,
  },
  scannerImage: {
    borderRadius: 18,
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
    borderColor: 'rgba(96, 165, 250, 0.34)',
    borderRadius: 20,
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
  scannerLine: {
    backgroundColor: 'rgba(96, 165, 250, 0.62)',
    height: 2,
    left: '8%',
    position: 'absolute',
    right: '8%',
    shadowColor: vaColors.blueSoft,
    shadowOpacity: 0.34,
    shadowRadius: 9,
  },
  scannerLineFace: {
    left: '18%',
    right: '18%',
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
    backgroundColor: 'rgba(30, 41, 59, 0.96)',
    borderRadius: 999,
    height: 10,
    overflow: 'hidden',
  },
  meterFill: {
    borderRadius: 999,
    height: '100%',
  },
  radarOuter: {
    alignItems: 'center',
    alignSelf: 'center',
    height: 196,
    justifyContent: 'center',
    marginBottom: 18,
    position: 'relative',
    width: 196,
  },
  radarRingLarge: {
    borderRadius: 999,
    borderWidth: 1,
    height: 184,
    opacity: 0.34,
    position: 'absolute',
    width: 184,
  },
  radarRingMedium: {
    borderRadius: 999,
    borderWidth: 1,
    height: 130,
    opacity: 0.52,
    position: 'absolute',
    width: 130,
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
    height: 2,
    opacity: 0.68,
    position: 'absolute',
    width: 88,
  },
  radarCore: {
    borderRadius: 999,
    height: 9,
    position: 'absolute',
    width: 9,
  },
});
