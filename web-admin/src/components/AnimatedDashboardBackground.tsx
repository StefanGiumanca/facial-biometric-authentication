const PARTICLES = [
  ['8%', '18%', '24s', '0s'],
  ['14%', '72%', '31s', '-9s'],
  ['21%', '36%', '27s', '-14s'],
  ['29%', '84%', '35s', '-6s'],
  ['36%', '14%', '29s', '-18s'],
  ['42%', '62%', '33s', '-11s'],
  ['49%', '29%', '25s', '-4s'],
  ['57%', '78%', '37s', '-22s'],
  ['64%', '21%', '30s', '-17s'],
  ['71%', '48%', '28s', '-8s'],
  ['79%', '12%', '34s', '-19s'],
  ['86%', '69%', '26s', '-10s'],
  ['92%', '34%', '32s', '-3s'],
  ['18%', '51%', '36s', '-25s'],
  ['53%', '9%', '27s', '-13s'],
  ['74%', '89%', '38s', '-28s'],
  ['31%', '6%', '31s', '-20s'],
  ['96%', '83%', '29s', '-15s'],
] as const;

const NETWORK_SEGMENTS = [
  { className: 'dashboard-network__segment dashboard-network__segment--one' },
  { className: 'dashboard-network__segment dashboard-network__segment--two' },
  { className: 'dashboard-network__segment dashboard-network__segment--three' },
] as const;

export function AnimatedDashboardBackground() {
  return (
    <div className="dashboard-background" aria-hidden="true">
      <div className="dashboard-background__grid" />
      <div className="dashboard-background__glow dashboard-background__glow--blue" />
      <div className="dashboard-background__glow dashboard-background__glow--cyan" />
      <div className="dashboard-background__glow dashboard-background__glow--violet" />
      <div className="dashboard-background__scanline" />
      <div className="dashboard-background__pulse dashboard-background__pulse--one" />
      <div className="dashboard-background__pulse dashboard-background__pulse--two" />
      <div className="dashboard-particles">
        {PARTICLES.map(([left, top, duration, delay]) => (
          <span
            key={`${left}-${top}`}
            className="dashboard-particle"
            style={{
              left,
              top,
              animationDuration: duration,
              animationDelay: delay,
            }}
          />
        ))}
      </div>
      <div className="dashboard-network">
        {NETWORK_SEGMENTS.map((segment) => (
          <span key={segment.className} className={segment.className} />
        ))}
      </div>
    </div>
  );
}
