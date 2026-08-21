import type { LucideIcon } from 'lucide-react';

interface KPICardProps {
  label: string;
  value: string | number;
  unit?: string;
  sub?: string;
  color?: 'blue' | 'cyan' | 'green' | 'amber' | 'red' | 'purple';
  icon?: LucideIcon;
}

const ACCENT_MAP: Record<string, { color: string; bg: string; border: string }> = {
  blue: { color: '#3b82f6', bg: 'rgba(59, 130, 246, 0.12)', border: 'rgba(59, 130, 246, 0.25)' },
  cyan: { color: '#38bdf8', bg: 'rgba(56, 189, 248, 0.12)', border: 'rgba(56, 189, 248, 0.25)' },
  green: { color: '#10b981', bg: 'rgba(16, 185, 129, 0.12)', border: 'rgba(16, 185, 129, 0.25)' },
  amber: { color: '#f59e0b', bg: 'rgba(245, 158, 11, 0.12)', border: 'rgba(245, 158, 11, 0.25)' },
  red: { color: '#f43f5e', bg: 'rgba(244, 63, 94, 0.12)', border: 'rgba(244, 63, 94, 0.25)' },
  purple: { color: '#a855f7', bg: 'rgba(168, 85, 247, 0.12)', border: 'rgba(168, 85, 247, 0.25)' },
};

export function KPICard({ label, value, unit, sub, color = 'blue', icon: Icon }: KPICardProps) {
  const accent = ACCENT_MAP[color] || ACCENT_MAP.blue;

  return (
    <div className={`kpi-card ${color} fade-in`}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12 }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div className="kpi-label">{label}</div>
          <div className="kpi-value" style={{ wordBreak: 'break-word' }}>
            {value}
            {unit && <span className="kpi-unit">{unit}</span>}
          </div>
          {sub && <div className="kpi-sub">{sub}</div>}
        </div>
        {Icon && (
          <div style={{
            width: 42,
            height: 42,
            minWidth: 42,
            borderRadius: 12,
            background: accent.bg,
            border: `1px solid ${accent.border}`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: `0 0 16px ${accent.bg}`,
          }}>
            <Icon size={20} style={{ color: accent.color }} />
          </div>
        )}
      </div>
    </div>
  );
}
