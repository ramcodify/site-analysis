import { Wifi, WifiOff, Clock, Cpu } from 'lucide-react';
import type { ConnectionStatus } from '../../types';

interface HeaderProps {
  title: string;
  subtitle?: string;
  connectionStatus?: ConnectionStatus;
  processingActive?: boolean;
}

export function Header({ title, subtitle, connectionStatus = 'disconnected', processingActive = false }: HeaderProps) {
  const now = new Date();
  const timeStr = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  const dateStr = now.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' });

  return (
    <header className="app-header">
      <div>
        <h2 style={{ fontSize: 17, fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '-0.3px' }}>{title}</h2>
        {subtitle && (
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2, fontWeight: 400 }}>{subtitle}</p>
        )}
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
        {/* Processing Status Pill */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 7,
          fontSize: 12,
          fontWeight: 600,
          padding: '5px 12px',
          borderRadius: 9999,
          background: processingActive ? 'rgba(56, 189, 248, 0.12)' : 'rgba(255, 255, 255, 0.04)',
          border: `1px solid ${processingActive ? 'rgba(56, 189, 248, 0.35)' : 'rgba(255, 255, 255, 0.08)'}`,
          color: processingActive ? '#38bdf8' : 'var(--text-muted)',
          boxShadow: processingActive ? '0 0 14px rgba(56, 189, 248, 0.15)' : 'none',
        }}>
          <Cpu size={14} className={processingActive ? 'spin' : ''} />
          <span>{processingActive ? 'AI Vision Active' : 'AI Standby'}</span>
          <span className={`status-dot ${processingActive ? 'connected' : 'disconnected'}`} />
        </div>

        {/* WebSocket Status Pill */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 7,
          fontSize: 12,
          fontWeight: 600,
          padding: '5px 12px',
          borderRadius: 9999,
          background: connectionStatus === 'connected' ? 'rgba(16, 185, 129, 0.12)' : 'rgba(244, 63, 94, 0.12)',
          border: `1px solid ${connectionStatus === 'connected' ? 'rgba(16, 185, 129, 0.35)' : 'rgba(244, 63, 94, 0.35)'}`,
          color: connectionStatus === 'connected' ? '#34d399' : '#fb7185',
        }}>
          {connectionStatus === 'connected' ? <Wifi size={14} /> : <WifiOff size={14} />}
          <span>{connectionStatus === 'connected' ? 'Live Stream' : 'Disconnected'}</span>
          <span className={`status-dot ${connectionStatus}`} />
        </div>

        {/* Date & Time */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          fontSize: 12,
          fontFamily: 'var(--font-mono)',
          color: 'var(--text-muted)',
          padding: '5px 10px',
          background: 'rgba(255, 255, 255, 0.02)',
          borderRadius: 8,
          border: '1px solid rgba(255, 255, 255, 0.05)',
        }}>
          <Clock size={13} style={{ color: 'var(--text-secondary)' }} />
          <span>{dateStr} {timeStr}</span>
        </div>
      </div>
    </header>
  );
}
