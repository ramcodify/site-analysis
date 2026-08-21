import { useState } from 'react';
import { Header } from '../components/common/Header';
import { Wifi, Square, AlertCircle, CheckCircle, Video } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface RTSPSource {
  name: string;
  rtsp_url: string;
  status: 'idle' | 'connecting' | 'active' | 'error';
  error?: string;
}

const PRESETS = [
  { label: 'Local Test (RTSP)', url: 'rtsp://localhost:8554/test' },
  { label: 'IP Camera (Common)', url: 'rtsp://admin:admin@192.168.1.64:554/h264/ch1/main/av_stream' },
  { label: 'Dahua Camera', url: 'rtsp://admin:admin@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0' },
  { label: 'Hikvision Camera', url: 'rtsp://admin:admin@192.168.1.108:554/Streaming/Channels/101' },
];

export default function RTSPSources() {
  const [name, setName] = useState('Site Camera 1');
  const [rtspUrl, setRtspUrl] = useState('');
  const [activeSources, setActiveSources] = useState<RTSPSource[]>([]);
  const [connecting, setConnecting] = useState(false);

  const handleConnect = async () => {
    if (!rtspUrl.trim() || !name.trim()) return;
    setConnecting(true);

    const newSource: RTSPSource = { name, rtsp_url: rtspUrl, status: 'connecting' };
    setActiveSources(prev => [...prev, newSource]);

    try {
      const res = await fetch(`${API_URL}/api/sources/rtsp`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, rtsp_url: rtspUrl, fps: 5 }),
      });
      if (!res.ok) throw new Error(await res.text());

      setActiveSources(prev =>
        prev.map(s => s.rtsp_url === rtspUrl ? { ...s, status: 'active' } : s)
      );
      setName('Site Camera 1');
      setRtspUrl('');
    } catch (err: any) {
      setActiveSources(prev =>
        prev.map(s => s.rtsp_url === rtspUrl ? { ...s, status: 'error', error: err.message } : s)
      );
    } finally {
      setConnecting(false);
    }
  };

  const handleStop = async () => {
    await fetch(`${API_URL}/api/sources/rtsp/stop`, { method: 'POST' });
    setActiveSources([]);
  };

  return (
    <>
      <Header title="RTSP / CCTV Sources" subtitle="Connect IP cameras and RTSP streams for live AI analysis" />
      <div className="app-content">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, alignItems: 'start' }}>

          {/* Connection Form */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Add RTSP Source</span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div>
                <label style={{ fontSize: 12, color: 'var(--text-muted)', display: 'block', marginBottom: 6 }}>
                  Camera Name
                </label>
                <input
                  type="text"
                  value={name}
                  onChange={e => setName(e.target.value)}
                  placeholder="Site Camera 1"
                  className="input-field"
                />
              </div>
              <div>
                <label style={{ fontSize: 12, color: 'var(--text-muted)', display: 'block', marginBottom: 6 }}>
                  RTSP URL
                </label>
                <input
                  type="text"
                  value={rtspUrl}
                  onChange={e => setRtspUrl(e.target.value)}
                  placeholder="rtsp://..."
                  className="input-field"
                  style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}
                />
              </div>

              {/* Presets */}
              <div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8 }}>Quick Presets</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {PRESETS.map(p => (
                    <button
                      key={p.url}
                      className="btn btn-ghost"
                      onClick={() => setRtspUrl(p.url)}
                      style={{ justifyContent: 'flex-start', fontSize: 12, textAlign: 'left' }}
                    >
                      <Video size={12} />
                      {p.label}
                    </button>
                  ))}
                </div>
              </div>

              <button
                className="btn btn-success"
                onClick={handleConnect}
                disabled={connecting || !rtspUrl.trim() || !name.trim()}
              >
                <Wifi size={16} />
                {connecting ? 'Connecting...' : 'Connect Stream'}
              </button>
            </div>
          </div>

          {/* Active Sources */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Active Sources</span>
              {activeSources.some(s => s.status === 'active') && (
                <button className="btn btn-danger" onClick={handleStop} style={{ padding: '4px 10px', fontSize: 12 }}>
                  <Square size={12} /> Stop All
                </button>
              )}
            </div>
            {activeSources.length === 0 ? (
              <div className="empty-state" style={{ padding: 30 }}>
                <Wifi size={36} style={{ opacity: 0.3 }} />
                <p style={{ marginTop: 8 }}>No active RTSP sources</p>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {activeSources.map((s, i) => (
                  <div key={i} style={{
                    padding: '12px 14px',
                    background: 'var(--bg-surface)',
                    borderRadius: 'var(--radius-sm)',
                    borderLeft: `3px solid ${
                      s.status === 'active' ? 'var(--accent-green)' :
                      s.status === 'error' ? 'var(--accent-red)' :
                      'var(--accent-amber)'
                    }`,
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <div style={{ fontWeight: 600, fontSize: 14 }}>{s.name}</div>
                      {s.status === 'active' && <CheckCircle size={14} style={{ color: 'var(--accent-green)' }} />}
                      {s.status === 'error' && <AlertCircle size={14} style={{ color: 'var(--accent-red)' }} />}
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4, fontFamily: 'var(--font-mono)' }}>
                      {s.rtsp_url}
                    </div>
                    {s.error && (
                      <div style={{ fontSize: 11, color: 'var(--accent-red)', marginTop: 4 }}>
                        Error: {s.error}
                      </div>
                    )}
                    <span className={`badge ${
                      s.status === 'active' ? 'safe' :
                      s.status === 'error' ? 'critical' : 'medium'
                    }`} style={{ marginTop: 6 }}>
                      {s.status.toUpperCase()}
                    </span>
                  </div>
                ))}
              </div>
            )}

            {/* Note */}
            <div style={{
              marginTop: 16,
              padding: '10px 12px',
              background: 'rgba(59,130,246,0.08)',
              borderRadius: 'var(--radius-sm)',
              fontSize: 12,
              color: 'var(--text-muted)',
              borderLeft: '3px solid var(--accent-blue)',
            }}>
              Connected streams are processed by the backend AI pipeline. View results on the Dashboard and Safety Analytics pages.
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
