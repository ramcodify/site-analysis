import { useState, useEffect } from 'react';
import { Header } from '../components/common/Header';


const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function Settings() {
  const [health, setHealth] = useState<any>(null);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const res = await fetch(`${API_URL}/health`);
        if (res.ok) setHealth(await res.json());
      } catch { /* */ }
    };
    fetchHealth();
  }, []);

  return (
    <>
      <Header title="Settings" subtitle="System configuration and diagnostics" />
      <div className="app-content">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          {/* System Status */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">System Status</span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 13 }}>
                <span style={{ color: 'var(--text-muted)' }}>Backend</span>
                <span className={`badge ${health ? 'safe' : 'critical'}`}>
                  {health ? 'CONNECTED' : 'DISCONNECTED'}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 13 }}>
                <span style={{ color: 'var(--text-muted)' }}>Status</span>
                <span>{health?.status ?? '—'}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 13 }}>
                <span style={{ color: 'var(--text-muted)' }}>Processing</span>
                <span>{health?.processing ? 'Active' : 'Idle'}</span>
              </div>
            </div>
          </div>

          {/* Model Status */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">AI Models</span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {health?.models ? (
                Object.entries(health.models).map(([name, status]: [string, any]) => (
                  <div key={name} style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '8px 12px',
                    background: 'var(--bg-surface)',
                    borderRadius: 'var(--radius-sm)',
                  }}>
                    <div>
                      <div style={{ fontSize: 13, fontWeight: 600 }}>
                        {name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                      </div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
                        {status.loaded ? status.model : status.error}
                      </div>
                    </div>
                    <span className={`badge ${status.loaded ? 'safe' : 'medium'}`}>
                      {status.loaded ? 'LOADED' : 'N/A'}
                    </span>
                  </div>
                ))
              ) : (
                <div style={{ textAlign: 'center', padding: 20, color: 'var(--text-muted)', fontSize: 13 }}>
                  Connect to backend to view model status
                </div>
              )}
            </div>
          </div>

          {/* Configuration Info */}
          <div className="card" style={{ gridColumn: '1 / -1' }}>
            <div className="card-header">
              <span className="card-title">Configuration</span>
            </div>
            <div style={{
              fontSize: 13,
              color: 'var(--text-secondary)',
              lineHeight: 1.8,
              fontFamily: 'var(--font-mono)',
              background: 'var(--bg-surface)',
              padding: 16,
              borderRadius: 'var(--radius-sm)',
            }}>
              <div>Backend URL: {API_URL}</div>
              <div>WebSocket URL: {import.meta.env.VITE_WS_URL || 'ws://localhost:8000'}</div>
              <div>Default Capture FPS: 10</div>
              <div>Frame Quality: JPEG 70%</div>
              <div>Max Processing Queue: 2</div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
