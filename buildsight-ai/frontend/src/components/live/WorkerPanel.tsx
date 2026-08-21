import type { TrackedWorker } from '../../types';
import { Users, ShieldAlert, AlertTriangle, UserCheck } from 'lucide-react';

interface WorkerPanelProps {
  workers: TrackedWorker[];
  activeViolations: number;
  ppeCompliance: number;
}

const RISK_COLORS: Record<string, string> = {
  SAFE: 'var(--safe)',
  LOW: 'var(--low)',
  MEDIUM: 'var(--medium)',
  HIGH: 'var(--high)',
  CRITICAL: 'var(--critical)',
};

export function WorkerPanel({ workers, activeViolations, ppeCompliance }: WorkerPanelProps) {
  const registeredCount = workers.filter(w => w.identity_status === 'REGISTERED').length;

  const sortedWorkers = [...workers].sort((a, b) => {
    const aReg = (a.identity_status === 'REGISTERED' && a.permanent_worker_id) ? 1 : 0;
    const bReg = (b.identity_status === 'REGISTERED' && b.permanent_worker_id) ? 1 : 0;
    if (aReg !== bReg) return bReg - aReg; // Registered on top
    return a.worker_id - b.worker_id;
  });

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title">Live Worker Identities</span>
        <span style={{
          fontSize: 18,
          fontWeight: 700,
          color: 'var(--accent-cyan)',
          fontVariantNumeric: 'tabular-nums',
        }}>
          {workers.length}
        </span>
      </div>

      {/* Quick Stats */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 6, marginBottom: 12 }}>
        <div style={{
          background: 'var(--bg-surface)',
          borderRadius: 'var(--radius-sm)',
          padding: '6px 8px',
          display: 'flex',
          alignItems: 'center',
          gap: 6,
        }}>
          <UserCheck size={12} style={{ color: 'var(--accent-cyan)' }} />
          <div>
            <div style={{ fontSize: 9, color: 'var(--text-muted)' }}>Registered</div>
            <div style={{ fontSize: 12, fontWeight: 600 }}>{registeredCount} / {workers.length}</div>
          </div>
        </div>
        <div style={{
          background: 'var(--bg-surface)',
          borderRadius: 'var(--radius-sm)',
          padding: '6px 8px',
          display: 'flex',
          alignItems: 'center',
          gap: 6,
        }}>
          <ShieldAlert size={12} style={{ color: 'var(--accent-green)' }} />
          <div>
            <div style={{ fontSize: 9, color: 'var(--text-muted)' }}>PPE Rate</div>
            <div style={{ fontSize: 12, fontWeight: 600 }}>{ppeCompliance.toFixed(0)}%</div>
          </div>
        </div>
        <div style={{
          background: 'var(--bg-surface)',
          borderRadius: 'var(--radius-sm)',
          padding: '6px 8px',
          display: 'flex',
          alignItems: 'center',
          gap: 6,
        }}>
          <AlertTriangle size={12} style={{ color: 'var(--accent-amber)' }} />
          <div>
            <div style={{ fontSize: 9, color: 'var(--text-muted)' }}>Violations</div>
            <div style={{ fontSize: 12, fontWeight: 600 }}>{activeViolations}</div>
          </div>
        </div>
      </div>

      {/* Worker List */}
      <div style={{ maxHeight: 340, overflowY: 'auto' }}>
        {workers.length === 0 ? (
          <div style={{ textAlign: 'center', padding: 24, color: 'var(--text-muted)', fontSize: 13 }}>
            <Users size={24} style={{ margin: '0 auto 8px', opacity: 0.4 }} />
            <p>No workers detected</p>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {sortedWorkers.map((w) => {
              const isRegistered = w.identity_status === 'REGISTERED' && w.permanent_worker_id;

              return (
                <div
                  key={w.worker_id}
                  style={{
                    padding: '8px 10px',
                    background: 'var(--bg-surface)',
                    borderRadius: 'var(--radius-sm)',
                    borderLeft: `3px solid ${RISK_COLORS[w.risk_level] || RISK_COLORS.SAFE}`,
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      {isRegistered ? (
                        <>
                          <span style={{ fontSize: 12, fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>
                            {w.permanent_worker_id}
                          </span>
                          <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>
                            {w.name}
                          </span>
                        </>
                      ) : (
                        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)' }}>
                          Unknown Worker
                        </span>
                      )}
                    </div>
                    <span className={`badge ${w.risk_level.toLowerCase()}`} style={{ fontSize: 9 }}>
                      {w.risk_level}
                    </span>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-muted)' }}>
                    <span>ByteTrack #{w.worker_id}</span>
                    <div style={{ display: 'flex', gap: 6, fontSize: 10, flexWrap: 'wrap' }}>
                      {w.helmet !== null && (
                        <span style={{ color: w.helmet ? 'var(--safe)' : 'var(--critical)', fontWeight: 600 }}>
                          🪖 {w.helmet ? 'Helmet ✓' : 'No Helmet ✗'}
                        </span>
                      )}
                      {w.vest !== null && (
                        <span style={{ color: w.vest ? 'var(--safe)' : 'var(--critical)', fontWeight: 600 }}>
                          🦺 {w.vest ? 'Vest ✓' : 'No Vest ✗'}
                        </span>
                      )}
                      {w.gloves !== null && (
                        <span style={{ color: w.gloves ? 'var(--safe)' : 'var(--critical)', fontWeight: 600 }}>
                          🧤 {w.gloves ? '✓' : '✗'}
                        </span>
                      )}
                      {w.face_mask !== null && (
                        <span style={{ color: w.face_mask ? 'var(--safe)' : 'var(--critical)', fontWeight: 600 }}>
                          😷 {w.face_mask ? '✓' : '✗'}
                        </span>
                      )}
                    </div>
                  </div>

                  {w.missing_ppe && w.missing_ppe.length > 0 && (
                    <div style={{ marginTop: 4, display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                      {w.missing_ppe.map((item) => (
                        <span
                          key={item}
                          style={{
                            fontSize: 9,
                            background: 'rgba(239, 68, 68, 0.15)',
                            color: '#f87171',
                            padding: '1px 5px',
                            borderRadius: 3,
                            fontWeight: 500,
                          }}
                        >
                          Missing {item.replace('_', ' ')}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
