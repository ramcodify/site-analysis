import { useState, useEffect, useCallback } from 'react';
import { Header } from '../components/common/Header';
import type { ViolationResponse } from '../types';
import {
  AlertTriangle, CheckCircle, Eye, Filter, Trash2,
  LayoutGrid, LayoutList, Clock, Shield, User, Zap
} from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const SEVERITY_ORDER = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'];
const STATUS_FILTERS = ['All', 'OPEN', 'ACKNOWLEDGED', 'RESOLVED'];
const SEVERITY_FILTERS = ['All', 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'];

const SEVERITY_COLORS: Record<string, string> = {
  CRITICAL: '#ef4444',
  HIGH: '#f97316',
  MEDIUM: '#f59e0b',
  LOW: '#3b82f6',
};

const SEVERITY_BG: Record<string, string> = {
  CRITICAL: 'rgba(239,68,68,0.10)',
  HIGH: 'rgba(249,115,22,0.10)',
  MEDIUM: 'rgba(245,158,11,0.10)',
  LOW: 'rgba(59,130,246,0.10)',
};

export default function Violations() {
  const [violations, setViolations] = useState<ViolationResponse[]>([]);
  const [statusFilter, setStatusFilter] = useState('All');
  const [severityFilter, setSeverityFilter] = useState('All');
  const [selected, setSelected] = useState<ViolationResponse | null>(null);
  const [updating, setUpdating] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'table' | 'cards'>('cards');

  const fetchViolations = useCallback(async () => {
    try {
      let url = `${API_URL}/api/violations`;
      const params = [];
      if (statusFilter !== 'All') params.push(`status=${statusFilter}`);
      if (severityFilter !== 'All') params.push(`severity=${severityFilter}`);
      if (params.length) url += '?' + params.join('&');
      const res = await fetch(url);
      if (res.ok) setViolations(await res.json());
    } catch { /* */ }
  }, [statusFilter, severityFilter]);

  useEffect(() => {
    fetchViolations();
    const iv = setInterval(fetchViolations, 4000);
    return () => clearInterval(iv);
  }, [fetchViolations]);

  const updateStatus = async (violationId: string, status: string) => {
    setUpdating(violationId);
    try {
      await fetch(`${API_URL}/api/violations/${violationId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status }),
      });
      await fetchViolations();
      if (selected?.violation_id === violationId) {
        setSelected(prev => prev ? { ...prev, status: status as any } : null);
      }
    } catch { /* */ } finally {
      setUpdating(null);
    }
  };

  const deleteViolation = async (violationId: string) => {
    if (!window.confirm('Are you sure you want to permanently delete this violation and its stored evidence photo from the database?')) return;
    setUpdating(violationId);
    try {
      const res = await fetch(`${API_URL}/api/violations/${violationId}`, { method: 'DELETE' });
      if (res.ok) {
        if (selected?.violation_id === violationId) setSelected(null);
        await fetchViolations();
      }
    } catch (err) {
      console.error('Failed to delete violation:', err);
    } finally {
      setUpdating(null);
    }
  };

  const clearAllViolations = async () => {
    if (!window.confirm('Are you sure you want to delete ALL violations and evidence photos from MongoDB? This action cannot be undone.')) return;
    try {
      const res = await fetch(`${API_URL}/api/violations`, { method: 'DELETE' });
      if (res.ok) { setSelected(null); await fetchViolations(); }
    } catch (err) {
      console.error('Failed to clear violations:', err);
    }
  };

  const sorted = [...violations].sort((a, b) => {
    const sa = SEVERITY_ORDER.indexOf(a.severity);
    const sb = SEVERITY_ORDER.indexOf(b.severity);
    return sa - sb;
  });

  const openCount = violations.filter(v => v.status === 'OPEN').length;
  const critCount = violations.filter(v => v.severity === 'CRITICAL').length;

  return (
    <>
      <Header title="Violations" subtitle="Safety violations management and resolution workflow" />
      <div className="app-content">
        {/* KPI Summary */}
        <div className="kpi-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)', marginBottom: 16 }}>
          <div className="kpi-card amber"><div className="kpi-label">Total Violations</div><div className="kpi-value">{violations.length}</div></div>
          <div className="kpi-card red"><div className="kpi-label">Open</div><div className="kpi-value">{openCount}</div></div>
          <div className="kpi-card red"><div className="kpi-label">Critical</div><div className="kpi-value">{critCount}</div></div>
          <div className="kpi-card green"><div className="kpi-label">Resolved</div><div className="kpi-value">{violations.filter(v => v.status === 'RESOLVED').length}</div></div>
        </div>

        {/* Filters + View Toggle */}
        <div style={{ display: 'flex', gap: 12, marginBottom: 16, flexWrap: 'wrap', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <Filter size={14} style={{ color: 'var(--text-muted)' }} />
            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Status:</span>
            <div className="filter-bar" style={{ margin: 0 }}>
              {STATUS_FILTERS.map(f => (
                <button key={f} className={`filter-chip ${statusFilter === f ? 'active' : ''}`} onClick={() => setStatusFilter(f)}>{f}</button>
              ))}
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Severity:</span>
            <div className="filter-bar" style={{ margin: 0 }}>
              {SEVERITY_FILTERS.map(f => (
                <button key={f} className={`filter-chip ${severityFilter === f ? 'active' : ''}`} onClick={() => setSeverityFilter(f)}>{f}</button>
              ))}
            </div>
          </div>
          <div style={{ marginLeft: 'auto', display: 'flex', gap: 8, alignItems: 'center' }}>
            {/* View Mode Toggle */}
            <div style={{ display: 'flex', background: 'var(--bg-surface)', borderRadius: 8, padding: 3, border: '1px solid var(--border-primary)', gap: 2 }}>
              <button
                className={`btn ${viewMode === 'cards' ? 'btn-primary' : 'btn-ghost'}`}
                style={{ padding: '4px 10px', fontSize: 12 }}
                onClick={() => setViewMode('cards')}
                title="Card Grid View"
              >
                <LayoutGrid size={13} />
              </button>
              <button
                className={`btn ${viewMode === 'table' ? 'btn-primary' : 'btn-ghost'}`}
                style={{ padding: '4px 10px', fontSize: 12 }}
                onClick={() => setViewMode('table')}
                title="Table View"
              >
                <LayoutList size={13} />
              </button>
            </div>
            <button
              className="btn btn-ghost"
              style={{ color: 'var(--accent-red)', fontSize: 12, padding: '4px 10px' }}
              onClick={clearAllViolations}
              disabled={violations.length === 0}
              title="Clear all violations and stored evidence photos from database"
            >
              <Trash2 size={13} /> Clear All
            </button>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: selected ? '1fr 380px' : '1fr', gap: 16 }}>

          {/* ─── CARD VIEW ─────────────────────────────────────────── */}
          {viewMode === 'cards' ? (
            <div>
              {sorted.length > 0 ? (
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
                  gap: 14,
                }}>
                  {sorted.map((v) => {
                    const sevColor = SEVERITY_COLORS[v.severity] || '#6b7280';
                    const sevBg = SEVERITY_BG[v.severity] || 'rgba(107,114,128,0.10)';
                    const isWorking = updating === v.violation_id;
                    return (
                      <div
                        key={v.violation_id}
                        onClick={() => setSelected(v)}
                        style={{
                          background: 'var(--bg-card)',
                          border: `1.5px solid ${selected?.violation_id === v.violation_id ? sevColor : 'var(--border-primary)'}`,
                          borderRadius: 12,
                          overflow: 'hidden',
                          cursor: 'pointer',
                          transition: 'transform 0.15s, box-shadow 0.15s',
                          boxShadow: selected?.violation_id === v.violation_id ? `0 0 0 2px ${sevColor}33` : '0 2px 8px rgba(0,0,0,0.18)',
                        }}
                        onMouseEnter={e => {
                          (e.currentTarget as HTMLDivElement).style.transform = 'translateY(-2px)';
                          (e.currentTarget as HTMLDivElement).style.boxShadow = `0 8px 24px rgba(0,0,0,0.28)`;
                        }}
                        onMouseLeave={e => {
                          (e.currentTarget as HTMLDivElement).style.transform = 'translateY(0)';
                          (e.currentTarget as HTMLDivElement).style.boxShadow = selected?.violation_id === v.violation_id ? `0 0 0 2px ${sevColor}33` : '0 2px 8px rgba(0,0,0,0.18)';
                        }}
                      >
                        {/* Card Top: Evidence Photo or Severity Banner */}
                        <div style={{ position: 'relative', height: 130, background: '#000', overflow: 'hidden' }}>
                          {(v.snapshot_base64 || v.evidence_path) ? (
                            <img
                              src={v.snapshot_base64 || (v.evidence_path?.startsWith('http') ? v.evidence_path : `${API_URL}${v.evidence_path}`)}
                              alt="Evidence"
                              style={{ width: '100%', height: '100%', objectFit: 'cover', opacity: 0.85 }}
                            />
                          ) : (
                            <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: sevBg }}>
                              <AlertTriangle size={36} style={{ color: sevColor, opacity: 0.6 }} />
                            </div>
                          )}
                          {/* Severity badge overlay */}
                          <div style={{
                            position: 'absolute', top: 8, left: 8,
                            background: sevColor, color: '#fff',
                            borderRadius: 5, padding: '2px 8px',
                            fontSize: 11, fontWeight: 700, letterSpacing: 0.5,
                          }}>
                            {v.severity}
                          </div>
                          {/* Status badge overlay */}
                          <div style={{
                            position: 'absolute', top: 8, right: 8,
                            background: v.status === 'OPEN' ? 'rgba(239,68,68,0.85)' : v.status === 'ACKNOWLEDGED' ? 'rgba(245,158,11,0.85)' : 'rgba(16,185,129,0.85)',
                            color: '#fff',
                            borderRadius: 5, padding: '2px 8px',
                            fontSize: 10, fontWeight: 600,
                          }}>
                            {v.status}
                          </div>
                        </div>

                        {/* Card Body */}
                        <div style={{ padding: '12px 14px' }}>
                          {/* Violation Type */}
                          <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 4, lineHeight: 1.3 }}>
                            {v.violation_type?.replace(/_/g, ' ') || 'Unknown Violation'}
                          </div>

                          {/* Worker info */}
                          <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 8 }}>
                            <User size={11} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
                            {(v.worker_name || v.worker_code) ? (
                              <>
                                <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>
                                  {v.worker_code || v.permanent_worker_id}
                                </span>
                                <span style={{ fontSize: 11, color: 'var(--text-primary)' }}>
                                  — {v.worker_name}
                                </span>
                              </>
                            ) : (
                              <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                                Unregistered · Track #{v.worker_id ?? '—'}
                              </span>
                            )}
                          </div>

                          {/* Missing items */}
                          {v.missing_items && v.missing_items.length > 0 && (
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 8 }}>
                              {v.missing_items.map(item => (
                                <span key={item} style={{
                                  fontSize: 10, background: 'rgba(239,68,68,0.12)', color: 'var(--accent-red)',
                                  borderRadius: 4, padding: '1px 6px', border: '1px solid rgba(239,68,68,0.2)',
                                }}>
                                  ✕ {item}
                                </span>
                              ))}
                            </div>
                          )}

                          {/* Meta row */}
                          <div style={{ display: 'flex', gap: 10, fontSize: 11, color: 'var(--text-muted)', marginBottom: 10 }}>
                            <span style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                              <Zap size={10} /> {v.risk_score?.toFixed(0) ?? '—'}
                            </span>
                            <span style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                              <Clock size={10} /> {v.duration_seconds?.toFixed(0) ?? 0}s
                            </span>
                            <span style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                              <Shield size={10} /> {v.timestamp ? new Date(v.timestamp).toLocaleTimeString() : '—'}
                            </span>
                          </div>

                          {/* Action Buttons */}
                          <div style={{ display: 'flex', gap: 6 }} onClick={e => e.stopPropagation()}>
                            {v.status === 'OPEN' && (
                              <button
                                className="btn btn-ghost"
                                style={{ padding: '3px 9px', fontSize: 11, flex: 1 }}
                                disabled={isWorking}
                                onClick={() => updateStatus(v.violation_id, 'ACKNOWLEDGED')}
                              >
                                <Eye size={10} /> Ack
                              </button>
                            )}
                            {v.status !== 'RESOLVED' && (
                              <button
                                className="btn btn-success"
                                style={{ padding: '3px 9px', fontSize: 11, flex: 1 }}
                                disabled={isWorking}
                                onClick={() => updateStatus(v.violation_id, 'RESOLVED')}
                              >
                                <CheckCircle size={10} /> Resolve
                              </button>
                            )}
                            <button
                              className="btn btn-danger"
                              style={{ padding: '3px 9px', fontSize: 11 }}
                              disabled={isWorking}
                              title="Delete violation and stored evidence photo"
                              onClick={() => deleteViolation(v.violation_id)}
                            >
                              <Trash2 size={10} />
                            </button>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="card">
                  <div className="empty-state">
                    <AlertTriangle size={36} style={{ opacity: 0.3 }} />
                    <h3>No Violations</h3>
                    <p>Violations appear here when PPE issues or safety risks are detected</p>
                  </div>
                </div>
              )}
            </div>
          ) : (
            /* ─── TABLE VIEW ─────────────────────────────────────────── */
            <div className="card" style={{ overflow: 'auto' }}>
              {sorted.length > 0 ? (
                <table className="data-table">
                  <thead><tr>
                    <th>Worker</th><th>Type</th><th>Severity</th>
                    <th>Risk Score</th><th>Status</th><th>Duration</th><th>Time</th><th>Actions</th>
                  </tr></thead>
                  <tbody>
                    {sorted.map((v, i) => (
                      <tr key={i} onClick={() => setSelected(v)} style={{ cursor: 'pointer' }}>
                        <td style={{ fontWeight: 600 }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            {(v.snapshot_base64 || v.evidence_path) ? (
                              <img
                                src={v.snapshot_base64 || (v.evidence_path?.startsWith('http') ? v.evidence_path : `${API_URL}${v.evidence_path}`)}
                                alt="Worker"
                                style={{ width: 34, height: 34, borderRadius: 6, objectFit: 'cover', border: '1px solid var(--border-primary)', flexShrink: 0 }}
                              />
                            ) : (
                              <div style={{ width: 34, height: 34, borderRadius: 6, background: 'var(--bg-surface)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 12, color: 'var(--text-muted)', flexShrink: 0 }}>
                                📸
                              </div>
                            )}
                            <div>
                              {v.worker_name || v.worker_code ? (
                                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                  <span className="badge cyan" style={{ padding: '2px 6px', fontSize: 11 }}>
                                    ✓ {v.worker_code || v.permanent_worker_id}
                                  </span>
                                  <span style={{ fontSize: 13, color: 'var(--text-primary)' }}>
                                    {v.worker_name || 'Registered'}
                                  </span>
                                </div>
                              ) : (
                                <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                                  Worker #{v.worker_id ?? '—'} (Unknown Tracker)
                                </span>
                              )}
                            </div>
                          </div>
                        </td>
                        <td style={{ maxWidth: 160 }}>
                          <div>{v.violation_type?.replace(/_/g, ' ')}</div>
                          {v.missing_items && v.missing_items.length > 0 && (
                            <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                              Missing: {v.missing_items.join(', ')}
                            </div>
                          )}
                        </td>
                        <td><span className={`badge ${v.severity?.toLowerCase()}`}>{v.severity}</span></td>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>{v.risk_score?.toFixed(0)}</td>
                        <td>
                          <span className={`badge ${v.status === 'OPEN' ? 'high' : v.status === 'ACKNOWLEDGED' ? 'medium' : 'safe'}`}>
                            {v.status}
                          </span>
                        </td>
                        <td style={{ fontSize: 11, color: 'var(--text-muted)' }}>{v.duration_seconds?.toFixed(0) ?? 0}s</td>
                        <td style={{ fontSize: 11, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                          {v.timestamp ? new Date(v.timestamp).toLocaleTimeString() : '—'}
                        </td>
                        <td onClick={e => e.stopPropagation()}>
                          <div style={{ display: 'flex', gap: 4 }}>
                            {v.status === 'OPEN' && (
                              <button
                                className="btn btn-ghost"
                                style={{ padding: '3px 8px', fontSize: 11 }}
                                disabled={updating === v.violation_id}
                                onClick={() => updateStatus(v.violation_id, 'ACKNOWLEDGED')}
                              >
                                <Eye size={10} /> Ack
                              </button>
                            )}
                            {v.status !== 'RESOLVED' && (
                              <button
                                className="btn btn-success"
                                style={{ padding: '3px 8px', fontSize: 11 }}
                                disabled={updating === v.violation_id}
                                onClick={() => updateStatus(v.violation_id, 'RESOLVED')}
                              >
                                <CheckCircle size={10} /> Resolve
                              </button>
                            )}
                            <button
                              className="btn btn-danger"
                              style={{ padding: '3px 8px', fontSize: 11 }}
                              disabled={updating === v.violation_id}
                              title="Delete violation and stored evidence photo"
                              onClick={() => deleteViolation(v.violation_id)}
                            >
                              <Trash2 size={10} /> Delete
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div className="empty-state">
                  <AlertTriangle size={36} style={{ opacity: 0.3 }} />
                  <h3>No Violations</h3>
                  <p>Violations appear here when PPE issues or safety risks are detected</p>
                </div>
              )}
            </div>
          )}

          {/* ─── DETAIL PANEL ─────────────────────────────────────── */}
          {selected && (
            <div className="card slide-in">
              <div className="card-header">
                <span className="card-title">Violation Detail</span>
                <button className="btn btn-ghost" onClick={() => setSelected(null)} style={{ padding: '4px 8px', fontSize: 12 }}>✕</button>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>

                {/* Evidence Photo */}
                {(selected.snapshot_base64 || selected.evidence_path) && (
                  <div style={{ padding: 10, background: 'var(--bg-surface)', borderRadius: 8, border: '1px solid var(--border-primary)' }}>
                    <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 4 }}>
                      📸 Violation Tracker Photo Snapshot
                    </div>
                    <div style={{ borderRadius: 6, overflow: 'hidden', border: '1px solid var(--border-primary)', background: '#000', textAlign: 'center' }}>
                      <img
                        src={selected.snapshot_base64 || (selected.evidence_path?.startsWith('http') ? selected.evidence_path : `${API_URL}${selected.evidence_path}`)}
                        alt="Violation Evidence"
                        style={{ width: '100%', maxHeight: 200, objectFit: 'contain', display: 'block' }}
                      />
                    </div>
                  </div>
                )}

                {[
                  ['Violation ID', selected.violation_id],
                  ['Worker Code', selected.worker_code || selected.permanent_worker_id || '—'],
                  ['Worker Name', selected.worker_name || 'Unregistered Worker (Unknown Tracker)'],
                  ['Employee ID', selected.employee_number || '—'],
                  ['Track ID', `#${selected.worker_id ?? '—'}`],
                  ['Violation Type', selected.violation_type?.replace(/_/g, ' ')],
                  ['Missing Items', selected.missing_items?.join(', ') || '—'],
                  ['Description', selected.description ?? '—'],
                  ['Severity', null],
                  ['Risk Score', selected.risk_score?.toFixed(1)],
                  ['Status', null],
                  ['Duration', `${selected.duration_seconds?.toFixed(0) ?? 0}s`],
                  ['Detected At', selected.timestamp ? new Date(selected.timestamp).toLocaleString() : '—'],
                  ['Resolved At', selected.resolved_at ? new Date(selected.resolved_at).toLocaleString() : (selected.status === 'RESOLVED' ? 'Resolved' : 'Still Active')],
                ].map(([label, val]) => (
                  <div key={label as string} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                    <span style={{ color: 'var(--text-muted)' }}>{label}</span>
                    {label === 'Severity' ? (
                      <span className={`badge ${selected.severity?.toLowerCase()}`}>{selected.severity}</span>
                    ) : label === 'Status' ? (
                      <span className={`badge ${selected.status === 'OPEN' ? 'high' : selected.status === 'ACKNOWLEDGED' ? 'medium' : 'safe'}`}>{selected.status}</span>
                    ) : (
                      <span style={{ fontWeight: 500, maxWidth: 200, textAlign: 'right', wordBreak: 'break-all' }}>{val as string}</span>
                    )}
                  </div>
                ))}

                <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 8, paddingTop: 12, borderTop: '1px solid var(--border-primary)' }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', marginBottom: 4 }}>ACTIONS</div>
                  {selected.status === 'OPEN' && (
                    <button className="btn btn-ghost" style={{ justifyContent: 'center' }} onClick={() => updateStatus(selected.violation_id, 'ACKNOWLEDGED')}>
                      <Eye size={14} /> Acknowledge
                    </button>
                  )}
                  {selected.status !== 'RESOLVED' && (
                    <button className="btn btn-success" style={{ justifyContent: 'center' }} onClick={() => updateStatus(selected.violation_id, 'RESOLVED')}>
                      <CheckCircle size={14} /> Mark Resolved
                    </button>
                  )}
                  <button
                    className="btn btn-danger"
                    style={{ justifyContent: 'center' }}
                    onClick={() => deleteViolation(selected.violation_id)}
                  >
                    <Trash2 size={14} /> Delete Violation & Stored Photo
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
