import { useState, useEffect, useMemo } from 'react';
import { Header } from '../components/common/Header';
import type { WorkerResponse, RegisteredWorker } from '../types';
import {
  Users, Search, Link2, CheckCircle, RotateCcw,
  ShieldCheck, AlertTriangle, AlertOctagon, ShieldAlert,
  LayoutGrid, List, Shield, Trash2
} from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const RISK_FILTERS = ['ALL', 'SAFE', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'] as const;

export default function Workers() {
  const [workers, setWorkers] = useState<WorkerResponse[]>([]);
  const [registeredList, setRegisteredList] = useState<RegisteredWorker[]>([]);
  const [filter, setFilter] = useState<string>('ALL');
  const [viewMode, setViewMode] = useState<'table' | 'grid'>('table');
  const [selectedWorker, setSelectedWorker] = useState<any>(null);
  const [search, setSearch] = useState('');
  const [linkingWorkerId, setLinkingWorkerId] = useState<number | null>(null);
  const [selectedRegCode, setSelectedRegCode] = useState<string>('');
  const [resetting, setResetting] = useState(false);

  const fetchWorkers = async () => {
    try {
      const res = await fetch(`${API_URL}/api/workers`);
      if (res.ok) {
        const data: WorkerResponse[] = await res.json();
        // Client-side unique deduplication safeguard
        const seenKeys = new Set<string>();
        const uniqueList: WorkerResponse[] = [];
        for (const w of data) {
          const code = w.permanent_worker_id || w.worker_code;
          const key = code ? `code_${code}` : `track_${w.track_id || w.worker_id}`;
          if (!seenKeys.has(key)) {
            seenKeys.add(key);
            uniqueList.push(w);
          }
        }
        setWorkers(uniqueList);
      }
    } catch { /* */ }
  };

  const handleResetTracks = async () => {
    setResetting(true);
    try {
      await fetch(`${API_URL}/api/workers`, { method: 'DELETE' });
      setWorkers([]);
      setSelectedWorker(null);
      await fetchWorkers();
    } catch {
      //
    } finally {
      setResetting(false);
    }
  };

  const handleDeleteWorkerTrack = async (trackId: number, e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    if (!window.confirm(`Are you sure you want to delete track #${trackId} from active live tracking?`)) {
      return;
    }
    try {
      await fetch(`${API_URL}/api/workers/${trackId}`, { method: 'DELETE' });
      setWorkers(prev => prev.filter(w => (w.track_id || w.worker_id) !== trackId));
      if (selectedWorker && (selectedWorker.track_id === trackId || selectedWorker.worker_id === trackId)) {
        setSelectedWorker(null);
      }
      await fetchWorkers();
    } catch {
      //
    }
  };

  const fetchRegistered = async () => {
    try {
      const res = await fetch(`${API_URL}/api/registered-workers?active_only=true`);
      if (res.ok) setRegisteredList(await res.json());
    } catch { /* */ }
  };

  useEffect(() => {
    fetchWorkers();
    fetchRegistered();
    const interval = setInterval(fetchWorkers, 3000);
    return () => clearInterval(interval);
  }, []);

  const counts = useMemo(() => {
    const c = { SAFE: 0, LOW: 0, MEDIUM: 0, HIGH: 0, CRITICAL: 0, ALL: workers.length };
    workers.forEach(w => {
      const lvl = (w.risk_level || 'SAFE').toUpperCase() as keyof typeof c;
      if (c[lvl] !== undefined) {
        c[lvl]++;
      }
    });
    return c;
  }, [workers]);

  const filteredWorkers = workers.filter(w => {
    if (filter !== 'ALL' && (w.risk_level || 'SAFE').toUpperCase() !== filter) return false;
    if (search) {
      const term = search.toLowerCase();
      const matchId = `#${w.track_id}`.includes(term) || `#${w.worker_id}`.includes(term);
      const matchCode = (w.permanent_worker_id || w.worker_code || '').toLowerCase().includes(term);
      const matchName = (w.name || '').toLowerCase().includes(term);
      if (!matchId && !matchCode && !matchName) return false;
    }
    return true;
  });

  const fetchWorkerDetail = async (workerId: number) => {
    try {
      const res = await fetch(`${API_URL}/api/workers/${workerId}`);
      if (res.ok) setSelectedWorker(await res.json());
    } catch { /* */ }
  };

  const handleLinkIdentity = async (trackId: number) => {
    if (!selectedRegCode) return;
    try {
      const res = await fetch(`${API_URL}/api/unknown-persons/link`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          track_id: trackId,
          worker_code: selectedRegCode,
        }),
      });
      if (res.ok) {
        setLinkingWorkerId(null);
        setSelectedRegCode('');
        await fetchWorkers();
        if (selectedWorker?.track_id === trackId) {
          fetchWorkerDetail(trackId);
        }
      }
    } catch { /* */ }
  };

  const getRiskColor = (level: string) => {
    switch (level?.toUpperCase()) {
      case 'SAFE': return '#10b981';
      case 'LOW': return '#38bdf8';
      case 'MEDIUM': return '#f59e0b';
      case 'HIGH': return '#f97316';
      case 'CRITICAL': return '#ef4444';
      default: return '#10b981';
    }
  };

  return (
    <>
      <Header title="Live Worker Tracking" subtitle="Real-time personnel telemetry, facial biometric identification & risk scoring" />
      <div className="app-content" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

        {/* Top Risk Level Breakdown KPI Grid */}
        <div className="kpi-grid" style={{ gridTemplateColumns: 'repeat(5, 1fr)' }}>
          <div
            className="kpi-card green"
            onClick={() => setFilter('SAFE')}
            style={{ cursor: 'pointer', border: filter === 'SAFE' ? '1px solid #10b981' : undefined }}
          >
            <div className="kpi-label" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <ShieldCheck size={14} style={{ color: '#10b981' }} /> SAFE (100% PPE)
            </div>
            <div className="kpi-value" style={{ color: '#10b981' }}>{counts.SAFE}</div>
            <div className="kpi-sub">Zero infractions</div>
          </div>

          <div
            className="kpi-card cyan"
            onClick={() => setFilter('LOW')}
            style={{ cursor: 'pointer', border: filter === 'LOW' ? '1px solid #38bdf8' : undefined }}
          >
            <div className="kpi-label" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <Shield size={14} style={{ color: '#38bdf8' }} /> LOW RISK
            </div>
            <div className="kpi-value" style={{ color: '#38bdf8' }}>{counts.LOW}</div>
            <div className="kpi-sub">Minor telemetry drift</div>
          </div>

          <div
            className="kpi-card amber"
            onClick={() => setFilter('MEDIUM')}
            style={{ cursor: 'pointer', border: filter === 'MEDIUM' ? '1px solid #f59e0b' : undefined }}
          >
            <div className="kpi-label" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <AlertTriangle size={14} style={{ color: '#f59e0b' }} /> MEDIUM RISK
            </div>
            <div className="kpi-value" style={{ color: '#f59e0b' }}>{counts.MEDIUM}</div>
            <div className="kpi-sub">Single PPE gap or alert</div>
          </div>

          <div
            className="kpi-card red"
            onClick={() => setFilter('HIGH')}
            style={{ cursor: 'pointer', border: filter === 'HIGH' ? '1px solid #f97316' : undefined }}
          >
            <div className="kpi-label" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <AlertOctagon size={14} style={{ color: '#f97316' }} /> HIGH RISK
            </div>
            <div className="kpi-value" style={{ color: '#f97316' }}>{counts.HIGH}</div>
            <div className="kpi-sub">Multiple missing PPE</div>
          </div>

          <div
            className="kpi-card red"
            onClick={() => setFilter('CRITICAL')}
            style={{ cursor: 'pointer', border: filter === 'CRITICAL' ? '1px solid #ef4444' : undefined }}
          >
            <div className="kpi-label" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <ShieldAlert size={14} style={{ color: '#ef4444' }} /> CRITICAL
            </div>
            <div className="kpi-value" style={{ color: '#ef4444' }}>{counts.CRITICAL}</div>
            <div className="kpi-sub">Danger zone / severe breach</div>
          </div>
        </div>

        {/* Filter Controls Bar */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            {/* Search Input */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              background: 'var(--bg-input)',
              border: '1px solid var(--border-secondary)',
              borderRadius: 'var(--radius-md)',
              padding: '6px 12px',
              minWidth: 220
            }}>
              <Search size={14} style={{ color: 'var(--text-muted)' }} />
              <input
                type="text"
                placeholder="Search ID, Code (W001), Name..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  outline: 'none',
                  color: 'var(--text-primary)',
                  fontSize: 13,
                  fontFamily: 'var(--font-sans)',
                  width: '100%'
                }}
              />
            </div>

            {/* Risk Level Filter Chips */}
            <div className="filter-bar" style={{ margin: 0 }}>
              {RISK_FILTERS.map(f => (
                <button
                  key={f}
                  className={`filter-chip ${filter === f ? 'active' : ''}`}
                  onClick={() => setFilter(f)}
                  style={{ fontSize: 11, padding: '4px 10px' }}
                >
                  {f} {f === 'ALL' ? `(${counts.ALL})` : `(${counts[f as keyof typeof counts] || 0})`}
                </button>
              ))}
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            {/* View Mode Toggle */}
            <div style={{ display: 'flex', background: 'rgba(255, 255, 255, 0.04)', padding: 3, borderRadius: 8, border: '1px solid var(--border-primary)' }}>
              <button
                className={`btn ${viewMode === 'table' ? 'btn-primary' : 'btn-ghost'}`}
                onClick={() => setViewMode('table')}
                style={{ padding: '4px 10px', fontSize: 12, height: 28 }}
                title="Table View"
              >
                <List size={14} /> Table
              </button>
              <button
                className={`btn ${viewMode === 'grid' ? 'btn-primary' : 'btn-ghost'}`}
                onClick={() => setViewMode('grid')}
                style={{ padding: '4px 10px', fontSize: 12, height: 28 }}
                title="Visual Card Grid View"
              >
                <LayoutGrid size={14} /> Cards
              </button>
            </div>

            <button
              className="btn btn-ghost"
              onClick={handleResetTracks}
              disabled={resetting}
              style={{ fontSize: 12, padding: '6px 12px' }}
              title="Clear all active in-memory tracker sessions and trajectories"
            >
              <RotateCcw size={14} className={resetting ? 'spin' : ''} />
              {resetting ? 'Clearing...' : 'Clear Live Tracking'}
            </button>
          </div>
        </div>

        {/* Main Content Area */}
        <div style={{ display: 'grid', gridTemplateColumns: selectedWorker ? '1fr 420px' : '1fr', gap: 16, alignItems: 'start' }}>
          
          {/* ========================================================================= */}
          {/* VIEW MODE 1: DATA TABLE */}
          {/* ========================================================================= */}
          {viewMode === 'table' && (
            <div className="card" style={{ overflow: 'auto', padding: 0 }}>
              {filteredWorkers.length > 0 ? (
                <table className="data-table">
                  <thead>
                    <tr>
                      <th style={{ width: 56 }}>Photo</th>
                      <th>Permanent ID</th>
                      <th>Identity Name</th>
                      <th>Track ID</th>
                      <th>Risk Level</th>
                      <th>Risk Score</th>
                      <th>Hardhat</th>
                      <th>Safety Vest</th>
                      <th>Live Status</th>
                      <th>Violations</th>
                      <th>Duration</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredWorkers.map(w => {
                      const isRegistered = Boolean(w.permanent_worker_id || w.worker_code);
                      const code = w.permanent_worker_id || w.worker_code;
                      const imgSrc = w.photo_url ? `${API_URL}${w.photo_url}` : w.face_crop_base64;
                      const riskLvl = (w.risk_level || 'SAFE').toUpperCase();
                      const riskColor = getRiskColor(riskLvl);

                      return (
                        <tr key={code ? `code_${code}` : `track_${w.track_id || w.worker_id}`} onClick={() => fetchWorkerDetail(w.track_id || w.worker_id)}>
                          <td>
                            <div style={{
                              width: 38,
                              height: 38,
                              borderRadius: '50%',
                              overflow: 'hidden',
                              background: '#0a0e1a',
                              border: `2px solid ${riskColor}`,
                              boxShadow: `0 0 8px ${riskColor}40`,
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              flexShrink: 0
                            }}>
                              {imgSrc ? (
                                <img
                                  src={imgSrc}
                                  alt={w.name || 'Worker'}
                                  style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                                  onError={(e) => {
                                    (e.target as HTMLElement).style.display = 'none';
                                  }}
                                />
                              ) : (
                                <Users size={16} style={{ color: 'var(--text-muted)' }} />
                              )}
                            </div>
                          </td>
                          <td style={{ fontWeight: 700, fontFamily: 'var(--font-mono)', color: isRegistered ? '#38bdf8' : 'var(--text-muted)' }}>
                            {isRegistered ? `🆔 ${code}` : 'UNKNOWN'}
                          </td>
                          <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                            {w.name || (isRegistered ? `Worker ${code}` : `Unknown Worker`)}
                          </td>
                          <td style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-muted)' }}>
                            #{w.track_id || w.worker_id}
                          </td>
                          <td>
                            <span
                              style={{
                                padding: '3px 8px',
                                borderRadius: 6,
                                fontSize: 11,
                                fontWeight: 700,
                                background: `${riskColor}20`,
                                color: riskColor,
                                border: `1px solid ${riskColor}50`
                              }}
                            >
                              {riskLvl}
                            </span>
                          </td>
                          <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                            {w.risk_score?.toFixed(0) ?? 0}
                          </td>
                          <td>{w.helmet === null ? '—' : w.helmet ? <span style={{ color: '#10b981', fontWeight: 600 }}>🪖 Worn</span> : <span style={{ color: '#ef4444', fontWeight: 600 }}>❌ Missing</span>}</td>
                          <td>{w.vest === null ? '—' : w.vest ? <span style={{ color: '#10b981', fontWeight: 600 }}>🦺 Worn</span> : <span style={{ color: '#ef4444', fontWeight: 600 }}>❌ Missing</span>}</td>
                          <td>
                            <span className={`badge ${w.is_live ? 'safe' : 'low'}`} style={{ fontSize: 10 }}>
                              {w.is_live ? 'LIVE' : 'RECORDED'}
                            </span>
                          </td>
                          <td>{w.violation_count}</td>
                          <td style={{ fontSize: 12, color: 'var(--text-muted)' }}>{w.tracking_duration?.toFixed(0) ?? 0}s</td>
                          <td onClick={e => e.stopPropagation()}>
                            <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                              {!isRegistered && (
                                <button
                                  className="btn btn-ghost"
                                  style={{ padding: '3px 8px', fontSize: 11 }}
                                  onClick={() => setLinkingWorkerId(w.track_id || w.worker_id)}
                                  title="Link track to registered worker"
                                >
                                  <Link2 size={11} /> Link
                                </button>
                              )}
                              <button
                                className="btn btn-ghost"
                                style={{ padding: '3px 8px', fontSize: 11, color: '#ef4444', borderColor: 'rgba(239, 68, 68, 0.3)' }}
                                onClick={(e) => handleDeleteWorkerTrack(w.track_id || w.worker_id, e)}
                                title={`Delete track #${w.track_id || w.worker_id} from live tracking`}
                              >
                                <Trash2 size={11} /> Delete
                              </button>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              ) : (
                <div className="empty-state" style={{ padding: 48 }}>
                  <Users size={40} style={{ opacity: 0.3, marginBottom: 12 }} />
                  <h3>No Unique Workers Found</h3>
                  <p>No active or recorded tracking entries match the "{filter}" risk criteria.</p>
                </div>
              )}
            </div>
          )}

          {/* ========================================================================= */}
          {/* VIEW MODE 2: VISUAL CARD GRID */}
          {/* ========================================================================= */}
          {viewMode === 'grid' && (
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))',
              gap: 16
            }}>
              {filteredWorkers.map(w => {
                const isRegistered = Boolean(w.permanent_worker_id || w.worker_code);
                const code = w.permanent_worker_id || w.worker_code;
                const imgSrc = w.photo_url ? `${API_URL}${w.photo_url}` : w.face_crop_base64;
                const riskLvl = (w.risk_level || 'SAFE').toUpperCase();
                const riskColor = getRiskColor(riskLvl);

                return (
                  <div
                    key={code ? `code_${code}` : `track_${w.track_id || w.worker_id}`}
                    className="card"
                    onClick={() => fetchWorkerDetail(w.track_id || w.worker_id)}
                    style={{
                      cursor: 'pointer',
                      border: `1px solid ${riskColor}40`,
                      background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.03) 0%, rgba(15, 23, 42, 0.95) 100%)',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 12,
                      padding: 16,
                      position: 'relative',
                      boxShadow: selectedWorker?.track_id === (w.track_id || w.worker_id) ? `0 0 16px ${riskColor}40` : undefined
                    }}
                  >
                    {/* Top Header: Photo + Identity */}
                    <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
                      <div style={{
                        width: 52,
                        height: 52,
                        borderRadius: 12,
                        overflow: 'hidden',
                        background: '#050814',
                        border: `2px solid ${riskColor}`,
                        boxShadow: `0 0 12px ${riskColor}40`,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        flexShrink: 0
                      }}>
                        {imgSrc ? (
                          <img
                            src={imgSrc}
                            alt={w.name || 'Worker'}
                            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                            onError={(e) => {
                              (e.target as HTMLElement).style.display = 'none';
                            }}
                          />
                        ) : (
                          <Users size={24} style={{ color: 'var(--text-muted)' }} />
                        )}
                      </div>

                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {w.name || (isRegistered ? `Worker ${code}` : `Unknown Track #${w.track_id || w.worker_id}`)}
                        </div>
                        <div style={{ fontSize: 11.5, fontFamily: 'var(--font-mono)', color: isRegistered ? '#38bdf8' : 'var(--text-muted)', marginTop: 2 }}>
                          {isRegistered ? `🆔 ${code}` : `Track #${w.track_id || w.worker_id}`}
                        </div>
                      </div>
                    </div>

                    {/* Risk Level Badge Banner */}
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      background: `${riskColor}15`,
                      padding: '6px 10px',
                      borderRadius: 8,
                      border: `1px solid ${riskColor}35`
                    }}>
                      <span style={{ fontSize: 12, fontWeight: 700, color: riskColor }}>
                        {riskLvl} RISK
                      </span>
                      <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', fontWeight: 600 }}>
                        Score: {w.risk_score?.toFixed(0) ?? 0}/100
                      </span>
                    </div>

                    {/* PPE Status Chips */}
                    <div style={{ display: 'flex', gap: 6, justifyContent: 'space-between', fontSize: 11 }}>
                      <span style={{
                        padding: '3px 8px',
                        borderRadius: 6,
                        background: w.helmet ? 'rgba(16, 185, 129, 0.12)' : 'rgba(239, 68, 68, 0.15)',
                        color: w.helmet ? '#34d399' : '#f87171',
                        border: `1px solid ${w.helmet ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
                        fontWeight: 600
                      }}>
                        {w.helmet ? '🪖 Hardhat Worn' : '❌ No Hardhat'}
                      </span>
                      <span style={{
                        padding: '3px 8px',
                        borderRadius: 6,
                        background: w.vest ? 'rgba(16, 185, 129, 0.12)' : 'rgba(239, 68, 68, 0.15)',
                        color: w.vest ? '#34d399' : '#f87171',
                        border: `1px solid ${w.vest ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
                        fontWeight: 600
                      }}>
                        {w.vest ? '🦺 Vest Worn' : '❌ No Vest'}
                      </span>
                    </div>

                    {/* Actions Row */}
                    <div style={{ display: 'flex', gap: 6, justifyContent: 'flex-end', marginTop: 4 }} onClick={e => e.stopPropagation()}>
                      {!isRegistered && (
                        <button
                          className="btn btn-ghost"
                          style={{ padding: '3px 8px', fontSize: 11 }}
                          onClick={() => setLinkingWorkerId(w.track_id || w.worker_id)}
                        >
                          <Link2 size={11} /> Link
                        </button>
                      )}
                      <button
                        className="btn btn-ghost"
                        style={{ padding: '3px 8px', fontSize: 11, color: '#ef4444', borderColor: 'rgba(239, 68, 68, 0.3)' }}
                        onClick={(e) => handleDeleteWorkerTrack(w.track_id || w.worker_id, e)}
                        title={`Delete track #${w.track_id || w.worker_id}`}
                      >
                        <Trash2 size={11} /> Delete
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* ========================================================================= */}
          {/* WORKER DETAIL INSPECTION DRAWER */}
          {/* ========================================================================= */}
          {selectedWorker && (
            <div className="card slide-in" style={{ border: `1px solid ${getRiskColor(selectedWorker.risk_level)}50` }}>
              <div className="card-header">
                <div>
                  <span className="card-title">
                    {selectedWorker.permanent_worker_id ? `${selectedWorker.permanent_worker_id} · ` : ''}
                    Track #{selectedWorker.track_id}
                  </span>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>
                    {selectedWorker.name || 'Unknown Worker'}
                  </div>
                </div>
                <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                  <button
                    className="btn btn-ghost"
                    style={{ color: '#ef4444', borderColor: 'rgba(239, 68, 68, 0.3)', padding: '4px 8px', fontSize: 11 }}
                    onClick={() => handleDeleteWorkerTrack(selectedWorker.track_id || selectedWorker.worker_id)}
                    title="Delete this track from live tracking"
                  >
                    <Trash2 size={12} /> Delete Track
                  </button>
                  <button className="btn btn-ghost" onClick={() => setSelectedWorker(null)} style={{ padding: '4px 8px', fontSize: 12 }}>✕</button>
                </div>
              </div>

              {/* Worker Profile Photo Showcase */}
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: 16,
                padding: '14px 16px',
                background: 'rgba(255, 255, 255, 0.03)',
                borderRadius: 12,
                border: `1px solid ${getRiskColor(selectedWorker.risk_level)}40`,
                marginBottom: 16,
                boxShadow: `0 0 16px ${getRiskColor(selectedWorker.risk_level)}20`
              }}>
                <div style={{
                  width: 72,
                  height: 72,
                  borderRadius: 14,
                  overflow: 'hidden',
                  background: '#0a0e1a',
                  border: `2.5px solid ${getRiskColor(selectedWorker.risk_level)}`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0,
                  boxShadow: '0 4px 16px rgba(0,0,0,0.4)'
                }}>
                  {selectedWorker.photo_url || selectedWorker.face_crop_base64 ? (
                    <img
                      src={selectedWorker.photo_url ? `${API_URL}${selectedWorker.photo_url}` : selectedWorker.face_crop_base64}
                      alt={selectedWorker.name || 'Worker'}
                      style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                      onError={(e) => {
                        (e.target as HTMLElement).style.display = 'none';
                      }}
                    />
                  ) : (
                    <Users size={32} style={{ color: 'var(--text-muted)' }} />
                  )}
                </div>
                <div>
                  <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)' }}>
                    {selectedWorker.name || (selectedWorker.permanent_worker_id ? `Worker ${selectedWorker.permanent_worker_id}` : `Live Track #${selectedWorker.track_id}`)}
                  </div>
                  <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginTop: 6, flexWrap: 'wrap' }}>
                    <span className={`badge ${selectedWorker.permanent_worker_id ? 'safe' : 'medium'}`} style={{ fontSize: 11 }}>
                      {selectedWorker.permanent_worker_id ? `ID: ${selectedWorker.permanent_worker_id}` : 'UNREGISTERED'}
                    </span>
                    <span
                      style={{
                        padding: '2px 8px',
                        borderRadius: 6,
                        fontSize: 11,
                        fontWeight: 700,
                        background: `${getRiskColor(selectedWorker.risk_level)}20`,
                        color: getRiskColor(selectedWorker.risk_level),
                        border: `1px solid ${getRiskColor(selectedWorker.risk_level)}50`
                      }}
                    >
                      {(selectedWorker.risk_level || 'SAFE').toUpperCase()} RISK
                    </span>
                  </div>
                </div>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                  <span style={{ color: 'var(--text-muted)' }}>Identity Status</span>
                  <span className={`badge ${selectedWorker.permanent_worker_id ? 'safe' : 'medium'}`}>
                    {selectedWorker.permanent_worker_id ? 'REGISTERED' : 'UNKNOWN'}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                  <span style={{ color: 'var(--text-muted)' }}>Permanent Worker Code</span>
                  <span style={{ fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--accent-blue)' }}>
                    {selectedWorker.permanent_worker_id || 'Not Assigned'}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                  <span style={{ color: 'var(--text-muted)' }}>Temporary Track ID</span>
                  <span style={{ fontFamily: 'var(--font-mono)' }}>#{selectedWorker.track_id}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                  <span style={{ color: 'var(--text-muted)' }}>Live Risk Score</span>
                  <span style={{ fontWeight: 700, color: getRiskColor(selectedWorker.risk_level) }}>
                    {selectedWorker.risk_score?.toFixed(0) ?? 0} / 100
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                  <span style={{ color: 'var(--text-muted)' }}>Helmet Protection</span>
                  <span>{selectedWorker.helmet === null ? 'Not checked' : selectedWorker.helmet ? '🪖✓ Worn' : '❌ Missing'}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                  <span style={{ color: 'var(--text-muted)' }}>High-Vis Safety Vest</span>
                  <span>{selectedWorker.vest === null ? 'Not checked' : selectedWorker.vest ? '🦺✓ Worn' : '❌ Missing'}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                  <span style={{ color: 'var(--text-muted)' }}>Tracking Duration</span>
                  <span>{selectedWorker.tracking_duration?.toFixed(0) ?? 0}s</span>
                </div>

                {/* Manual Link Identity Section */}
                {!selectedWorker.permanent_worker_id && (
                  <div style={{ marginTop: 8, padding: 10, background: 'var(--bg-surface)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-primary)' }}>
                    <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 6 }}>
                      Manual Identity Link
                    </div>
                    <div style={{ display: 'flex', gap: 6 }}>
                      <select
                        className="control-select"
                        value={selectedRegCode}
                        onChange={e => setSelectedRegCode(e.target.value)}
                        style={{ flex: 1 }}
                      >
                        <option value="">Select Registered Worker...</option>
                        {registeredList.map(r => (
                          <option key={r.worker_code} value={r.worker_code}>
                            {r.worker_code} - {r.name} ({r.employee_number})
                          </option>
                        ))}
                      </select>
                      <button
                        className="btn btn-primary"
                        style={{ padding: '6px 10px', fontSize: 12 }}
                        disabled={!selectedRegCode}
                        onClick={() => handleLinkIdentity(selectedWorker.track_id)}
                      >
                        <CheckCircle size={12} /> Link
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Quick Link Modal */}
        {linkingWorkerId && (
          <div style={{
            position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
            background: 'rgba(0,0,0,0.75)', zIndex: 9999,
            display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20,
          }}>
            <div className="card fade-in" style={{ width: '100%', maxWidth: 420, padding: 20 }}>
              <div className="card-header">
                <span className="card-title">Link Track #{linkingWorkerId} to Registered Worker</span>
                <button className="btn btn-ghost" onClick={() => setLinkingWorkerId(null)} style={{ padding: '4px 8px' }}>✕</button>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginTop: 12 }}>
                <select
                  className="control-select"
                  value={selectedRegCode}
                  onChange={e => setSelectedRegCode(e.target.value)}
                  style={{ width: '100%' }}
                >
                  <option value="">Select Registered Worker...</option>
                  {registeredList.map(r => (
                    <option key={r.worker_code} value={r.worker_code}>
                      {r.worker_code} - {r.name} ({r.employee_number})
                    </option>
                  ))}
                </select>
                <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
                  <button className="btn btn-ghost" onClick={() => setLinkingWorkerId(null)}>Cancel</button>
                  <button
                    className="btn btn-primary"
                    disabled={!selectedRegCode}
                    onClick={() => handleLinkIdentity(linkingWorkerId)}
                  >
                    Confirm Link
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

      </div>
    </>
  );
}
