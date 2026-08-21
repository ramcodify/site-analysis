import { useState, useEffect, useRef, useCallback } from 'react';
import { Header } from '../components/common/Header';
import type { RegisteredWorker, RegisteredWorkerDetail, QualityCheckResult } from '../types';
import {
  Users, UserPlus,
  Camera, Upload, RefreshCw, X, Eye, Power, Lock, Sparkles, Trash2, Download, FileSpreadsheet
} from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const SAMPLE_PROMPTS = [
  { id: 'front', label: '1. Front Facing', desc: 'Look directly into camera with neutral expression' },
  { id: 'left', label: '2. Slight Left', desc: 'Turn head slightly (15°) to your left' },
  { id: 'right', label: '3. Slight Right', desc: 'Turn head slightly (15°) to your right' },
  { id: 'smile', label: '4. Natural Expression', desc: 'Slight smile or open mouth expression' },
];

export default function RegisteredWorkers() {
  const [workers, setWorkers] = useState<RegisteredWorker[]>([]);
  const [selectedWorker, setSelectedWorker] = useState<RegisteredWorkerDetail | null>(null);
  const [showRegisterModal, setShowRegisterModal] = useState(false);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Form State
  const [name, setName] = useState('');
  const [empNo, setEmpNo] = useState('');
  const [dept, setDept] = useState('Construction Core');
  const [role, setRole] = useState('Site Operative');

  // Multi-sample captures state
  const [capturedSamples, setCapturedSamples] = useState<string[]>([]);
  const [currentPromptIdx, setCurrentPromptIdx] = useState(0);
  const [captureMode, setCaptureMode] = useState<'camera' | 'upload'>('camera');
  const [qualityFeedback, setQualityFeedback] = useState<QualityCheckResult | null>(null);

  // Camera Refs
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const fetchWorkers = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/api/registered-workers`);
      if (res.ok) setWorkers(await res.json());
    } catch {
      // Backend not reached
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchWorkers();
  }, [fetchWorkers]);

  // ── Camera Management ─────────────────────────────────────────

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
    } catch (err) {
      console.warn('Camera start failed:', err);
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
  };

  useEffect(() => {
    if (showRegisterModal && captureMode === 'camera') {
      startCamera();
    } else {
      stopCamera();
    }
    return () => stopCamera();
  }, [showRegisterModal, captureMode]);

  // ── Real-time Quality Verification ────────────────────────────

  const captureFrameBase64 = (): string | null => {
    const video = videoRef.current;
    if (!video || video.readyState < 2) return null;
    const canvas = canvasRef.current || document.createElement('canvas');
    const w = video.videoWidth || 640;
    const h = video.videoHeight || 480;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    // Flip horizontally to undo mirror so face recognition gets correct orientation
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, w, h);
    ctx.restore();
    return canvas.toDataURL('image/jpeg', 0.85);
  };

  const checkQuality = async (imgB64: string): Promise<QualityCheckResult | null> => {
    try {
      const res = await fetch(`${API_URL}/api/registered-workers/validate-image`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imgB64 }),
      });
      if (res.ok) {
        const data = await res.json();
        setQualityFeedback(data);
        return data;
      }
    } catch {
      // ignore
    }
    return null;
  };

  const handleCaptureSample = async () => {
    const b64 = captureFrameBase64();
    if (!b64) return;

    const quality = await checkQuality(b64);
    if (quality && quality.is_valid) {
      setCapturedSamples(prev => [...prev, b64]);
      if (currentPromptIdx < SAMPLE_PROMPTS.length - 1) {
        setCurrentPromptIdx(prev => prev + 1);
      }
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const reader = new FileReader();
      reader.onload = async (event) => {
        const b64 = event.target?.result as string;
        if (b64) {
          const quality = await checkQuality(b64);
          if (quality && quality.is_valid) {
            setCapturedSamples(prev => [...prev, b64]);
          }
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const handleRemoveSample = (index: number) => {
    setCapturedSamples(prev => prev.filter((_, i) => i !== index));
  };

  // ── Register Submit ───────────────────────────────────────────

  const handleSubmitRegistration = async () => {
    setErrorMsg(null);
    if (!name.trim() || !empNo.trim()) {
      setErrorMsg('Worker name and employee number are required');
      return;
    }
    if (capturedSamples.length === 0) {
      setErrorMsg('Please capture at least 1 high-quality face sample');
      return;
    }

    setSubmitting(true);
    try {
      const res = await fetch(`${API_URL}/api/registered-workers`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: name.trim(),
          employee_number: empNo.trim(),
          department: dept.trim(),
          role: role.trim(),
          images: capturedSamples,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Registration failed');
      }

      await fetchWorkers();
      setShowRegisterModal(false);
      resetForm();
    } catch (err: any) {
      setErrorMsg(err.message || 'Registration failed');
    } finally {
      setSubmitting(false);
    }
  };

  const resetForm = () => {
    setName('');
    setEmpNo('');
    setDept('Construction Core');
    setRole('Site Operative');
    setCapturedSamples([]);
    setCurrentPromptIdx(0);
    setQualityFeedback(null);
    setErrorMsg(null);
  };

  const fetchNextEmpNo = async () => {
    try {
      const res = await fetch(`${API_URL}/api/registered-workers/next-code`);
      if (res.ok) {
        const data = await res.json();
        setEmpNo(data.next_employee_number || `EMP-${String(workers.length + 1).padStart(3, '0')}`);
      } else {
        setEmpNo(`EMP-${String(workers.length + 1).padStart(3, '0')}`);
      }
    } catch {
      setEmpNo(`EMP-${String(workers.length + 1).padStart(3, '0')}`);
    }
  };

  const openRegisterModal = async () => {
    resetForm();
    await fetchNextEmpNo();
    setShowRegisterModal(true);
  };

  const openWorkerDetail = async (workerCode: string) => {
    try {
      const res = await fetch(`${API_URL}/api/registered-workers/${workerCode}`);
      if (res.ok) setSelectedWorker(await res.json());
    } catch {
      // ignore
    }
  };

  const toggleWorkerStatus = async (workerCode: string, currentStatus: string) => {
    const newStatus = currentStatus === 'ACTIVE' ? 'INACTIVE' : 'ACTIVE';
    try {
      await fetch(`${API_URL}/api/registered-workers/${workerCode}/status`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ active_status: newStatus }),
      });
      await fetchWorkers();
      if (selectedWorker?.worker_code === workerCode) {
        setSelectedWorker(prev => prev ? { ...prev, active_status: newStatus as any } : null);
      }
    } catch {
      // ignore
    }
  };

  const deleteWorker = async (workerCode: string, workerName: string) => {
    if (!window.confirm(`Are you sure you want to permanently delete registered worker ${workerCode} (${workerName})?\n\nThis will remove their profile and all biometric templates from MongoDB.`)) {
      return;
    }
    try {
      const res = await fetch(`${API_URL}/api/registered-workers/${workerCode}?permanent=true`, {
        method: 'DELETE',
      });
      if (res.ok) {
        if (selectedWorker?.worker_code === workerCode) {
          setSelectedWorker(null);
        }
        await fetchWorkers();
      }
    } catch (err) {
      console.error("Failed to delete worker:", err);
    }
  };

  const clearAllWorkers = async () => {
    if (!window.confirm(`⚠️ DANGER: Are you sure you want to permanently delete ALL ${workers.length} registered workers?\n\nThis will remove all worker profiles, IDs, employee records, and face biometric models from MongoDB. This action cannot be undone.`)) {
      return;
    }
    try {
      const res = await fetch(`${API_URL}/api/registered-workers`, {
        method: 'DELETE',
      });
      if (res.ok) {
        setSelectedWorker(null);
        await fetchWorkers();
      }
    } catch (err) {
      console.error("Failed to clear all registered workers:", err);
    }
  };

  const downloadWorkersCSV = () => {
    window.open(`${API_URL}/api/reports/workers/export/csv`, '_blank');
  };

  const activeCount = workers.filter(w => w.active_status === 'ACTIVE').length;

  return (
    <>
      <Header
        title="Registered Workers"
        subtitle="Permanent biometric identity management & multi-sample registration"
      />
      <div className="app-content">

        {/* Top Summary KPI Cards */}
        <div className="kpi-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)', marginBottom: 20 }}>
          <div className="kpi-card cyan">
            <div className="kpi-label">Registered Workers</div>
            <div className="kpi-value">{workers.length}</div>
            <div className="kpi-sub">Permanent Biometric Identities</div>
          </div>
          <div className="kpi-card green">
            <div className="kpi-label">Active Status</div>
            <div className="kpi-value">{activeCount}</div>
            <div className="kpi-sub">{workers.length - activeCount} Inactive</div>
          </div>
          <div className="kpi-card blue">
            <div className="kpi-label">Biometric Model</div>
            <div className="kpi-value" style={{ fontSize: 18, marginTop: 4 }}>YuNet + SFace</div>
            <div className="kpi-sub">128-d L2 Local Embeddings</div>
          </div>
          <div className="kpi-card purple">
            <div className="kpi-label">Security & Privacy</div>
            <div className="kpi-value" style={{ fontSize: 18, marginTop: 4 }}>Protected</div>
            <div className="kpi-sub">Zero Raw Embeddings Exposed</div>
          </div>
        </div>

        {/* Action Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>
            Total Permanent Workers: <strong style={{ color: 'var(--text-primary)' }}>{workers.length}</strong>
          </div>
          <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
            <button className="btn btn-ghost" onClick={fetchWorkers} disabled={loading}>
              <RefreshCw size={14} className={loading ? 'spin' : ''} /> Refresh
            </button>
            <button
              className="btn btn-ghost"
              onClick={() => window.open(`${API_URL}/api/reports/workers/export/xlsx`, '_blank')}
              disabled={workers.length === 0}
              title="Download Microsoft Excel (.xlsx) sheet with embedded face photos, IDs, roles & compliance metrics"
              style={{ borderColor: 'rgba(34, 197, 94, 0.4)', color: 'var(--green)' }}
            >
              <FileSpreadsheet size={14} /> Download Excel Sheet (.xlsx with Photos)
            </button>
            <button
              className="btn btn-ghost"
              onClick={downloadWorkersCSV}
              disabled={workers.length === 0}
              title="Download full worker roster spreadsheet (CSV) with photos and IDs"
              style={{ borderColor: 'rgba(56, 189, 248, 0.4)', color: 'var(--cyan)' }}
            >
              <Download size={14} /> Download CSV
            </button>
            {workers.length > 0 && (
              <button
                className="btn btn-ghost"
                onClick={clearAllWorkers}
                title="Clear all registered workers from database and biometric cache"
                style={{ borderColor: 'rgba(239, 68, 68, 0.4)', color: 'var(--red)' }}
              >
                <Trash2 size={14} /> Clear All Registered Workers
              </button>
            )}
            <button className="btn btn-primary" onClick={openRegisterModal}>
              <UserPlus size={16} /> + Register New Worker
            </button>
          </div>
        </div>

        {/* Registered Workers List / Drawer Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: selectedWorker ? '1fr 400px' : '1fr', gap: 16 }}>

          {/* Table */}
          <div className="card" style={{ overflow: 'auto' }}>
            {workers.length > 0 ? (
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Photo</th>
                    <th>Worker ID</th>
                    <th>Name</th>
                    <th>Employee No.</th>
                    <th>Department</th>
                    <th>Role</th>
                    <th>Templates</th>
                    <th>Status</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {workers.map((w) => (
                    <tr key={w.worker_code} onClick={() => openWorkerDetail(w.worker_code)} style={{ cursor: 'pointer' }}>
                      <td style={{ width: 44 }}>
                        {w.profile_image_path ? (
                          <img
                            src={`${API_URL}${w.profile_image_path}`}
                            alt={w.name}
                            style={{ width: 34, height: 34, borderRadius: '50%', objectFit: 'cover', border: '1px solid var(--border-secondary)' }}
                          />
                        ) : (
                          <div style={{ width: 34, height: 34, borderRadius: '50%', background: 'var(--bg-surface)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <Users size={16} style={{ color: 'var(--text-muted)' }} />
                          </div>
                        )}
                      </td>
                      <td style={{ fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--accent-blue)' }}>
                        {w.worker_code}
                      </td>
                      <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                        {w.name}
                      </td>
                      <td style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                        {w.employee_number}
                      </td>
                      <td>{w.department}</td>
                      <td>{w.role}</td>
                      <td style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                        {w.total_embeddings || 1} samples
                      </td>
                      <td>
                        <span className={`badge ${w.active_status === 'ACTIVE' ? 'safe' : 'medium'}`}>
                          {w.active_status}
                        </span>
                      </td>
                      <td onClick={e => e.stopPropagation()}>
                        <div style={{ display: 'flex', gap: 6 }}>
                          <button
                            className="btn btn-ghost"
                            style={{ padding: '3px 8px', fontSize: 11 }}
                            onClick={() => openWorkerDetail(w.worker_code)}
                          >
                            <Eye size={11} /> View
                          </button>
                          <button
                            className={`btn ${w.active_status === 'ACTIVE' ? 'btn-ghost' : 'btn-success'}`}
                            style={{ padding: '3px 8px', fontSize: 11 }}
                            onClick={() => toggleWorkerStatus(w.worker_code, w.active_status)}
                          >
                            <Power size={11} /> {w.active_status === 'ACTIVE' ? 'Deactivate' : 'Activate'}
                          </button>
                          <button
                            className="btn btn-danger"
                            style={{ padding: '3px 8px', fontSize: 11 }}
                            title="Permanently Delete Worker"
                            onClick={() => deleteWorker(w.worker_code, w.name)}
                          >
                            <Trash2 size={11} /> Delete
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="empty-state" style={{ padding: 40 }}>
                <Users size={48} style={{ opacity: 0.3, marginBottom: 12 }} />
                <h3>No Registered Workers Found</h3>
                <p>Register permanent worker identities to enable real-time facial recognition and track merging.</p>
                <button className="btn btn-primary" style={{ marginTop: 16 }} onClick={() => { resetForm(); setShowRegisterModal(true); }}>
                  <UserPlus size={16} /> Register First Worker
                </button>
              </div>
            )}
          </div>

          {/* Worker Detail Sidebar Drawer */}
          {selectedWorker && (
            <div className="card slide-in">
              <div className="card-header">
                <span className="card-title">Permanent Identity: {selectedWorker.worker_code}</span>
                <button className="btn btn-ghost" onClick={() => setSelectedWorker(null)} style={{ padding: '4px 8px', fontSize: 12 }}>
                  <X size={14} />
                </button>
              </div>

              {/* Profile Header */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 16, paddingBottom: 14, borderBottom: '1px solid var(--border-primary)' }}>
                {selectedWorker.profile_image_path ? (
                  <img
                    src={`${API_URL}${selectedWorker.profile_image_path}`}
                    alt={selectedWorker.name}
                    style={{ width: 64, height: 64, borderRadius: '50%', objectFit: 'cover', border: '2px solid var(--accent-blue)' }}
                  />
                ) : (
                  <div style={{ width: 64, height: 64, borderRadius: '50%', background: 'var(--bg-surface)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Users size={28} style={{ color: 'var(--text-muted)' }} />
                  </div>
                )}
                <div>
                  <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)' }}>{selectedWorker.name}</div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>{selectedWorker.role} · {selectedWorker.department}</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--accent-blue)', marginTop: 4 }}>{selectedWorker.employee_number}</div>
                </div>
              </div>

              {/* Stats Grid */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 16 }}>
                <div style={{ background: 'var(--bg-surface)', padding: 10, borderRadius: 6, border: '1px solid var(--border-primary)' }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Status</div>
                  <div style={{ fontWeight: 600, fontSize: 13, marginTop: 2, color: selectedWorker.is_currently_active ? 'var(--safe)' : 'var(--text-secondary)' }}>
                    {selectedWorker.is_currently_active ? '● Live on Site' : '○ Offline'}
                  </div>
                </div>
                <div style={{ background: 'var(--bg-surface)', padding: 10, borderRadius: 6, border: '1px solid var(--border-primary)' }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Violations</div>
                  <div style={{ fontWeight: 600, fontSize: 13, marginTop: 2, color: selectedWorker.total_violations_count > 0 ? 'var(--accent-red)' : 'var(--safe)' }}>
                    {selectedWorker.total_violations_count}
                  </div>
                </div>
                <div style={{ background: 'var(--bg-surface)', padding: 10, borderRadius: 6, border: '1px solid var(--border-primary)' }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Avg PPE Compliance</div>
                  <div style={{ fontWeight: 600, fontSize: 13, marginTop: 2 }}>
                    {selectedWorker.avg_ppe_compliance?.toFixed(1) ?? '—'}%
                  </div>
                </div>
                <div style={{ background: 'var(--bg-surface)', padding: 10, borderRadius: 6, border: '1px solid var(--border-primary)' }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Latest Risk Level</div>
                  <div style={{ fontWeight: 600, fontSize: 13, marginTop: 2 }}>
                    <span className={`badge ${selectedWorker.latest_risk_level?.toLowerCase() || 'safe'}`}>
                      {selectedWorker.latest_risk_level || 'SAFE'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Lifetime Merged Statistics */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.6px' }}>
                  Merged Lifetime Statistics
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                  <span style={{ color: 'var(--text-muted)' }}>Employee ID</span>
                  <span style={{ fontFamily: 'var(--font-mono)' }}>{selectedWorker.employee_number}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                  <span style={{ color: 'var(--text-muted)' }}>Lifetime Tracking Duration</span>
                  <span style={{ fontWeight: 600 }}>{selectedWorker.lifetime_tracking_duration?.toFixed(0)}s</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                  <span style={{ color: 'var(--text-muted)' }}>Avg. PPE Compliance</span>
                  <span style={{ fontWeight: 600, color: selectedWorker.avg_ppe_compliance >= 80 ? 'var(--accent-green)' : 'var(--accent-amber)' }}>
                    {selectedWorker.avg_ppe_compliance?.toFixed(1)}%
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                  <span style={{ color: 'var(--text-muted)' }}>Total Linked Violations</span>
                  <span style={{ fontWeight: 600, color: selectedWorker.total_violations_count > 0 ? 'var(--accent-red)' : 'var(--text-primary)' }}>
                    {selectedWorker.total_violations_count}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                  <span style={{ color: 'var(--text-muted)' }}>Last Recognized</span>
                  <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                    {selectedWorker.last_recognized ? new Date(selectedWorker.last_recognized).toLocaleString() : 'Never'}
                  </span>
                </div>

                <div style={{ marginTop: 14, paddingTop: 14, borderTop: '1px solid var(--border-primary)', display: 'flex', flexDirection: 'column', gap: 8 }}>
                  <button
                    className={`btn ${selectedWorker.active_status === 'ACTIVE' ? 'btn-ghost' : 'btn-success'}`}
                    style={{ width: '100%', justifyContent: 'center' }}
                    onClick={() => toggleWorkerStatus(selectedWorker.worker_code, selectedWorker.active_status)}
                  >
                    <Power size={14} /> {selectedWorker.active_status === 'ACTIVE' ? 'Deactivate Worker' : 'Activate Worker'}
                  </button>
                  <button
                    className="btn btn-danger"
                    style={{ width: '100%', justifyContent: 'center' }}
                    onClick={() => deleteWorker(selectedWorker.worker_code, selectedWorker.name)}
                  >
                    <Trash2 size={14} /> Delete Worker Record Permanently
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* ── Register New Worker Modal ────────────────────────────── */}
        {showRegisterModal && (
          <div style={{
            position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
            background: 'rgba(0,0,0,0.75)', zIndex: 9999,
            display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20,
          }}>
            <div className="card fade-in" style={{ width: '100%', maxWidth: 760, maxHeight: '92vh', overflowY: 'auto', padding: 24, border: '1px solid var(--border-accent)' }}>

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 18 }}>
                <div>
                  <h2 style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>Register Permanent Worker</h2>
                  <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
                    Multi-sample face registration for high-accuracy local facial recognition (YuNet + SFace)
                  </p>
                </div>
                <button className="btn btn-ghost" onClick={() => { setShowRegisterModal(false); stopCamera(); }}>
                  <X size={18} />
                </button>
              </div>

              {errorMsg && (
                <div style={{ padding: '10px 14px', background: 'rgba(239, 68, 68, 0.12)', borderLeft: '3px solid var(--accent-red)', borderRadius: 6, color: 'var(--accent-red)', fontSize: 13, marginBottom: 16 }}>
                  {errorMsg}
                </div>
              )}

              {/* Form Fields */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 18 }}>
                <div>
                  <label style={{ fontSize: 12, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>Full Name *</label>
                  <input
                    type="text"
                    className="input-field"
                    placeholder="e.g. Alice Smith"
                    value={name}
                    onChange={e => setName(e.target.value)}
                  />
                </div>
                <div>
                  <label style={{ fontSize: 12, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>
                    Employee Number
                    <span style={{ marginLeft: 6, fontSize: 10, color: 'var(--accent-blue)', background: 'rgba(59,130,246,0.12)', borderRadius: 3, padding: '1px 5px' }}>
                      AUTO-GENERATED
                    </span>
                  </label>
                  <div style={{
                    display: 'flex', alignItems: 'center', gap: 8,
                    background: 'var(--bg-surface)', border: '1px solid var(--border-primary)',
                    borderRadius: 'var(--radius-md)', padding: '8px 12px',
                  }}>
                    <Lock size={13} style={{ color: 'var(--accent-blue)', flexShrink: 0 }} />
                    <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: 15, color: 'var(--accent-blue)', letterSpacing: 1, flex: 1 }}>
                      {empNo || '—'}
                    </span>
                    <button
                      type="button"
                      className="btn btn-ghost"
                      style={{ padding: '2px 7px', fontSize: 11 }}
                      onClick={fetchNextEmpNo}
                      title="Re-fetch next available Employee ID from database"
                    >
                      <RefreshCw size={11} /> Refresh
                    </button>
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 3 }}>
                    Assigned in real-time from DB · Next in series
                  </div>
                </div>
                <div>
                  <label style={{ fontSize: 12, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>Department</label>
                  <input
                    type="text"
                    className="input-field"
                    placeholder="e.g. Civil Engineering"
                    value={dept}
                    onChange={e => setDept(e.target.value)}
                  />
                </div>
                <div>
                  <label style={{ fontSize: 12, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>Role / Trade</label>
                  <input
                    type="text"
                    className="input-field"
                    placeholder="e.g. Scaffold Inspector"
                    value={role}
                    onChange={e => setRole(e.target.value)}
                  />
                </div>
              </div>

              {/* Multi-Sample Capture Section */}
              <div style={{ background: 'var(--bg-surface)', borderRadius: 'var(--radius-lg)', padding: 16, border: '1px solid var(--border-primary)', marginBottom: 18 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: 6 }}>
                    <Sparkles size={14} style={{ color: 'var(--accent-blue)' }} /> Face Biometric Samples ({capturedSamples.length}/4 minimum 1)
                  </div>
                  <div style={{ display: 'flex', gap: 6 }}>
                    <button
                      className={`btn ${captureMode === 'camera' ? 'btn-primary' : 'btn-ghost'}`}
                      style={{ padding: '4px 10px', fontSize: 12 }}
                      onClick={() => setCaptureMode('camera')}
                    >
                      <Camera size={12} /> Live Camera
                    </button>
                    <button
                      className={`btn ${captureMode === 'upload' ? 'btn-primary' : 'btn-ghost'}`}
                      style={{ padding: '4px 10px', fontSize: 12 }}
                      onClick={() => setCaptureMode('upload')}
                    >
                      <Upload size={12} /> Upload Files
                    </button>
                  </div>
                </div>

                {captureMode === 'camera' ? (
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 240px', gap: 14 }}>
                    {/* Live Camera View */}
                    <div style={{ position: 'relative', background: '#000', borderRadius: 'var(--radius-md)', overflow: 'hidden', height: 260, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <video
                        ref={videoRef}
                        playsInline
                        muted
                        style={{ width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)' }}
                      />
                      {/* Face Target Box */}
                      <div style={{
                        position: 'absolute', width: 140, height: 170,
                        border: `2px dashed ${qualityFeedback?.is_valid ? 'var(--accent-green)' : 'var(--accent-blue)'}`,
                        borderRadius: 14, pointerEvents: 'none',
                      }} />
                      {/* Mirror indicator */}
                      <div style={{ position: 'absolute', top: 8, right: 8, background: 'rgba(0,0,0,0.6)', padding: '2px 7px', borderRadius: 4, fontSize: 10, color: '#fff', display: 'flex', alignItems: 'center', gap: 3 }}>
                        🪞 Mirror
                      </div>
                      <div style={{ position: 'absolute', bottom: 8, left: 8, right: 8, textAlign: 'center', background: 'rgba(0,0,0,0.65)', padding: '4px 8px', borderRadius: 4, fontSize: 11, color: '#fff' }}>
                        {SAMPLE_PROMPTS[currentPromptIdx]?.desc}
                      </div>
                    </div>

                    {/* Quality Feedback & Capture Button */}
                    <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                      <div>
                        <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', marginBottom: 6 }}>
                          {SAMPLE_PROMPTS[currentPromptIdx]?.label}
                        </div>
                        {qualityFeedback && (
                          <div style={{ fontSize: 12, display: 'flex', flexDirection: 'column', gap: 4, marginBottom: 10 }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                              <span>Face:</span>
                              <span style={{ color: qualityFeedback.face_detected ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                                {qualityFeedback.face_detected ? '✓ Detected' : '✗ None'}
                              </span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                              <span>Sharpness:</span>
                              <span style={{ color: qualityFeedback.sharpness_score >= 35 ? 'var(--accent-green)' : 'var(--accent-amber)' }}>
                                {qualityFeedback.sharpness_score}
                              </span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                              <span>Quality Score:</span>
                              <span style={{ fontWeight: 600 }}>{(qualityFeedback.score * 100).toFixed(0)}%</span>
                            </div>
                            {qualityFeedback.issues.length > 0 && (
                              <div style={{ fontSize: 11, color: 'var(--accent-red)', marginTop: 4 }}>
                                ⚠ {qualityFeedback.issues[0]}
                              </div>
                            )}
                          </div>
                        )}
                      </div>

                      <button
                        className="btn btn-success"
                        style={{ width: '100%', justifyContent: 'center', padding: '10px' }}
                        onClick={handleCaptureSample}
                      >
                        <Camera size={16} /> Capture Sample #{capturedSamples.length + 1}
                      </button>
                    </div>
                  </div>
                ) : (
                  <div>
                    <label style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: 30, border: '2px dashed var(--border-secondary)', borderRadius: 'var(--radius-md)', cursor: 'pointer', background: 'var(--bg-input)' }}>
                      <Upload size={28} style={{ color: 'var(--text-muted)', marginBottom: 8 }} />
                      <span style={{ fontSize: 13, fontWeight: 600 }}>Click to Upload Multi-Angle Photos</span>
                      <span style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>Select 2-5 clear face photos (.jpg, .png)</span>
                      <input type="file" multiple accept="image/*" onChange={handleFileUpload} style={{ display: 'none' }} />
                    </label>
                  </div>
                )}

                {/* Captured Sample Thumbnails */}
                {capturedSamples.length > 0 && (
                  <div style={{ marginTop: 14 }}>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>CAPTURED TEMPLATES ({capturedSamples.length}):</div>
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                      {capturedSamples.map((s, idx) => (
                        <div key={idx} style={{ position: 'relative', width: 60, height: 60, borderRadius: 6, overflow: 'hidden', border: '1px solid var(--accent-green)' }}>
                          <img src={s} alt={`Sample ${idx}`} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                          <button
                            onClick={() => handleRemoveSample(idx)}
                            style={{ position: 'absolute', top: 2, right: 2, background: 'rgba(0,0,0,0.7)', border: 'none', borderRadius: '50%', color: '#fff', width: 16, height: 16, display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', fontSize: 10 }}
                          >
                            ✕
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Modal Actions */}
              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
                <button className="btn btn-ghost" onClick={() => { setShowRegisterModal(false); stopCamera(); }}>
                  Cancel
                </button>
                <button
                  className="btn btn-primary"
                  onClick={handleSubmitRegistration}
                  disabled={submitting || capturedSamples.length === 0 || !name.trim() || !empNo.trim()}
                >
                  <Lock size={14} /> {submitting ? 'Registering...' : 'Save & Register Worker'}
                </button>
              </div>

            </div>
          </div>
        )}

      </div>
    </>
  );
}
