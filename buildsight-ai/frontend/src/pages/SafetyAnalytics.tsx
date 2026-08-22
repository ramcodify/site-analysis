import { useState, useEffect, useRef, useCallback } from 'react';
import { Header } from '../components/common/Header';
import { useWebSocket } from '../hooks/useWebSocket';
import { useSearchParams } from 'react-router-dom';
import type { AnalyticsMessage, ConnectionStatus, ViolationResponse } from '../types';
import {
  Shield, Users, TrendingUp, AlertTriangle, MapPin,
  CheckCircle2, RefreshCw, Trash2, Plus,
  Eye, X, ShieldAlert, LayoutGrid, LayoutList, Clock, Zap, User
} from 'lucide-react';
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell,
  PieChart, Pie, LineChart, Line, CartesianGrid, Legend
} from 'recharts';

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const RISK_COLORS: Record<string, string> = {
  safe: '#10b981', low: '#38bdf8', medium: '#f59e0b', high: '#f97316', critical: '#f43f5e',
};

const STATUS_FILTERS = ['ALL', 'OPEN', 'ACKNOWLEDGED', 'RESOLVED'] as const;
const SEVERITY_FILTERS = ['ALL', 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'] as const;

const ZONE_COLORS: Record<string, string> = {
  RESTRICTED: '#ef4444',
  HAZARD: '#f97316',
  EQUIPMENT: '#f59e0b',
  EDGE: '#8b5cf6',
};

type ZoneType = 'RESTRICTED' | 'HAZARD' | 'EQUIPMENT' | 'EDGE';
type DrawMode = 'view' | 'draw';

interface DangerZone {
  id: number;
  name: string;
  zone_type: ZoneType;
  polygon_data: [number, number][];
  risk_weight: number;
  is_active: boolean;
}

const CHART_STYLE = {
  background: 'rgba(15, 23, 42, 0.95)',
  border: '1px solid rgba(255, 255, 255, 0.1)',
  borderRadius: 8,
  fontSize: 12,
};

interface ComplianceTrend {
  time: string;
  compliance: number;
  workers: number;
  violations: number;
}

export default function SafetyAnalytics() {
  const [searchParams, setSearchParams] = useSearchParams();
  const rawTab = searchParams.get('tab');
  const activeTab = (rawTab === 'violations' ? 'violations' : rawTab === 'zones' ? 'zones' : 'overview');

  const [analytics, setAnalytics] = useState<AnalyticsMessage | null>(null);
  const [violations, setViolations] = useState<ViolationResponse[]>([]);
  const [trend, setTrend] = useState<ComplianceTrend[]>([]);
  
  // Violations Hub states
  const [statusFilter, setStatusFilter] = useState<string>('ALL');
  const [severityFilter, setSeverityFilter] = useState<string>('ALL');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedViolation, setSelectedViolation] = useState<ViolationResponse | null>(null);
  const [updatingId, setUpdatingId] = useState<string | number | null>(null);
  const [violationViewMode, setViolationViewMode] = useState<'table' | 'cards'>('table');

  // Danger Zones Studio states
  const [zones, setZones] = useState<DangerZone[]>([]);
  const [drawMode, setDrawMode] = useState<DrawMode>('view');
  const [currentPoints, setCurrentPoints] = useState<[number, number][]>([]);
  const [zoneName, setZoneName] = useState('Restricted Area');
  const [zoneType, setZoneType] = useState<ZoneType>('RESTRICTED');
  const [riskWeight, setRiskWeight] = useState(30);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const fetchViolations = useCallback(async () => {
    try {
      let url = `${API_URL}/api/violations`;
      const params = [];
      if (statusFilter !== 'ALL') params.push(`status=${statusFilter}`);
      if (severityFilter !== 'ALL') params.push(`severity=${severityFilter}`);
      if (params.length) url += '?' + params.join('&');

      const res = await fetch(url);
      if (res.ok) {
        setViolations(await res.json());
      }
    } catch {
      //
    }
  }, [statusFilter, severityFilter]);

  const fetchZones = async () => {
    try {
      const res = await fetch(`${API_URL}/api/danger-zones`);
      if (res.ok) setZones(await res.json());
    } catch {
      //
    }
  };

  const handleMessage = useCallback((data: AnalyticsMessage) => {
    if (data.type === 'analytics_update') {
      setAnalytics(data);
      const now = new Date().toLocaleTimeString('en', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      setTrend(prev => {
        const next = [...prev, {
          time: now,
          compliance: data.safety.ppe_compliance_percentage,
          workers: data.workers.active_count,
          violations: data.safety.active_violations,
        }].slice(-20);
        return next;
      });
    }
  }, []);

  const { status } = useWebSocket({ url: `${WS_URL}/ws/analytics`, onMessage: handleMessage });

  useEffect(() => {
    fetchViolations();
    fetchZones();
    const iv = setInterval(() => {
      fetchViolations();
      fetchZones();
    }, 4000);
    return () => clearInterval(iv);
  }, [fetchViolations]);

  // Canvas Drawing Logic for Danger Zones Studio
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Subtle Architectural Grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let x = 0; x < canvas.width; x += 40) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += 40) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
    }

    // Header Tag
    ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.font = 'bold 12px Inter, sans-serif';
    ctx.fillText('Site Surveillance Camera Geofence Coordinates (16:9)', 20, 26);

    // Existing Zones
    for (const zone of zones) {
      if (zone.polygon_data.length < 3) continue;
      const color = ZONE_COLORS[zone.zone_type] || '#ef4444';
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.fillStyle = color.replace(')', ', 0.16)').replace('rgb', 'rgba').replace('#ef4444', 'rgba(239, 68, 68, 0.16)').replace('#f97316', 'rgba(249, 115, 22, 0.16)').replace('#f59e0b', 'rgba(245, 158, 11, 0.16)').replace('#8b5cf6', 'rgba(139, 92, 246, 0.16)');
      ctx.beginPath();
      ctx.moveTo(zone.polygon_data[0][0], zone.polygon_data[0][1]);
      zone.polygon_data.slice(1).forEach(p => ctx.lineTo(p[0], p[1]));
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // Center Name Badge
      const cx = zone.polygon_data.reduce((s, p) => s + p[0], 0) / zone.polygon_data.length;
      const cy = zone.polygon_data.reduce((s, p) => s + p[1], 0) / zone.polygon_data.length;
      ctx.fillStyle = color;
      ctx.font = 'bold 12px Inter, sans-serif';
      ctx.fillText(`🚨 ${zone.name}`, cx - 30, cy);
    }

    // Active Drawing In-Progress
    if (currentPoints.length > 0) {
      const color = ZONE_COLORS[zoneType] || '#ef4444';
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(currentPoints[0][0], currentPoints[0][1]);
      currentPoints.forEach(p => ctx.lineTo(p[0], p[1]));
      ctx.stroke();
      ctx.setLineDash([]);

      // Points
      currentPoints.forEach((p, i) => {
        ctx.fillStyle = i === 0 ? '#ffffff' : color;
        ctx.beginPath();
        ctx.arc(p[0], p[1], 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      });
    }
  }, [zones, currentPoints, zoneType]);

  useEffect(() => {
    if (activeTab === 'zones') {
      drawCanvas();
    }
  }, [drawCanvas, activeTab]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (drawMode !== 'draw') return;
    const rect = canvasRef.current!.getBoundingClientRect();
    const scaleX = canvasRef.current!.width / rect.width;
    const scaleY = canvasRef.current!.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    setCurrentPoints(prev => [...prev, [x, y]]);
  };

  const handleCanvasDoubleClick = () => {
    if (drawMode !== 'draw' || currentPoints.length < 3) return;
    saveZone();
  };

  const saveZone = async () => {
    if (currentPoints.length < 3) return;
    try {
      const res = await fetch(`${API_URL}/api/danger-zones`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: zoneName,
          zone_type: zoneType,
          polygon_data: currentPoints,
          risk_weight: riskWeight,
        }),
      });
      if (res.ok) {
        setCurrentPoints([]);
        setDrawMode('view');
        await fetchZones();
      }
    } catch {
      //
    }
  };

  const deleteZone = async (id: number) => {
    if (!window.confirm("Delete this danger zone polygon?")) return;
    await fetch(`${API_URL}/api/danger-zones/${id}`, { method: 'DELETE' });
    setZones(prev => prev.filter(z => z.id !== id));
  };

  const handleUpdateStatus = async (id: string | number, newStatus: 'ACKNOWLEDGED' | 'RESOLVED') => {
    setUpdatingId(id);
    try {
      const res = await fetch(`${API_URL}/api/violations/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: newStatus }),
      });
      if (res.ok) {
        await fetchViolations();
        if (selectedViolation && selectedViolation.violation_id === id) {
          setSelectedViolation(prev => prev ? { ...prev, status: newStatus } : null);
        }
      }
    } catch {
      //
    } finally {
      setUpdatingId(null);
    }
  };

  const handleDeleteViolation = async (violationId: string | number) => {
    if (!window.confirm("Are you sure you want to permanently delete this violation and its stored evidence photo from the database?")) {
      return;
    }
    setUpdatingId(violationId);
    try {
      const res = await fetch(`${API_URL}/api/violations/${violationId}`, {
        method: 'DELETE',
      });
      if (res.ok) {
        if (selectedViolation && selectedViolation.violation_id === violationId) {
          setSelectedViolation(null);
        }
        await fetchViolations();
      }
    } catch (err) {
      console.error("Failed to delete violation:", err);
    } finally {
      setUpdatingId(null);
    }
  };

  const handleClearAllViolations = async () => {
    if (!window.confirm("Are you sure you want to delete ALL violations and evidence photos from MongoDB? This action cannot be undone.")) {
      return;
    }
    try {
      const res = await fetch(`${API_URL}/api/violations`, {
        method: 'DELETE',
      });
      if (res.ok) {
        setSelectedViolation(null);
        await fetchViolations();
      }
    } catch (err) {
      console.error("Failed to clear violations:", err);
    }
  };

  const w = analytics?.workers;
  const s = analytics?.safety;

  const riskData = w ? [
    { name: 'Safe',     value: w.risk_distribution.safe,     color: RISK_COLORS.safe },
    { name: 'Low',      value: w.risk_distribution.low,      color: RISK_COLORS.low },
    { name: 'Medium',   value: w.risk_distribution.medium,   color: RISK_COLORS.medium },
    { name: 'High',     value: w.risk_distribution.high,     color: RISK_COLORS.high },
    { name: 'Critical', value: w.risk_distribution.critical, color: RISK_COLORS.critical },
  ].filter(d => d.value > 0) : [];

  const violationTypes: Record<string, number> = {};
  violations.forEach(v => {
    violationTypes[v.violation_type] = (violationTypes[v.violation_type] || 0) + 1;
  });
  const violationChartData = Object.entries(violationTypes).map(([type, count]) => ({
    type: type.replace(/_/g, ' '),
    count,
  })).sort((a, b) => b.count - a.count);

  const openCount = violations.filter(v => v.status === 'OPEN').length;
  const ackCount  = violations.filter(v => v.status === 'ACKNOWLEDGED').length;
  const resCount  = violations.filter(v => v.status === 'RESOLVED').length;
  const critCount = violations.filter(v => v.severity === 'CRITICAL').length;

  const filteredViolations = violations.filter(v => {
    if (searchTerm) {
      const q = searchTerm.toLowerCase();
      const matchWorker = String(v.worker_id).includes(q) || (v.worker_code && v.worker_code.toLowerCase().includes(q)) || (v.worker_name && v.worker_name.toLowerCase().includes(q));
      const matchType = v.violation_type?.toLowerCase().includes(q);
      const matchDesc = v.description?.toLowerCase().includes(q);
      if (!matchWorker && !matchType && !matchDesc) return false;
    }
    return true;
  });

  return (
    <>
      <Header
        title="Safety Analytics"
        subtitle="Real-time PPE compliance, violations incident management & danger zone spatial analytics"
        connectionStatus={status as ConnectionStatus}
        processingActive={!!analytics}
      />
      <div className="app-content" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
        
        {/* Navigation Tabs Bar */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
          <div style={{
            display: 'flex',
            gap: 6,
            background: 'var(--bg-card)',
            padding: 4,
            borderRadius: 10,
            border: '1px solid var(--border-primary)',
            boxShadow: 'var(--shadow-card)'
          }}>
            <button
              className={`btn ${activeTab === 'overview' ? 'btn-primary' : 'btn-ghost'}`}
              onClick={() => setSearchParams({ tab: 'overview' })}
              style={{ fontSize: 13, padding: '7px 16px' }}
            >
              <Shield size={15} /> Safety Overview
            </button>
            <button
              className={`btn ${activeTab === 'violations' ? 'btn-primary' : 'btn-ghost'}`}
              onClick={() => setSearchParams({ tab: 'violations' })}
              style={{ fontSize: 13, padding: '7px 16px' }}
            >
              <AlertTriangle size={15} /> Violations Hub ({openCount} Open)
            </button>
            <button
              className={`btn ${activeTab === 'zones' ? 'btn-primary' : 'btn-ghost'}`}
              onClick={() => setSearchParams({ tab: 'zones' })}
              style={{ fontSize: 13, padding: '7px 16px' }}
            >
              <MapPin size={15} /> Danger Zones Studio ({zones.length} Zones)
            </button>
          </div>

          <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
            {activeTab === 'violations' && violations.length > 0 && (
              <button
                className="btn btn-danger"
                onClick={handleClearAllViolations}
                style={{ fontSize: 12, padding: '6px 12px' }}
                title="Clear all violations from database"
              >
                <Trash2 size={14} /> Clear All Violations
              </button>
            )}
            <button
              className="btn btn-ghost"
              onClick={() => { fetchViolations(); fetchZones(); }}
              style={{ fontSize: 12, padding: '6px 12px' }}
            >
              <RefreshCw size={14} /> Refresh Data
            </button>
          </div>
        </div>

        {/* ========================================================================= */}
        {/* TAB 1: SAFETY OVERVIEW */}
        {/* ========================================================================= */}
        {activeTab === 'overview' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            {/* KPIs */}
            <div className="kpi-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
              <div className="kpi-card green">
                <div className="kpi-label">Live PPE Compliance</div>
                <div className="kpi-value">{s?.ppe_compliance_percentage?.toFixed(0) ?? '100'}<span className="kpi-unit">%</span></div>
                <div className="kpi-sub">Evaluated across {w?.active_count ?? 0} active workers</div>
              </div>
              <div className="kpi-card amber" onClick={() => setSearchParams({ tab: 'violations' })} style={{ cursor: 'pointer' }}>
                <div className="kpi-label">Open Violations</div>
                <div className="kpi-value">{openCount}</div>
                <div className="kpi-sub">{ackCount} acknowledged · {resCount} resolved</div>
              </div>
              <div className="kpi-card red">
                <div className="kpi-label">Critical Safety Alerts</div>
                <div className="kpi-value">{(w?.risk_distribution.high ?? 0) + (w?.risk_distribution.critical ?? 0)}</div>
                <div className="kpi-sub">{w?.risk_distribution.critical ?? 0} high-risk breaches</div>
              </div>
              <div className="kpi-card cyan">
                <div className="kpi-label">Active Tracked Personnel</div>
                <div className="kpi-value">{w?.active_count ?? 0}</div>
                <div className="kpi-sub">Continuous ByteTrack + YOLO11 inference</div>
              </div>
            </div>

            {/* What is a Danger Zone? Educational & Status Card */}
            <div className="card" style={{
              background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.85), rgba(15, 23, 42, 0.95))',
              borderColor: 'rgba(56, 189, 248, 0.3)',
              boxShadow: '0 0 20px rgba(56, 189, 248, 0.08)'
            }}>
              <div className="card-header" style={{ marginBottom: 12 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <div style={{
                    width: 32,
                    height: 32,
                    borderRadius: 8,
                    background: 'rgba(56, 189, 248, 0.15)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: '#38bdf8'
                  }}>
                    <MapPin size={18} />
                  </div>
                  <div>
                    <h3 style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-primary)' }}>
                      What is a Danger Zone?
                    </h3>
                    <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                      Geofenced Polygonal Hazard Boundaries & Automated Intrusion Prevention
                    </p>
                  </div>
                </div>
                <button
                  className="btn btn-primary"
                  onClick={() => setSearchParams({ tab: 'zones' })}
                  style={{ fontSize: 12, padding: '6px 14px' }}
                >
                  <MapPin size={14} /> Open Danger Zones Studio
                </button>
              </div>

              <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: 16 }}>
                A <strong>Danger Zone</strong> in BuildSight AI is an operator-configured <strong>virtual geofence polygon</strong> mapped directly on top of site camera feeds. When a worker's tracked bounding box intersects with any restricted hazard perimeter, our <strong>Point-in-Polygon (Ray-Casting)</strong> engine instantly triggers high-severity alerts and logs a <code>DANGER_ZONE_BREACH</code> incident.
              </p>

              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
                gap: 12
              }}>
                <div style={{
                  padding: 12,
                  background: 'rgba(239, 68, 68, 0.10)',
                  borderRadius: 10,
                  border: '1px solid rgba(239, 68, 68, 0.25)'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
                    <span style={{ fontSize: 16 }}>🚧</span>
                    <strong style={{ fontSize: 12.5, color: '#f87171' }}>1. Excavation & Trench Pits</strong>
                  </div>
                  <p style={{ fontSize: 11.5, color: 'var(--text-muted)', lineHeight: 1.4 }}>
                    Unshored trenches and collapse-risk pits. Triggers alerts if workers enter without tie-off harnesses.
                  </p>
                </div>

                <div style={{
                  padding: 12,
                  background: 'rgba(249, 115, 22, 0.10)',
                  borderRadius: 10,
                  border: '1px solid rgba(249, 115, 22, 0.25)'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
                    <span style={{ fontSize: 16 }}>🏗️</span>
                    <strong style={{ fontSize: 12.5, color: '#fb923c' }}>2. Crane & Machinery Swing</strong>
                  </div>
                  <p style={{ fontSize: 11.5, color: 'var(--text-muted)', lineHeight: 1.4 }}>
                    Blind-spot perimeter around active excavators and suspended crane loads. Prevents crush injuries.
                  </p>
                </div>

                <div style={{
                  padding: 12,
                  background: 'rgba(245, 158, 11, 0.10)',
                  borderRadius: 10,
                  border: '1px solid rgba(245, 158, 11, 0.25)'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
                    <span style={{ fontSize: 16 }}>⚡</span>
                    <strong style={{ fontSize: 12.5, color: '#fbbf24' }}>3. High-Voltage Enclosures</strong>
                  </div>
                  <p style={{ fontSize: 11.5, color: 'var(--text-muted)', lineHeight: 1.4 }}>
                    Electrical transformers and arc-flash zones. Strictly permits only certified personnel.
                  </p>
                </div>

                <div style={{
                  padding: 12,
                  background: 'rgba(168, 85, 247, 0.10)',
                  borderRadius: 10,
                  border: '1px solid rgba(168, 85, 247, 0.25)'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
                    <span style={{ fontSize: 16 }}>🧱</span>
                    <strong style={{ fontSize: 12.5, color: '#c084fc' }}>4. Leading Edges & Voids</strong>
                  </div>
                  <p style={{ fontSize: 11.5, color: 'var(--text-muted)', lineHeight: 1.4 }}>
                    Unprotected floor openings and roof perimeters requiring 100% continuous tie-off fall protection.
                  </p>
                </div>
              </div>
            </div>

            {/* Charts Row */}
            <div className="chart-grid" style={{ gridTemplateColumns: '1.2fr 1fr 1fr' }}>
              {/* PPE Compliance Trend */}
              <div className="chart-card">
                <h3>PPE Compliance Trend (Live Stream)</h3>
                {trend.length > 1 ? (
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={trend}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="time" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} interval="preserveStartEnd" />
                      <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                      <Tooltip contentStyle={CHART_STYLE} />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      <Line type="monotone" dataKey="compliance" name="PPE Score %" stroke="#10b981" strokeWidth={2.5} dot={false} />
                      <Line type="monotone" dataKey="violations" name="Active Viols" stroke="#f43f5e" strokeWidth={1.8} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="empty-state" style={{ padding: 30 }}>
                    <TrendingUp size={32} style={{ opacity: 0.3 }} />
                    <p>Collecting data — start webcam or video stream</p>
                  </div>
                )}
              </div>

              {/* Risk Distribution Donut */}
              <div className="chart-card">
                <h3>Worker Risk Distribution</h3>
                {riskData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={220}>
                    <PieChart>
                      <Pie data={riskData} cx="50%" cy="50%" innerRadius={55} outerRadius={85}
                        paddingAngle={3} dataKey="value" nameKey="name">
                        {riskData.map((entry, i) => <Cell key={i} fill={entry.color} />)}
                      </Pie>
                      <Tooltip contentStyle={CHART_STYLE} />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                    </PieChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="empty-state" style={{ padding: 30 }}>
                    <Users size={32} style={{ opacity: 0.3 }} />
                    <p>No active workers in frame</p>
                  </div>
                )}
              </div>

              {/* Violations by Type Bar Chart */}
              <div className="chart-card">
                <h3>Violations Breakdown</h3>
                {violationChartData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={violationChartData} layout="vertical">
                      <XAxis type="number" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                      <YAxis dataKey="type" type="category" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} width={110} />
                      <Tooltip contentStyle={CHART_STYLE} />
                      <Bar dataKey="count" fill="#f59e0b" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="empty-state" style={{ padding: 30 }}>
                    <Shield size={32} style={{ opacity: 0.3 }} />
                    <p>No safety violations recorded</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 2: FULL VIOLATIONS INCIDENT HUB */}
        {/* ========================================================================= */}
        {activeTab === 'violations' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            {/* KPI Summary */}
            <div className="kpi-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
              <div className="kpi-card amber">
                <div className="kpi-label">Total Violations</div>
                <div className="kpi-value">{violations.length}</div>
                <div className="kpi-sub">Lifetime incidents recorded</div>
              </div>
              <div className="kpi-card red">
                <div className="kpi-label">Open Incidents</div>
                <div className="kpi-value">{openCount}</div>
                <div className="kpi-sub">Requiring supervisor attention</div>
              </div>
              <div className="kpi-card red">
                <div className="kpi-label">Critical Incidents</div>
                <div className="kpi-value">{critCount}</div>
                <div className="kpi-sub">Highest danger severity</div>
              </div>
              <div className="kpi-card green">
                <div className="kpi-label">Resolved Incidents</div>
                <div className="kpi-value">{resCount}</div>
                <div className="kpi-sub">{ackCount} acknowledged</div>
              </div>
            </div>

            {/* Filter Toolbar */}
            <div className="controls-bar" style={{ justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
                {/* Search */}
                <input
                  type="text"
                  placeholder="Search by worker ID, name, or violation type..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="control-select"
                  style={{ minWidth: 260 }}
                />

                {/* Status Filter */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Status:</span>
                  <div style={{ display: 'flex', gap: 4, background: 'rgba(255, 255, 255, 0.04)', padding: 3, borderRadius: 8 }}>
                    {STATUS_FILTERS.map(f => (
                      <button
                        key={f}
                        onClick={() => setStatusFilter(f)}
                        className={`btn ${statusFilter === f ? 'btn-primary' : 'btn-ghost'}`}
                        style={{ fontSize: 11, padding: '4px 10px', height: 26 }}
                      >
                        {f}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Severity Filter */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Severity:</span>
                  <div style={{ display: 'flex', gap: 4, background: 'rgba(255, 255, 255, 0.04)', padding: 3, borderRadius: 8 }}>
                    {SEVERITY_FILTERS.map(sev => (
                      <button
                        key={sev}
                        onClick={() => setSeverityFilter(sev)}
                        className={`btn ${severityFilter === sev ? 'btn-primary' : 'btn-ghost'}`}
                        style={{ fontSize: 11, padding: '4px 10px', height: 26 }}
                      >
                        {sev}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                  Showing <strong>{filteredViolations.length}</strong> of {violations.length} incidents
                </div>
                {/* View Mode Toggle */}
                <div style={{ display: 'flex', background: 'rgba(255,255,255,0.04)', borderRadius: 8, padding: 3, border: '1px solid var(--border-primary)', gap: 2 }}>
                  <button
                    className={`btn ${violationViewMode === 'table' ? 'btn-primary' : 'btn-ghost'}`}
                    style={{ padding: '3px 9px', fontSize: 11, height: 26 }}
                    onClick={() => setViolationViewMode('table')}
                    title="Table View"
                  >
                    <LayoutList size={12} />
                  </button>
                  <button
                    className={`btn ${violationViewMode === 'cards' ? 'btn-primary' : 'btn-ghost'}`}
                    style={{ padding: '3px 9px', fontSize: 11, height: 26 }}
                    onClick={() => setViolationViewMode('cards')}
                    title="Card Grid View"
                  >
                    <LayoutGrid size={12} />
                  </button>
                </div>
              </div>
            </div>

            {/* Violations: Card or Table View */}
            {filteredViolations.length === 0 ? (
              <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
                <div className="empty-state" style={{ padding: 60 }}>
                  <CheckCircle2 size={48} style={{ color: 'var(--accent-green)', opacity: 0.6, marginBottom: 16 }} />
                  <h3 style={{ fontSize: 16, color: 'var(--text-primary)' }}>No Matching Safety Violations</h3>
                  <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>
                    No violation records match the selected status, severity, or search criteria.
                  </p>
                </div>
              </div>
            ) : violationViewMode === 'cards' ? (
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(290px, 1fr))',
                gap: 14,
              }}>
                {filteredViolations.map((v) => {
                  const sev = v.severity?.toLowerCase() || 'medium';
                  const sevColorMap: Record<string, string> = {
                    critical: '#ef4444', high: '#f97316', medium: '#f59e0b', low: '#3b82f6',
                  };
                  const sevColor = sevColorMap[sev] || '#6b7280';
                  const isRegistered = Boolean(v.worker_code || (v.worker_id && v.worker_id >= 100));
                  const isActive = updatingId === v.violation_id;
                  
                  const imgSrc = v.snapshot_base64 || (v.evidence_url ? (v.evidence_url.startsWith('http') ? v.evidence_url : `${API_URL}${v.evidence_url}`) : (v.evidence_path ? (v.evidence_path.startsWith('http') ? v.evidence_path : `${API_URL}${v.evidence_path}`) : null));

                  return (
                    <div
                      key={v.violation_id}
                      onClick={() => setSelectedViolation(v)}
                      style={{
                        background: 'var(--bg-card, #111827)',
                        border: `1.5px solid ${selectedViolation?.violation_id === v.violation_id ? sevColor : 'rgba(255,255,255,0.08)'}`,
                        borderRadius: 14,
                        overflow: 'hidden',
                        cursor: 'pointer',
                        transition: 'transform 0.15s, box-shadow 0.15s',
                        boxShadow: selectedViolation?.violation_id === v.violation_id ? `0 0 0 2px ${sevColor}40, 0 8px 24px rgba(0,0,0,0.4)` : '0 4px 12px rgba(0,0,0,0.22)',
                      }}
                      onMouseEnter={e => {
                        (e.currentTarget as HTMLDivElement).style.transform = 'translateY(-2px)';
                        (e.currentTarget as HTMLDivElement).style.boxShadow = `0 8px 24px ${sevColor}25, 0 4px 16px rgba(0,0,0,0.4)`;
                      }}
                      onMouseLeave={e => {
                        (e.currentTarget as HTMLDivElement).style.transform = 'translateY(0)';
                        (e.currentTarget as HTMLDivElement).style.boxShadow = selectedViolation?.violation_id === v.violation_id ? `0 0 0 2px ${sevColor}40, 0 8px 24px rgba(0,0,0,0.4)` : '0 4px 12px rgba(0,0,0,0.22)';
                      }}
                    >
                      {/* Photo / Banner */}
                      <div style={{ position: 'relative', height: 135, background: '#0a0e1a', overflow: 'hidden' }}>
                        {imgSrc ? (
                          <img
                            src={imgSrc}
                            alt="Evidence"
                            style={{ width: '100%', height: '100%', objectFit: 'cover', opacity: 0.9 }}
                            onError={(e) => {
                              (e.target as HTMLElement).style.display = 'none';
                            }}
                          />
                        ) : (
                          <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', background: 'radial-gradient(circle at center, rgba(239,68,68,0.15), rgba(10,14,26,0.9))' }}>
                            <AlertTriangle size={36} style={{ color: sevColor, opacity: 0.8, marginBottom: 4 }} />
                            <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                              {v.worker_code ? `🆔 ${v.worker_code}` : `Track #${v.worker_id ?? '?'}`}
                            </span>
                          </div>
                        )}
                        {/* Severity tag */}
                        <div style={{
                          position: 'absolute', top: 8, left: 8,
                          background: `${sevColor}`, color: '#fff',
                          borderRadius: 6, padding: '3px 9px',
                          fontSize: 10, fontWeight: 800, letterSpacing: 0.5,
                          boxShadow: `0 2px 8px ${sevColor}60`
                        }}>
                          {v.severity}
                        </div>
                        {/* Status tag */}
                        <div style={{
                          position: 'absolute', top: 8, right: 8,
                          background: v.status === 'OPEN' ? 'rgba(239,68,68,0.92)' : v.status === 'ACKNOWLEDGED' ? 'rgba(245,158,11,0.92)' : 'rgba(16,185,129,0.92)',
                          color: '#fff', borderRadius: 6, padding: '3px 8px', fontSize: 10, fontWeight: 700,
                          backdropFilter: 'blur(4px)',
                        }}>
                          {v.status}
                        </div>
                      </div>

                      {/* Body */}
                      <div style={{ padding: '12px 14px' }}>
                        <div style={{ fontSize: 13, fontWeight: 800, color: 'var(--text-primary)', marginBottom: 5, lineHeight: 1.3, letterSpacing: -0.2 }}>
                          {v.violation_type?.replace(/_/g, ' ') || 'Safety Non-Compliance'}
                        </div>

                        {/* Worker Identity */}
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
                          <User size={12} style={{ color: isRegistered ? '#38bdf8' : 'var(--text-muted)', flexShrink: 0 }} />
                          <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 800, fontSize: 11.5, color: isRegistered ? '#38bdf8' : 'var(--text-secondary)' }}>
                            {v.worker_code ? `🆔 ${v.worker_code}` : `Track #${v.worker_id ?? '?'}`}
                          </span>
                          {v.worker_name && (
                            <span style={{ fontSize: 11, color: 'var(--text-primary)', fontWeight: 600 }}>({v.worker_name})</span>
                          )}
                        </div>

                        {/* Enhanced Missing PPE symbols & chips */}
                        {v.missing_items && v.missing_items.length > 0 ? (
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 8 }}>
                            {v.missing_items.map((item, idx) => {
                              const s = item.toLowerCase();
                              let icon = '❌';
                              let label = item.replace(/_/g, ' ');
                              let bg = 'rgba(239,68,68,0.14)';
                              let border = 'rgba(239,68,68,0.3)';
                              let color = '#f87171';

                              if (s.includes('helmet') || s.includes('hardhat')) {
                                icon = '🪖';
                                label = 'Hardhat Missing';
                              } else if (s.includes('vest')) {
                                icon = '🦺';
                                label = 'Vest Missing';
                                bg = 'rgba(249,115,22,0.14)';
                                border = 'rgba(249,115,22,0.3)';
                                color = '#fb923c';
                              } else if (s.includes('glove')) {
                                icon = '🧤';
                                label = 'Gloves Missing';
                                bg = 'rgba(234,179,8,0.14)';
                                border = 'rgba(234,179,8,0.3)';
                                color = '#facc15';
                              } else if (s.includes('mask')) {
                                icon = '😷';
                                label = 'Mask Missing';
                                bg = 'rgba(56,189,248,0.14)';
                                border = 'rgba(56,189,248,0.3)';
                                color = '#38bdf8';
                              }

                              return (
                                <span key={idx} style={{
                                  display: 'inline-flex', alignItems: 'center', gap: 4,
                                  fontSize: 10.5, fontWeight: 700,
                                  background: bg, color: color,
                                  borderRadius: 6, padding: '2px 7px', border: `1px solid ${border}`,
                                }}>
                                  <span>{icon}</span> {label}
                                </span>
                              );
                            })}
                          </div>
                        ) : (
                          <div style={{ marginBottom: 8 }}>
                            <span style={{ fontSize: 10.5, color: '#f87171', fontWeight: 600, background: 'rgba(239,68,68,0.12)', padding: '2px 6px', borderRadius: 4 }}>
                              ❌ Critical PPE Gap Detected
                            </span>
                          </div>
                        )}

                        {/* Meta metrics */}
                        <div style={{ display: 'flex', gap: 12, fontSize: 11, color: 'var(--text-muted)', marginBottom: 10, borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: 6 }}>
                          <span style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                            <Zap size={10} style={{ color: '#f59e0b' }} /> {v.risk_score?.toFixed(0) ?? '—'} Risk
                          </span>
                          <span style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                            <Clock size={10} /> {v.duration_seconds?.toFixed(0) ?? 0}s
                          </span>
                          <span style={{ display: 'flex', alignItems: 'center', gap: 3, marginLeft: 'auto' }}>
                            🕐 {v.timestamp ? new Date(v.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '—'}
                          </span>
                        </div>

                        {/* Actions Row */}
                        <div style={{ display: 'flex', gap: 6 }} onClick={e => e.stopPropagation()}>
                          <button
                            className="btn btn-ghost"
                            onClick={() => setSelectedViolation(v)}
                            style={{ fontSize: 11, padding: '4px 10px' }}
                          >
                            <Eye size={11} /> View
                          </button>
                          {v.status === 'OPEN' && (
                            <button
                              className="btn btn-ghost"
                              onClick={() => handleUpdateStatus(v.violation_id, 'ACKNOWLEDGED')}
                              disabled={isActive}
                              style={{ fontSize: 11, padding: '4px 10px', borderColor: 'rgba(245,158,11,0.4)', color: '#fbbf24' }}
                            >
                              Ack
                            </button>
                          )}
                          {v.status !== 'RESOLVED' && (
                            <button
                              className="btn btn-ghost"
                              onClick={() => handleUpdateStatus(v.violation_id, 'RESOLVED')}
                              disabled={isActive}
                              style={{ fontSize: 11, padding: '4px 10px', borderColor: 'rgba(16,185,129,0.4)', color: '#34d399' }}
                            >
                              Resolve
                            </button>
                          )}
                          <button
                            className="btn btn-ghost"
                            onClick={() => handleDeleteViolation(v.violation_id)}
                            disabled={isActive}
                            style={{ fontSize: 11, padding: '4px 10px', borderColor: 'rgba(239,68,68,0.3)', color: '#f87171', marginLeft: 'auto' }}
                          >
                            <Trash2 size={11} />
                          </button>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              /* ── TABLE VIEW ── */
              <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
                <div style={{ overflowX: 'auto' }}>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Worker ID</th>
                        <th>Violation Type</th>
                        <th>Missing PPE / Description</th>
                        <th>Severity</th>
                        <th>Risk Score</th>
                        <th>Status</th>
                        <th>Time</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredViolations.map((v) => {
                        const sev = v.severity?.toLowerCase() || 'medium';
                        const isRegistered = v.worker_code || (v.worker_id && v.worker_id >= 100);

                        return (
                          <tr key={v.violation_id} onClick={() => setSelectedViolation(v)}>
                            <td>
                              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                <span style={{
                                  fontFamily: 'var(--font-mono)',
                                  fontWeight: 700,
                                  color: isRegistered ? '#38bdf8' : 'var(--text-secondary)'
                                }}>
                                  {v.worker_code ? `🆔 ${v.worker_code}` : `#${v.worker_id ?? '?'}`}
                                </span>
                                {v.worker_name && (
                                  <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                                    ({v.worker_name})
                                  </span>
                                )}
                              </div>
                            </td>
                            <td>
                              <strong style={{ color: 'var(--text-primary)' }}>
                                {v.violation_type?.replace(/_/g, ' ')}
                              </strong>
                            </td>
                            <td>
                              <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                                {v.missing_items && v.missing_items.length > 0 ? (
                                  v.missing_items.map((item, idx) => (
                                    <span key={idx} style={{
                                      fontSize: 10.5, padding: '2px 6px', borderRadius: 4,
                                      background: 'rgba(239,68,68,0.15)', color: '#f87171',
                                      border: '1px solid rgba(239,68,68,0.3)', fontWeight: 600,
                                    }}>
                                      ✗ {item}
                                    </span>
                                  ))
                                ) : (
                                  <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                                    {(v as any).zone_name ? `Zone: ${(v as any).zone_name}` : (v.description || 'Safety Non-Compliance')}
                                  </span>
                                )}
                              </div>
                            </td>
                            <td>
                              <span className={`badge ${sev}`}>{v.severity}</span>
                            </td>
                            <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                              {v.risk_score?.toFixed(0)}
                            </td>
                            <td>
                              <span style={{
                                padding: '3px 8px', borderRadius: 6, fontSize: 11, fontWeight: 700,
                                background: v.status === 'OPEN' ? 'rgba(239,68,68,0.15)' : v.status === 'ACKNOWLEDGED' ? 'rgba(245,158,11,0.15)' : 'rgba(16,185,129,0.15)',
                                color: v.status === 'OPEN' ? '#f87171' : v.status === 'ACKNOWLEDGED' ? '#fbbf24' : '#34d399',
                                border: `1px solid ${v.status === 'OPEN' ? 'rgba(239,68,68,0.3)' : v.status === 'ACKNOWLEDGED' ? 'rgba(245,158,11,0.3)' : 'rgba(16,185,129,0.3)'}`,
                              }}>
                                {v.status}
                              </span>
                            </td>
                            <td style={{ fontSize: 11, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                              {v.timestamp ? new Date(v.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '—'}
                            </td>
                            <td onClick={(e) => e.stopPropagation()}>
                              <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                                <button
                                  className="btn btn-ghost"
                                  onClick={() => setSelectedViolation(v)}
                                  style={{ fontSize: 11, padding: '3px 8px' }}
                                  title="View evidence photo & incident details"
                                >
                                  <Eye size={12} /> View
                                </button>
                                {v.status === 'OPEN' && (
                                  <button
                                    className="btn btn-ghost"
                                    onClick={() => handleUpdateStatus(v.violation_id, 'ACKNOWLEDGED')}
                                    disabled={updatingId === v.violation_id}
                                    style={{ fontSize: 11, padding: '3px 8px', borderColor: 'rgba(245,158,11,0.4)', color: '#fbbf24' }}
                                    title="Acknowledge violation"
                                  >
                                    Ack
                                  </button>
                                )}
                                {v.status !== 'RESOLVED' && (
                                  <button
                                    className="btn btn-ghost"
                                    onClick={() => handleUpdateStatus(v.violation_id, 'RESOLVED')}
                                    disabled={updatingId === v.violation_id}
                                    style={{ fontSize: 11, padding: '3px 8px', borderColor: 'rgba(16,185,129,0.4)', color: '#34d399' }}
                                    title="Resolve violation"
                                  >
                                    Resolve
                                  </button>
                                )}
                                <button
                                  className="btn btn-ghost"
                                  onClick={() => handleDeleteViolation(v.violation_id)}
                                  disabled={updatingId === v.violation_id}
                                  style={{ fontSize: 11, padding: '3px 8px', borderColor: 'rgba(239,68,68,0.3)', color: '#f87171' }}
                                  title="Permanently delete violation record"
                                >
                                  <Trash2 size={12} />
                                </button>
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 3: DANGER ZONES STUDIO */}
        {/* ========================================================================= */}
        {activeTab === 'zones' && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: 20, alignItems: 'start' }}>
            {/* Left: Zone Canvas */}
            <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
              <div style={{
                padding: '14px 18px',
                borderBottom: '1px solid var(--border-primary)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                flexWrap: 'wrap',
                gap: 12
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <ShieldAlert size={18} style={{ color: '#f59e0b' }} />
                  <span className="card-title" style={{ fontSize: 14 }}>Site Camera Geofence Drawing Canvas</span>
                </div>

                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  {drawMode === 'view' ? (
                    <button className="btn btn-primary" onClick={() => setDrawMode('draw')}>
                      <Plus size={14} /> Draw New Danger Zone
                    </button>
                  ) : (
                    <>
                      <button
                        className="btn btn-success"
                        onClick={saveZone}
                        disabled={currentPoints.length < 3}
                      >
                        ✓ Save Zone ({currentPoints.length} Pts)
                      </button>
                      <button
                        className="btn btn-ghost"
                        onClick={() => { setDrawMode('view'); setCurrentPoints([]); }}
                      >
                        Cancel
                      </button>
                      <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                        {currentPoints.length < 3 ? `Click to add points (${currentPoints.length}/3 min)` : 'Double-click to finish'}
                      </span>
                    </>
                  )}
                </div>
              </div>

              <div ref={containerRef} style={{ position: 'relative' }}>
                <canvas
                  ref={canvasRef}
                  width={840}
                  height={472}
                  onClick={handleCanvasClick}
                  onDoubleClick={handleCanvasDoubleClick}
                  style={{
                    width: '100%',
                    display: 'block',
                    cursor: drawMode === 'draw' ? 'crosshair' : 'default',
                    background: '#040711',
                  }}
                />
              </div>
            </div>

            {/* Right: Zone Configuration & Zone Roster */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              {/* Zone Config Form */}
              <div className="card">
                <div className="card-header">
                  <span className="card-title">Zone Settings</span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                  <div>
                    <label style={{ fontSize: 12, color: 'var(--text-muted)', display: 'block', marginBottom: 6 }}>
                      Zone Name
                    </label>
                    <input
                      type="text"
                      value={zoneName}
                      onChange={e => setZoneName(e.target.value)}
                      className="control-select"
                      style={{ width: '100%' }}
                      placeholder="e.g. Crane Swing Zone, Excavation Pit"
                    />
                  </div>

                  <div>
                    <label style={{ fontSize: 12, color: 'var(--text-muted)', display: 'block', marginBottom: 6 }}>
                      Hazard Category
                    </label>
                    <select
                      value={zoneType}
                      onChange={e => setZoneType(e.target.value as ZoneType)}
                      className="control-select"
                      style={{ width: '100%' }}
                    >
                      <option value="RESTRICTED">🔴 Restricted Area</option>
                      <option value="HAZARD">🟠 Hazardous Area</option>
                      <option value="EQUIPMENT">🟡 Heavy Equipment Radius</option>
                      <option value="EDGE">🟣 Leading Edge / Fall Hazard</option>
                    </select>
                  </div>

                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: 'var(--text-muted)', marginBottom: 6 }}>
                      <span>Risk Score Weight:</span>
                      <strong style={{ color: 'var(--accent-blue)' }}>+{riskWeight} pts</strong>
                    </div>
                    <input
                      type="range"
                      min="10"
                      max="100"
                      value={riskWeight}
                      onChange={e => setRiskWeight(Number(e.target.value))}
                      style={{ width: '100%' }}
                    />
                  </div>

                  <div style={{ fontSize: 11.5, color: 'var(--text-muted)', lineHeight: 1.5, background: 'rgba(255, 255, 255, 0.03)', padding: 10, borderRadius: 8 }}>
                    ℹ️ Click on the canvas to place at least 3 polygonal points. Double click or press Save to commit to MongoDB.
                  </div>
                </div>
              </div>

              {/* Active Zones List */}
              <div className="card">
                <div className="card-header">
                  <span className="card-title">Configured Zones ({zones.length})</span>
                </div>
                {zones.length === 0 ? (
                  <div style={{ textAlign: 'center', padding: 20, fontSize: 13, color: 'var(--text-muted)' }}>
                    No zones defined yet. Click "Draw New Danger Zone" to create one.
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 300, overflowY: 'auto' }}>
                    {zones.map(zone => {
                      const color = ZONE_COLORS[zone.zone_type] || '#ef4444';
                      return (
                        <div
                          key={zone.id}
                          style={{
                            padding: '10px 12px',
                            background: 'rgba(255, 255, 255, 0.03)',
                            borderRadius: 8,
                            border: `1px solid ${color}40`,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between'
                          }}
                        >
                          <div>
                            <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>
                              {zone.name}
                            </div>
                            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
                              <span style={{ color }}>{zone.zone_type}</span> · +{zone.risk_weight} pts · {zone.polygon_data?.length ?? 4} pts
                            </div>
                          </div>
                          <button
                            className="btn btn-ghost"
                            onClick={() => deleteZone(zone.id)}
                            style={{ padding: '4px 8px', color: '#f87171' }}
                            title="Delete danger zone"
                          >
                            <Trash2 size={13} />
                          </button>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Violation Evidence Modal */}
        {selectedViolation && (
          <div
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: 'rgba(0, 0, 0, 0.80)',
              backdropFilter: 'blur(8px)',
              zIndex: 1000,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: 20
            }}
            onClick={() => setSelectedViolation(null)}
          >
            <div
              className="card"
              style={{
                maxWidth: 580,
                width: '100%',
                maxHeight: '90vh',
                overflowY: 'auto',
                padding: 24,
                position: 'relative'
              }}
              onClick={(e) => e.stopPropagation()}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 }}>
                <div>
                  <h3 style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-primary)' }}>
                    Violation Evidence Snapshot
                  </h3>
                  <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                    Incident #{selectedViolation.violation_id || selectedViolation.id}
                  </p>
                </div>
                <button
                  className="btn btn-ghost"
                  onClick={() => setSelectedViolation(null)}
                  style={{ padding: 4 }}
                >
                  <X size={18} />
                </button>
              </div>

              {/* Evidence Photo */}
              <div style={{
                width: '100%',
                height: 260,
                background: '#000',
                borderRadius: 12,
                overflow: 'hidden',
                marginBottom: 16,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                border: '1px solid var(--border-primary)'
              }}>
                {selectedViolation.evidence_url || selectedViolation.evidence_path ? (
                  <img
                    src={selectedViolation.evidence_url ? `${API_URL}${selectedViolation.evidence_url}` : `${API_URL}/data/evidence/${selectedViolation.evidence_path?.split('/').pop()}`}
                    alt="Evidence Snapshot"
                    style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                    onError={(e) => {
                      (e.target as HTMLElement).style.display = 'none';
                    }}
                  />
                ) : (
                  <div style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
                    <Eye size={36} style={{ opacity: 0.3, marginBottom: 8 }} />
                    <p style={{ fontSize: 12 }}>No snapshot image attached</p>
                  </div>
                )}
              </div>

              {/* Incident Details Grid */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 20 }}>
                <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: 10, borderRadius: 8, border: '1px solid var(--border-primary)' }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Worker</div>
                  <div style={{ fontSize: 14, fontWeight: 700, color: '#38bdf8', marginTop: 2 }}>
                    {selectedViolation.worker_code ? `🆔 ${selectedViolation.worker_code}` : `#${selectedViolation.worker_id ?? '?'}`}
                    {selectedViolation.worker_name && ` · ${selectedViolation.worker_name}`}
                  </div>
                </div>

                <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: 10, borderRadius: 8, border: '1px solid var(--border-primary)' }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Severity & Risk</div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 4 }}>
                    <span className={`badge ${selectedViolation.severity?.toLowerCase()}`}>{selectedViolation.severity}</span>
                    <span style={{ fontSize: 12, fontWeight: 600 }}>Score: {selectedViolation.risk_score?.toFixed(0)}</span>
                  </div>
                </div>

                <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: 10, borderRadius: 8, border: '1px solid var(--border-primary)' }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Violation Type</div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', marginTop: 2 }}>
                    {selectedViolation.violation_type?.replace(/_/g, ' ')}
                  </div>
                </div>

                <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: 10, borderRadius: 8, border: '1px solid var(--border-primary)' }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Logged Timestamp</div>
                  <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 2 }}>
                    {selectedViolation.timestamp ? new Date(selectedViolation.timestamp).toLocaleString() : '—'}
                  </div>
                </div>
              </div>

              {/* Action Buttons in Modal */}
              <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end' }}>
                {selectedViolation.status === 'OPEN' && (
                  <button
                    className="btn btn-ghost"
                    onClick={() => handleUpdateStatus(selectedViolation.violation_id, 'ACKNOWLEDGED')}
                    style={{ borderColor: 'rgba(245, 158, 11, 0.4)', color: '#fbbf24' }}
                  >
                    Acknowledge
                  </button>
                )}
                {selectedViolation.status !== 'RESOLVED' && (
                  <button
                    className="btn btn-success"
                    onClick={() => handleUpdateStatus(selectedViolation.violation_id, 'RESOLVED')}
                  >
                    Mark Resolved
                  </button>
                )}
                <button
                  className="btn btn-danger"
                  onClick={() => handleDeleteViolation(selectedViolation.violation_id)}
                >
                  <Trash2 size={14} /> Delete
                </button>
              </div>
            </div>
          </div>
        )}

      </div>
    </>
  );
}
