import { useState, useEffect } from 'react';
import { Header } from '../components/common/Header';
import { useNavigate } from 'react-router-dom';
import type { ProgressResponse, StageDetail, ProgressHistoryEntry } from '../types';
import { CheckCircle, Circle, Clock, TrendingUp, ChevronRight, Calendar, Zap, MapPin, ExternalLink } from 'lucide-react';
import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis,
  Tooltip, CartesianGrid
} from 'recharts';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const CHART_STYLE = {
  background: 'var(--bg-card)', border: '1px solid var(--border-primary)',
  borderRadius: 8, fontSize: 12,
};

interface DelayForecast {
  planned_progress_pct: number;
  actual_progress_pct: number;
  progress_variance_pct: number;
  delay_probability: number;
  is_delay_predicted: boolean;
  predicted_delay_days: number;
  planned_completion_date: string;
  predicted_completion_date: string;
  model_confidence: number;
  top_contributors: Array<{ feature: string; importance: number; description: string }>;
  explanations: string[];
}

export default function ProgressAnalysis() {
  const navigate = useNavigate();
  const [progress, setProgress] = useState<ProgressResponse | null>(null);
  const [history, setHistory] = useState<ProgressHistoryEntry[]>([]);
  const [delayForecast, setDelayForecast] = useState<DelayForecast | null>(null);
  const [dangerZones, setDangerZones] = useState<any[]>([]);
  const [saving, setSaving] = useState<number | null>(null);

  const fetchProgress = async () => {
    try {
      const res = await fetch(`${API_URL}/api/progress`);
      if (res.ok) setProgress(await res.json());
    } catch { /* */ }
  };

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_URL}/api/progress/history?limit=30`);
      if (res.ok) setHistory(await res.json());
    } catch { /* */ }
  };

  const fetchDelayPrediction = async () => {
    try {
      const res = await fetch(`${API_URL}/api/delay/prediction`);
      if (res.ok) setDelayForecast(await res.json());
    } catch { /* */ }
  };

  const fetchDangerZones = async () => {
    try {
      const res = await fetch(`${API_URL}/api/danger-zones`);
      if (res.ok) setDangerZones(await res.json());
    } catch { /* */ }
  };

  useEffect(() => {
    fetchProgress();
    fetchHistory();
    fetchDelayPrediction();
    fetchDangerZones();
    const iv = setInterval(() => {
      fetchProgress();
      fetchHistory();
      fetchDelayPrediction();
      fetchDangerZones();
    }, 5000);
    return () => clearInterval(iv);
  }, []);

  const setStage = async (stageIndex: number) => {
    await fetch(`${API_URL}/api/progress/stage`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stage_index: stageIndex, completion: 0 }),
    });
    fetchProgress();
    fetchDelayPrediction();
  };

  const setCompletion = async (stageIndex: number, completion: number) => {
    setSaving(stageIndex);
    await fetch(`${API_URL}/api/progress/stage`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stage_index: stageIndex, completion }),
    });
    setTimeout(() => { setSaving(null); fetchProgress(); fetchDelayPrediction(); }, 500);
  };

  const historyChartData = history.slice().reverse().map((h, i) => ({
    i,
    time: h.timestamp ? new Date(h.timestamp).toLocaleTimeString('en', { hour: '2-digit', minute: '2-digit' }) : '',
    progress: h.overall_progress,
    stage_completion: h.stage_completion,
  }));

  const overall = progress?.overall_progress_percentage ?? 0;
  const stages = progress?.stages ?? [];

  return (
    <>
      <Header title="Progress Analysis & Delay Forecasting" subtitle="9-Stage CNN stage recognition and Gradient Boosting delay prediction" />
      <div className="app-content">

        {/* Overall Progress Banner */}
        <div className="card" style={{ marginBottom: 16, background: 'linear-gradient(135deg, #1a1d28 0%, #1e2340 100%)', border: '1px solid rgba(59,130,246,0.2)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
            <div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.6px', marginBottom: 4 }}>Overall Construction Progress</div>
              <div style={{ fontSize: 36, fontWeight: 700, color: 'var(--text-primary)' }}>{overall.toFixed(1)}<span style={{ fontSize: 18, color: 'var(--text-muted)' }}>%</span></div>
              <div style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 4 }}>
                Current Stage: <strong style={{ color: 'var(--accent-blue)' }}>{progress?.current_stage ?? 'Not Started'}</strong>
                {progress && <> · {progress.stage_completion_percentage.toFixed(0)}% complete</>}
              </div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <span className={`badge ${progress?.progress_status === 'AHEAD' ? 'safe' : progress?.progress_status === 'DELAYED' ? 'high' : 'low'}`} style={{ fontSize: 12, padding: '4px 12px' }}>
                {progress?.progress_status ?? 'ON_TRACK'}
              </span>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6 }}>
                {progress?.is_model_prediction ? 'AI Vision Detection' : 'Manual Tracking Mode'}
              </div>
            </div>
          </div>
          {/* Progress bar */}
          <div style={{ height: 8, background: 'var(--bg-surface)', borderRadius: 4, overflow: 'hidden' }}>
            <div style={{
              height: '100%',
              width: `${overall}%`,
              background: 'linear-gradient(90deg, #3b82f6, #06b6d4)',
              borderRadius: 4,
              transition: 'width 0.5s ease',
            }} />
          </div>
        </div>

        {/* Delay Prediction Forecast Card */}
        {delayForecast && (
          <div className="card" style={{ marginBottom: 16, background: 'linear-gradient(135deg, rgba(30, 35, 55, 0.7) 0%, rgba(20, 24, 40, 0.9) 100%)', border: '1px solid rgba(245, 158, 11, 0.3)' }}>
            <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#f59e0b' }}>
                <Zap size={18} />
                Real-Time Construction Delay Prediction (Gradient Boosting Model)
              </span>
              <span className={`badge ${delayForecast.is_delay_predicted ? 'high' : 'safe'}`}>
                {delayForecast.is_delay_predicted ? '⚠️ DELAY RISK DETECTED' : '✓ SCHEDULE ON TRACK'}
              </span>
            </div>
            <div style={{ padding: 16 }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16, marginBottom: 16 }}>
                <div style={{ background: 'var(--bg-card-subtle)', padding: 12, borderRadius: 8 }}>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Delay Probability</div>
                  <div style={{ fontSize: 24, fontWeight: 700, color: delayForecast.delay_probability > 0.4 ? '#ef4444' : '#10b981' }}>
                    {(delayForecast.delay_probability * 100).toFixed(0)}%
                  </div>
                </div>

                <div style={{ background: 'var(--bg-card-subtle)', padding: 12, borderRadius: 8 }}>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Forecasted Delay Duration</div>
                  <div style={{ fontSize: 24, fontWeight: 700, color: 'var(--text-primary)' }}>
                    {delayForecast.predicted_delay_days.toFixed(1)} <span style={{ fontSize: 14, color: 'var(--text-muted)' }}>days</span>
                  </div>
                </div>

                <div style={{ background: 'var(--bg-card-subtle)', padding: 12, borderRadius: 8 }}>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Planned Completion Date</div>
                  <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--text-secondary)', marginTop: 4 }}>
                    <Calendar size={14} style={{ display: 'inline', marginRight: 4 }} />
                    {delayForecast.planned_completion_date}
                  </div>
                </div>

                <div style={{ background: 'var(--bg-card-subtle)', padding: 12, borderRadius: 8 }}>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Predicted Completion Date</div>
                  <div style={{ fontSize: 15, fontWeight: 700, color: delayForecast.predicted_delay_days > 0 ? '#f59e0b' : '#10b981', marginTop: 4 }}>
                    <Calendar size={14} style={{ display: 'inline', marginRight: 4 }} />
                    {delayForecast.predicted_completion_date}
                  </div>
                </div>
              </div>

              {/* Explanations & Feature Importance */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
                <div>
                  <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', marginBottom: 6 }}>
                    DIAGNOSTIC EXPLANATIONS
                  </div>
                  {delayForecast.explanations.map((exp, i) => (
                    <div key={i} style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 4 }}>
                      • {exp}
                    </div>
                  ))}
                </div>

                <div>
                  <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', marginBottom: 6 }}>
                    TOP FEATURE CONTRIBUTORS
                  </div>
                  {delayForecast.top_contributors.map((c, i) => (
                    <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, padding: '4px 0', borderBottom: '1px solid var(--border-secondary)' }}>
                      <span style={{ color: 'var(--text-secondary)' }}>{c.description}</span>
                      <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-blue)', fontWeight: 600 }}>
                        {(c.importance * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>

            </div>
          </div>
        )}

        {/* Stage-Linked Danger Zones & Geofenced Hazard Intelligence Card */}
        <div className="card" style={{
          marginBottom: 16,
          background: 'linear-gradient(135deg, rgba(20, 27, 45, 0.85), rgba(15, 23, 42, 0.95))',
          border: '1px solid rgba(56, 189, 248, 0.25)',
          boxShadow: 'var(--shadow-card)'
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
                <h3 style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)' }}>
                  Stage-Linked Danger Zones & Geofence Spatial Intelligence
                </h3>
                <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                  Active hazard boundaries correlating with current milestone: <strong style={{ color: 'var(--accent-blue)' }}>{progress?.current_stage ?? 'Structural Work'}</strong>
                </p>
              </div>
            </div>

            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <span className="badge safe" style={{ fontSize: 11 }}>
                {dangerZones.filter(z => z.is_active !== false).length} Active Zones Monitored
              </span>
              <button
                className="btn btn-primary"
                onClick={() => navigate('/danger-zones')}
                style={{ fontSize: 11, padding: '4px 10px' }}
              >
                <ExternalLink size={12} /> Configure Danger Zones
              </button>
            </div>
          </div>

          {dangerZones.length === 0 ? (
            <div style={{ padding: 12, background: 'rgba(255, 255, 255, 0.02)', borderRadius: 8, textAlign: 'center', color: 'var(--text-muted)', fontSize: 12 }}>
              No danger zones configured yet for this construction stage. <button className="btn btn-ghost" onClick={() => navigate('/danger-zones')} style={{ fontSize: 11, padding: '2px 8px', marginLeft: 6 }}>+ Draw New Zone</button>
            </div>
          ) : (
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
              gap: 10
            }}>
              {dangerZones.map((zone: any, idx: number) => {
                const type = zone.zone_type || 'RESTRICTED';
                const color = type === 'RESTRICTED' ? '#ef4444' : type === 'HAZARD' ? '#f97316' : type === 'EQUIPMENT' ? '#f59e0b' : '#8b5cf6';

                return (
                  <div
                    key={zone.id || idx}
                    style={{
                      padding: 10,
                      background: 'rgba(255, 255, 255, 0.03)',
                      borderRadius: 8,
                      border: `1px solid ${color}40`,
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 4
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <strong style={{ fontSize: 12.5, color: 'var(--text-primary)' }}>
                        {zone.name || `Danger Zone #${idx + 1}`}
                      </strong>
                      <span
                        style={{
                          fontSize: 10,
                          fontWeight: 700,
                          padding: '2px 6px',
                          borderRadius: 4,
                          background: `${color}25`,
                          color: color,
                          border: `1px solid ${color}60`
                        }}
                      >
                        {type}
                      </span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
                      <span>Risk Weight: <strong style={{ color: 'var(--text-secondary)' }}>+{zone.risk_weight ?? 30} pts</strong></span>
                      <span>Polygon Points: <strong style={{ color: 'var(--text-secondary)' }}>{zone.polygon_data?.length ?? 4} pts</strong></span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 380px', gap: 16 }}>
          {/* Left: Timeline + History Chart */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

            {/* Progress History Chart */}
            <div className="chart-card">
              <h3>Progress Over Time</h3>
              {historyChartData.length > 1 ? (
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={historyChartData}>
                    <defs>
                      <linearGradient id="prog-grad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                    <XAxis dataKey="time" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} interval="preserveStartEnd" />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                    <Tooltip contentStyle={CHART_STYLE} formatter={(val: any) => [`${Number(val || 0).toFixed(1)}%`]} />
                    <Area type="monotone" dataKey="progress" name="Overall %" stroke="#3b82f6" fill="url(#prog-grad)" strokeWidth={2} dot={false} />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <div className="empty-state" style={{ padding: 30 }}>
                  <TrendingUp size={32} style={{ opacity: 0.3 }} />
                  <p>Progress history will appear here as analysis runs</p>
                </div>
              )}
            </div>

            {/* Stage Timeline */}
            <div className="card">
              <div className="card-header"><span className="card-title">Construction Stages (9-Stage Model)</span></div>
              <div className="stage-timeline">
                {stages.map((stage: StageDetail) => (
                  <div key={stage.index} className={`stage-item ${stage.status}`}>
                    <div className={`stage-marker ${stage.status}`}>
                      {stage.status === 'completed' ? <CheckCircle size={12} /> :
                       stage.status === 'current' ? <ChevronRight size={12} /> :
                       <Circle size={12} />}
                    </div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 2 }}>
                        <span style={{ fontSize: 13, fontWeight: stage.status === 'current' ? 600 : 400, color: stage.status === 'pending' ? 'var(--text-muted)' : 'var(--text-primary)' }}>
                          {stage.name}
                        </span>
                        <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                          {stage.completion.toFixed(0)}% · weight {stage.weight}%
                        </span>
                      </div>
                      {stage.status === 'current' && (
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 6 }}>
                          <input
                            type="range"
                            min={0} max={100}
                            value={stage.completion}
                            onChange={e => setCompletion(stage.index, Number(e.target.value))}
                            style={{ flex: 1 }}
                          />
                          <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--accent-blue)', minWidth: 32 }}>
                            {saving === stage.index ? '...' : `${stage.completion.toFixed(0)}%`}
                          </span>
                        </div>
                      )}
                      {stage.status !== 'current' && stage.status !== 'pending' && (
                        <div style={{ height: 3, background: 'var(--bg-surface)', borderRadius: 2, marginTop: 4, overflow: 'hidden' }}>
                          <div style={{ height: '100%', width: `${stage.completion}%`, background: stage.status === 'completed' ? 'var(--accent-green)' : 'var(--accent-blue)', borderRadius: 2 }} />
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right: Stage Controls */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div className="card">
              <div className="card-header"><span className="card-title">Set Active Stage</span></div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {stages.map((stage: StageDetail) => (
                  <button
                    key={stage.index}
                    onClick={() => setStage(stage.index)}
                    style={{
                      padding: '10px 14px',
                      background: stage.status === 'current' ? 'rgba(59,130,246,0.12)' : 'var(--bg-surface)',
                      border: `1px solid ${stage.status === 'current' ? 'rgba(59,130,246,0.4)' : 'var(--border-secondary)'}`,
                      borderRadius: 'var(--radius-sm)',
                      color: stage.status === 'current' ? 'var(--accent-blue)' : 'var(--text-secondary)',
                      cursor: 'pointer',
                      textAlign: 'left',
                      fontSize: 13,
                      fontFamily: 'var(--font-sans)',
                      fontWeight: stage.status === 'current' ? 600 : 400,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      transition: 'all 0.15s',
                    }}
                  >
                    <span>{stage.index + 1}. {stage.name}</span>
                    <span className={`badge ${stage.status === 'completed' ? 'safe' : stage.status === 'current' ? 'low' : ''}`} style={{ fontSize: 10 }}>
                      {stage.status}
                    </span>
                  </button>
                ))}
              </div>
            </div>

            {/* AI Image Analysis & Upload Card */}
            <div className="card" style={{ border: '1px solid rgba(59,130,246,0.3)', background: 'linear-gradient(135deg, rgba(20,25,45,0.8) 0%, rgba(15,20,35,0.9) 100%)' }}>
              <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--accent-blue)' }}>
                  <Zap size={14} /> AI Site Photo Classifier
                </span>
                <span className="badge cyan" style={{ fontSize: 10 }}>9-Stage CNN</span>
              </div>
              <div style={{ padding: '4px 0' }}>
                <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10 }}>
                  Upload a construction site photo to run instant deep-learning progress classification.
                </p>
                
                <input
                  type="file"
                  id="progress-image-upload"
                  accept="image/*"
                  style={{ display: 'none' }}
                  onChange={async (e) => {
                    const file = e.target.files?.[0];
                    if (!file) return;
                    const formData = new FormData();
                    formData.append('file', file);
                    try {
                      const res = await fetch(`${API_URL}/api/progress/analyze-image`, {
                        method: 'POST',
                        body: formData,
                      });
                      if (res.ok) {
                        const data = await res.json();
                        fetchProgress();
                        fetchHistory();
                        fetchDelayPrediction();
                        alert(`✓ AI Predicted Stage: ${data.predicted_stage} (${(data.confidence * 100).toFixed(1)}% confidence)`);
                      }
                    } catch (err) {
                      console.error('Image analysis error:', err);
                    }
                  }}
                />
                
                <label
                  htmlFor="progress-image-upload"
                  className="btn btn-primary"
                  style={{ width: '100%', justifyContent: 'center', cursor: 'pointer', marginBottom: 12 }}
                >
                  📸 Upload Site Photo for AI Analysis
                </label>

                {/* 9-Stage AI Probability Distribution */}
                {stages.some(s => (s.probability ?? 0) > 0) && (
                  <div>
                    <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 6 }}>
                      AI Class Probability Distribution
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                      {stages.map((s: StageDetail) => {
                        const prob = s.probability ?? 0;
                        return (
                          <div key={s.index} style={{ fontSize: 11 }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                              <span style={{ color: s.status === 'current' ? 'var(--accent-blue)' : 'var(--text-secondary)', fontWeight: s.status === 'current' ? 600 : 400 }}>
                                {s.name}
                              </span>
                              <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                                {(prob * 100).toFixed(1)}%
                              </span>
                            </div>
                            <div style={{ height: 4, background: 'var(--bg-surface)', borderRadius: 2, overflow: 'hidden' }}>
                              <div style={{
                                height: '100%',
                                width: `${Math.min(100, prob * 100)}%`,
                                background: s.status === 'current' ? 'var(--accent-blue)' : 'rgba(255,255,255,0.2)',
                                borderRadius: 2,
                              }} />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="card">
              <div className="card-header"><span className="card-title">Tracking Mode</span></div>
              <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.8 }}>
                <p><strong>Current Mode:</strong> {progress?.is_model_prediction ? 'AI Vision Detection' : 'Manual Input'}</p>
                <p style={{ marginTop: 8, fontSize: 12, color: 'var(--text-muted)' }}>
                  {progress?.is_model_prediction
                    ? 'Stage is detected automatically from camera frames using the 9-stage progress model.'
                    : 'Use the sliders and stage buttons above to manually update progress. Snapshots are saved every 30 seconds into MongoDB.'}
                </p>
              </div>
            </div>

            <div className="card">
              <div className="card-header"><span className="card-title">Recent MongoDB Records</span></div>
              {history.length === 0 ? (
                <div style={{ fontSize: 13, color: 'var(--text-muted)', padding: '8px 0' }}>
                  No progress records yet
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {history.slice(0, 6).map((h, i) => (
                    <div key={i} style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      fontSize: 12,
                      padding: '6px 0',
                      borderBottom: i < 5 ? '1px solid var(--border-primary)' : 'none',
                    }}>
                      <span style={{ color: 'var(--text-muted)' }}>
                        <Clock size={10} style={{ display: 'inline', marginRight: 4 }} />
                        {h.timestamp ? new Date(h.timestamp).toLocaleTimeString() : '—'}
                      </span>
                      <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-blue)' }}>
                        {h.overall_progress?.toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
