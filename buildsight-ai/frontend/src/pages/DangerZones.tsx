import { useState, useEffect, useRef, useCallback } from 'react';
import { Header } from '../components/common/Header';
import { Plus, Trash2 } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const ZONE_COLORS = {
  RESTRICTED: '#ef4444',
  HAZARD: '#f97316',
  EQUIPMENT: '#f59e0b',
  EDGE: '#8b5cf6',
};

type DrawMode = 'view' | 'draw';
type ZoneType = 'RESTRICTED' | 'HAZARD' | 'EQUIPMENT' | 'EDGE';

interface DangerZone {
  id: number;
  name: string;
  zone_type: ZoneType;
  polygon_data: [number, number][];
  risk_weight: number;
  is_active: boolean;
}

export default function DangerZones() {
  const [zones, setZones] = useState<DangerZone[]>([]);
  const [drawMode, setDrawMode] = useState<DrawMode>('view');
  const [currentPoints, setCurrentPoints] = useState<[number, number][]>([]);
  const [zoneName, setZoneName] = useState('Restricted Area');
  const [zoneType, setZoneType] = useState<ZoneType>('RESTRICTED');
  const [riskWeight, setRiskWeight] = useState(30);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const fetchZones = async () => {
    try {
      const res = await fetch(`${API_URL}/api/danger-zones`);
      if (res.ok) setZones(await res.json());
    } catch { /* */ }
  };

  useEffect(() => {
    fetchZones();
    const interval = setInterval(fetchZones, 5000);
    return () => clearInterval(interval);
  }, []);

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 1;
    for (let x = 0; x < canvas.width; x += 40) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += 40) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
    }

    // Label
    ctx.fillStyle = 'rgba(255,255,255,0.1)';
    ctx.font = '13px Inter, sans-serif';
    ctx.fillText('Construction Site View (16:9)', 20, 30);

    // Existing zones
    for (const zone of zones) {
      if (zone.polygon_data.length < 3) continue;
      const color = ZONE_COLORS[zone.zone_type] || '#ef4444';
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.fillStyle = color.replace(')', ', 0.15)').replace('rgb', 'rgba');
      ctx.beginPath();
      ctx.moveTo(zone.polygon_data[0][0], zone.polygon_data[0][1]);
      zone.polygon_data.slice(1).forEach(p => ctx.lineTo(p[0], p[1]));
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // Label
      const cx = zone.polygon_data.reduce((s, p) => s + p[0], 0) / zone.polygon_data.length;
      const cy = zone.polygon_data.reduce((s, p) => s + p[1], 0) / zone.polygon_data.length;
      ctx.fillStyle = color;
      ctx.font = 'bold 11px Inter';
      ctx.fillText(zone.name, cx - 30, cy);
    }

    // Current drawing
    if (currentPoints.length > 0) {
      const color = ZONE_COLORS[zoneType];
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(currentPoints[0][0], currentPoints[0][1]);
      currentPoints.forEach(p => ctx.lineTo(p[0], p[1]));
      ctx.stroke();
      ctx.setLineDash([]);

      // Points
      currentPoints.forEach((p, i) => {
        ctx.fillStyle = i === 0 ? '#ffffff' : color;
        ctx.beginPath();
        ctx.arc(p[0], p[1], 5, 0, Math.PI * 2);
        ctx.fill();
      });
    }
  }, [zones, currentPoints, zoneType]);

  useEffect(() => {
    drawCanvas();
  }, [drawCanvas]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (drawMode !== 'draw') return;
    const rect = canvasRef.current!.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
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
    } catch { /* */ }
  };

  const deleteZone = async (id: number) => {
    await fetch(`${API_URL}/api/danger-zones/${id}`, { method: 'DELETE' });
    setZones(prev => prev.filter(z => z.id !== id));
  };

  return (
    <>
      <Header title="Danger Zones" subtitle="Define restricted areas for worker safety monitoring" />
      <div className="app-content">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 16, alignItems: 'start' }}>

          {/* Canvas */}
          <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
            <div style={{
              padding: '12px 16px',
              borderBottom: '1px solid var(--border-primary)',
              display: 'flex',
              alignItems: 'center',
              gap: 12,
            }}>
              <span className="card-title">Zone Canvas</span>
              {drawMode === 'view' ? (
                <button className="btn btn-primary" onClick={() => setDrawMode('draw')}>
                  <Plus size={14} /> Draw Zone
                </button>
              ) : (
                <>
                  <button
                    className="btn btn-success"
                    onClick={saveZone}
                    disabled={currentPoints.length < 3}
                  >
                    Save Zone
                  </button>
                  <button className="btn btn-ghost" onClick={() => { setDrawMode('view'); setCurrentPoints([]); }}>
                    Cancel
                  </button>
                  <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                    {currentPoints.length < 3 ? `Click to add points (${currentPoints.length}/3 min)` : 'Double-click to finish'}
                  </span>
                </>
              )}
            </div>
            <div ref={containerRef} style={{ position: 'relative' }}>
              <canvas
                ref={canvasRef}
                width={760}
                height={428}
                onClick={handleCanvasClick}
                onDoubleClick={handleCanvasDoubleClick}
                style={{
                  width: '100%',
                  display: 'block',
                  cursor: drawMode === 'draw' ? 'crosshair' : 'default',
                  background: 'var(--bg-surface)',
                }}
              />
            </div>
          </div>

          {/* Controls */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

            {/* Zone Config */}
            <div className="card">
              <div className="card-header"><span className="card-title">Zone Config</span></div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <div>
                  <label style={{ fontSize: 12, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>Name</label>
                  <input type="text" value={zoneName} onChange={e => setZoneName(e.target.value)} className="input-field" />
                </div>
                <div>
                  <label style={{ fontSize: 12, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>Type</label>
                  <select value={zoneType} onChange={e => setZoneType(e.target.value as ZoneType)} className="control-select" style={{ width: '100%' }}>
                    <option value="RESTRICTED">Restricted</option>
                    <option value="HAZARD">Hazard</option>
                    <option value="EQUIPMENT">Equipment</option>
                    <option value="EDGE">Edge / Drop</option>
                  </select>
                </div>
                <div>
                  <label style={{ fontSize: 12, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>
                    Risk Weight: {riskWeight}
                  </label>
                  <input
                    type="range" min="0" max="100" value={riskWeight}
                    onChange={e => setRiskWeight(Number(e.target.value))}
                    style={{ width: '100%' }}
                  />
                </div>
              </div>
            </div>

            {/* Existing Zones List */}
            <div className="card">
              <div className="card-header"><span className="card-title">Zones ({zones.length})</span></div>
              {zones.length === 0 ? (
                <div style={{ textAlign: 'center', padding: 20, fontSize: 13, color: 'var(--text-muted)' }}>
                  No zones defined yet
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {zones.map(zone => (
                    <div key={zone.id} style={{
                      padding: '8px 12px',
                      background: 'var(--bg-surface)',
                      borderRadius: 'var(--radius-sm)',
                      borderLeft: `3px solid ${ZONE_COLORS[zone.zone_type]}`,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                    }}>
                      <div>
                        <div style={{ fontSize: 13, fontWeight: 600 }}>{zone.name}</div>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                          {zone.zone_type} · Risk: {zone.risk_weight}
                        </div>
                      </div>
                      <button
                        className="btn btn-ghost"
                        onClick={() => deleteZone(zone.id)}
                        style={{ padding: '4px 8px' }}
                      >
                        <Trash2 size={12} />
                      </button>
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
