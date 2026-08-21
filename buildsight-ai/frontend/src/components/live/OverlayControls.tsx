import type { OverlaySettings } from '../../types';

interface OverlayControlsProps {
  settings: OverlaySettings;
  onChange: (settings: OverlaySettings) => void;
}

function Toggle({ label, active, onToggle }: { label: string; active: boolean; onToggle: () => void }) {
  return (
    <label className="toggle" onClick={onToggle}>
      <div className={`toggle-switch ${active ? 'active' : ''}`} />
      <span>{label}</span>
    </label>
  );
}

export function OverlayControls({ settings, onChange }: OverlayControlsProps) {
  const toggle = (key: keyof OverlaySettings) => {
    onChange({ ...settings, [key]: !settings[key] });
  };

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title">Overlays</span>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <Toggle label="Bounding Boxes" active={settings.showBoundingBoxes} onToggle={() => toggle('showBoundingBoxes')} />
        <Toggle label="Worker IDs" active={settings.showWorkerIds} onToggle={() => toggle('showWorkerIds')} />
        <Toggle label="PPE Status" active={settings.showPPEStatus} onToggle={() => toggle('showPPEStatus')} />
        <Toggle label="Risk Labels" active={settings.showRiskLabels} onToggle={() => toggle('showRiskLabels')} />
        <Toggle label="Confidence" active={settings.showConfidence} onToggle={() => toggle('showConfidence')} />
      </div>
    </div>
  );
}
