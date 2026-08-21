import { useState, useRef, useCallback } from 'react';
import { Header } from '../components/common/Header';
import { VideoOverlay } from '../components/live/VideoOverlay';
import { OverlayControls } from '../components/live/OverlayControls';
import { WorkerPanel } from '../components/live/WorkerPanel';
import { useWebcam } from '../hooks/useWebcam';
import { useFrameProcessor } from '../hooks/useFrameProcessor';
import {
  Video, VideoOff, Camera, MonitorPlay, Wifi, WifiOff,
  Cpu, Timer, Users, RotateCcw, FlipHorizontal
} from 'lucide-react';
import type { OverlaySettings, AnalyticsMessage, ConnectionStatus } from '../types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function LiveMonitoring() {
  const webcam = useWebcam();
  const videoElementRef = useRef<HTMLVideoElement | null>(null);
  const [resetting, setResetting] = useState(false);
  const [isMirrored, setIsMirrored] = useState(true);

  const [overlaySettings, setOverlaySettings] = useState<OverlaySettings>({
    showBoundingBoxes: true,
    showWorkerIds: true,
    showPPEStatus: true,
    showRiskLabels: true,
    showConfidence: false,
  });

  const [analytics, setAnalytics] = useState<AnalyticsMessage | null>(null);

  const handleAnalytics = useCallback((data: AnalyticsMessage) => {
    setAnalytics(data);
  }, []);

  const frameProcessor = useFrameProcessor({
    fps: 10,
    onAnalytics: handleAnalytics,
  });

  // Start webcam + processing
  const handleStart = async () => {
    await webcam.start();
    if (webcam.videoRef.current) {
      frameProcessor.startCapture(webcam.videoRef.current);
      videoElementRef.current = webcam.videoRef.current;
    }
  };

  // Stop webcam + processing
  const handleStop = () => {
    frameProcessor.stopCapture();
    webcam.stop();
    videoElementRef.current = null;
    setAnalytics(null);
  };

  const handleResetTracks = async () => {
    setResetting(true);
    try {
      await fetch(`${API_URL}/api/video/reset`, { method: 'POST' });
      await fetch(`${API_URL}/api/workers`, { method: 'DELETE' });
      setAnalytics(null);
    } catch {
      //
    } finally {
      setResetting(false);
    }
  };

  const handleCameraChange = async (deviceId: string) => {
    webcam.setSelectedCamera(deviceId);
    if (webcam.isActive) {
      handleStop();
      setTimeout(handleStart, 300);
    }
  };

  const handleResolutionChange = (idx: number) => {
    webcam.setSelectedResolution(webcam.resolutions[idx]);
    if (webcam.isActive) {
      handleStop();
      setTimeout(handleStart, 300);
    }
  };

  const perf = analytics?.performance;
  const safety = analytics?.safety;

  return (
    <>
      <Header
        title="Live Monitoring"
        subtitle="Real-time AI-powered safety analysis"
        connectionStatus={frameProcessor.frameWsStatus as ConnectionStatus}
        processingActive={frameProcessor.isCapturing}
      />
      <div className="app-content" style={{ padding: 16 }}>
        <div className="live-container">
          {/* Left: Video + Controls */}
          <div className="live-video-section">
            {/* Controls Bar */}
            <div className="controls-bar">
              {!webcam.isActive ? (
                <button className="btn btn-success" onClick={handleStart}>
                  <Video size={16} />
                  Start Webcam
                </button>
              ) : (
                <button className="btn btn-danger" onClick={handleStop}>
                  <VideoOff size={16} />
                  Stop Webcam
                </button>
              )}

              {/* Clear Tracks Button */}
              <button
                className="btn btn-ghost"
                onClick={handleResetTracks}
                disabled={resetting}
                title="Clear live worker tracker trajectories and reset tracking memory"
                style={{ fontSize: 12, padding: '6px 12px' }}
              >
                <RotateCcw size={14} className={resetting ? 'spin' : ''} />
                {resetting ? 'Clearing...' : 'Clear Live Tracks'}
              </button>

              {/* Camera Orientation / Unmirror Toggle */}
              <button
                className={`btn ${isMirrored ? 'btn-primary' : 'btn-ghost'}`}
                onClick={() => setIsMirrored(!isMirrored)}
                title="Click to toggle between Mirrored (selfie) and Unmirrored (true scene) camera view"
                style={{
                  fontSize: 12,
                  padding: '6px 12px',
                  borderColor: isMirrored ? 'var(--cyan)' : 'var(--border-primary)',
                  color: isMirrored ? '#ffffff' : 'var(--text-secondary)'
                }}
              >
                <FlipHorizontal size={14} />
                {isMirrored ? '🪞 Mirrored View' : '📷 Unmirrored View'}
              </button>

              {/* Camera Selector */}
              {webcam.cameras.length > 0 && (
                <select
                  className="control-select"
                  value={webcam.selectedCamera}
                  onChange={(e) => handleCameraChange(e.target.value)}
                >
                  {webcam.cameras.map((cam) => (
                    <option key={cam.deviceId} value={cam.deviceId}>
                      {cam.label}
                    </option>
                  ))}
                </select>
              )}

              {/* Resolution Selector */}
              <select
                className="control-select"
                value={webcam.resolutions.indexOf(webcam.selectedResolution)}
                onChange={(e) => handleResolutionChange(Number(e.target.value))}
              >
                {webcam.resolutions.map((res, i) => (
                  <option key={i} value={i}>{res.label}</option>
                ))}
              </select>

              {/* Status */}
              <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 16 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
                  {frameProcessor.frameWsStatus === 'connected' ? (
                    <Wifi size={14} style={{ color: 'var(--accent-green)' }} />
                  ) : (
                    <WifiOff size={14} style={{ color: 'var(--accent-red)' }} />
                  )}
                  <span style={{ color: 'var(--text-muted)' }}>
                    {frameProcessor.frameWsStatus}
                  </span>
                </div>
              </div>
            </div>

            {/* Video Area */}
            <div className="video-wrapper">
              <video
                ref={webcam.videoRef}
                style={{
                  width: '100%',
                  height: '100%',
                  objectFit: 'contain',
                  display: webcam.isActive ? 'block' : 'none',
                  transform: isMirrored ? 'scaleX(-1)' : 'none',
                }}
                playsInline
                muted
              />

              {webcam.isActive && (
                <VideoOverlay
                  videoElement={webcam.videoRef.current}
                  workers={frameProcessor.trackedWorkers}
                  overlaySettings={overlaySettings}
                  isMirrored={isMirrored}
                />
              )}

              {/* Overlay Stats */}
              {webcam.isActive && (
                <div className="video-overlay-stats">
                  <div className="overlay-badge green">
                    <Cpu size={10} style={{ marginRight: 4 }} />
                    {perf?.inference_fps?.toFixed(1) ?? '0.0'} FPS
                  </div>
                  <div className="overlay-badge blue">
                    <Camera size={10} style={{ marginRight: 4 }} />
                    {frameProcessor.captureFps} CAP
                  </div>
                  <div className="overlay-badge amber">
                    <Timer size={10} style={{ marginRight: 4 }} />
                    {perf?.latency_ms?.toFixed(0) ?? '0'}ms
                  </div>
                  <div className="overlay-badge">
                    <Users size={10} style={{ marginRight: 4 }} />
                    {frameProcessor.trackedWorkers.length}
                  </div>
                </div>
              )}

              {/* Empty State */}
              {!webcam.isActive && (
                <div className="empty-state">
                  <MonitorPlay size={64} style={{ opacity: 0.3, marginBottom: 16 }} />
                  <h3>No Active Feed</h3>
                  <p>Click Start Webcam to begin real-time analysis</p>
                  {webcam.error && (
                    <div style={{
                      marginTop: 16,
                      padding: '10px 16px',
                      background: 'rgba(239, 68, 68, 0.1)',
                      border: '1px solid rgba(239, 68, 68, 0.3)',
                      borderRadius: 'var(--radius-md)',
                      color: 'var(--accent-red)',
                      fontSize: 13,
                      maxWidth: 400,
                    }}>
                      {webcam.error}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Right: Analytics Sidebar */}
          <div className="live-sidebar">
            {/* Workers Panel */}
            <WorkerPanel
              workers={frameProcessor.trackedWorkers}
              activeViolations={safety?.active_violations ?? 0}
              ppeCompliance={safety?.ppe_compliance_percentage ?? 0}
            />

            {/* Overlay Controls */}
            <OverlayControls
              settings={overlaySettings}
              onChange={setOverlaySettings}
            />

            {/* Performance Card */}
            <div className="card">
              <div className="card-header">
                <span className="card-title">Performance</span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                  <span style={{ color: 'var(--text-muted)' }}>Capture FPS</span>
                  <span style={{ fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
                    {frameProcessor.captureFps}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                  <span style={{ color: 'var(--text-muted)' }}>AI Inference FPS</span>
                  <span style={{ fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
                    {perf?.inference_fps?.toFixed(1) ?? '0.0'}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                  <span style={{ color: 'var(--text-muted)' }}>Latency</span>
                  <span style={{ fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
                    {perf?.latency_ms?.toFixed(0) ?? '0'} ms
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                  <span style={{ color: 'var(--text-muted)' }}>Worker Count</span>
                  <span style={{ fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
                    {frameProcessor.trackedWorkers.length}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
