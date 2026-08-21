import { useState, useRef, useCallback } from 'react';
import { Header } from '../components/common/Header';
import { Upload, Play, Pause, Square, FileVideo, CheckCircle } from 'lucide-react';
import type { AnalyticsMessage, ConnectionStatus } from '../types';
import { VideoOverlay } from '../components/live/VideoOverlay';
import { WorkerPanel } from '../components/live/WorkerPanel';
import { OverlayControls } from '../components/live/OverlayControls';
import { useFrameProcessor } from '../hooks/useFrameProcessor';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

type UploadState = 'idle' | 'uploading' | 'uploaded' | 'analyzing' | 'paused' | 'complete';

export default function UploadAnalysis() {
  const [uploadState, setUploadState] = useState<UploadState>('idle');
  const [uploadedFile, setUploadedFile] = useState<{ name: string; size_mb: number; file_path: string } | null>(null);
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [analytics, setAnalytics] = useState<AnalyticsMessage | null>(null);
  const [overlaySettings, setOverlaySettings] = useState({
    showBoundingBoxes: true,
    showWorkerIds: true,
    showPPEStatus: true,
    showRiskLabels: true,
    showConfidence: false,
  });

  const videoRef = useRef<HTMLVideoElement>(null);
  const objectUrlRef = useRef<string | null>(null);

  const handleAnalytics = useCallback((data: AnalyticsMessage) => {
    setAnalytics(data);
  }, []);

  const frameProcessor = useFrameProcessor({ fps: 8, onAnalytics: handleAnalytics });

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
  }, []);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFileSelect(file);
  };

  const handleFileSelect = async (file: File) => {
    const allowed = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska'];
    if (!allowed.includes(file.type) && !file.name.match(/\.(mp4|avi|mov|mkv)$/i)) {
      setError('Unsupported file format. Please upload MP4, AVI, MOV, or MKV.');
      return;
    }

    setError(null);
    setUploadState('uploading');

    // Create local object URL for instant preview
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    const url = URL.createObjectURL(file);
    objectUrlRef.current = url;
    setVideoSrc(url);

    const sizeMb = Math.round((file.size / (1024 * 1024)) * 100) / 100;
    setUploadedFile({ name: file.name, size_mb: sizeMb, file_path: file.name });

    // Upload to backend
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch(`${API_URL}/api/sources/upload`, {
        method: 'POST',
        body: formData,
      });
      if (res.ok) {
        const data = await res.json();
        setUploadedFile({
          name: data.filename || file.name,
          size_mb: data.size_mb || sizeMb,
          file_path: data.file_path || file.name,
        });
      }
      setUploadState('uploaded');
    } catch (err: any) {
      // Even if backend upload API has a network glitch, keep local object URL ready
      setUploadState('uploaded');
    }
  };

  const handleAnalyze = async () => {
    if (!videoSrc) return;
    setUploadState('analyzing');
    setError(null);

    try {
      if (videoRef.current) {
        videoRef.current.src = videoSrc;
        videoRef.current.currentTime = 0;
        videoRef.current.muted = true;
        await videoRef.current.play();
        frameProcessor.startCapture(videoRef.current);
      }
    } catch (err: any) {
      setError(`Playback error: ${err.message}`);
      setUploadState('uploaded');
    }
  };

  const handlePause = () => {
    videoRef.current?.pause();
    setUploadState('paused');
  };

  const handleResume = async () => {
    try {
      if (videoRef.current) {
        await videoRef.current.play();
        setUploadState('analyzing');
      }
    } catch (err: any) {
      setError(`Resume error: ${err.message}`);
    }
  };

  const handleStop = () => {
    frameProcessor.stopCapture();
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.currentTime = 0;
    }
    setUploadState('uploaded');
    setAnalytics(null);
  };

  const handleReset = () => {
    handleStop();
    setUploadedFile(null);
    setVideoSrc(null);
    setUploadState('idle');
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
    if (videoRef.current) videoRef.current.src = '';
  };

  const safety = analytics?.safety;
  const perf = analytics?.performance;

  return (
    <>
      <Header
        title="Video Upload Analysis"
        subtitle="Analyze recorded construction video files"
        connectionStatus={frameProcessor.frameWsStatus as ConnectionStatus}
        processingActive={uploadState === 'analyzing'}
      />
      <div className="app-content" style={{ padding: 16 }}>
        <div className="live-container">
          {/* Left: Video Area */}
          <div className="live-video-section">
            {/* Controls */}
            <div className="controls-bar">
              {uploadState === 'idle' || uploadState === 'uploading' ? (
                <label className="btn btn-primary" style={{ cursor: 'pointer' }}>
                  <Upload size={16} />
                  {uploadState === 'uploading' ? 'Uploading...' : 'Upload Video'}
                  <input type="file" accept="video/*" onChange={handleFileInput} style={{ display: 'none' }} />
                </label>
              ) : null}

              {uploadState === 'uploaded' && (
                <button className="btn btn-success" onClick={handleAnalyze}>
                  <Play size={16} /> Start Analysis
                </button>
              )}
              {uploadState === 'analyzing' && (
                <>
                  <button className="btn btn-ghost" onClick={handlePause}>
                    <Pause size={16} /> Pause
                  </button>
                  <button className="btn btn-danger" onClick={handleStop}>
                    <Square size={16} /> Stop
                  </button>
                </>
              )}
              {uploadState === 'paused' && (
                <>
                  <button className="btn btn-success" onClick={handleResume}>
                    <Play size={16} /> Resume
                  </button>
                  <button className="btn btn-danger" onClick={handleStop}>
                    <Square size={16} /> Stop
                  </button>
                </>
              )}
              {(uploadState === 'uploaded' || uploadState === 'paused') && (
                <button className="btn btn-ghost" onClick={handleReset}>Reset</button>
              )}

              <div style={{ marginLeft: 'auto', fontSize: 12, color: 'var(--text-muted)' }}>
                {uploadedFile && `${uploadedFile.name} (${uploadedFile.size_mb} MB)`}
              </div>
            </div>

            {/* Video / Drop Zone */}
            <div className="video-wrapper">
              {uploadState === 'idle' ? (
                <div
                  className="empty-state"
                  style={{
                    border: '2px dashed var(--border-secondary)',
                    borderRadius: 'var(--radius-lg)',
                    cursor: 'pointer',
                    width: '100%',
                    height: '100%',
                  }}
                  onDrop={handleDrop}
                  onDragOver={(e) => e.preventDefault()}
                >
                  <FileVideo size={64} style={{ opacity: 0.3, marginBottom: 16 }} />
                  <h3>Drop Video Here</h3>
                  <p>or click Upload Video to browse</p>
                  <p style={{ marginTop: 8, fontSize: 12 }}>Supports: MP4, AVI, MOV, MKV</p>
                </div>
              ) : (
                <>
                  <video
                    ref={videoRef}
                    src={videoSrc || undefined}
                    style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                    controls
                    muted
                    loop
                    playsInline
                  />
                  {(uploadState === 'analyzing' || uploadState === 'paused') && (
                    <VideoOverlay
                      videoElement={videoRef.current}
                      workers={frameProcessor.trackedWorkers}
                      overlaySettings={overlaySettings}
                    />
                  )}
                  {/* Performance overlay */}
                  {uploadState === 'analyzing' && (
                    <div className="video-overlay-stats">
                      <div className="overlay-badge green">{perf?.inference_fps?.toFixed(1)} FPS</div>
                      <div className="overlay-badge amber">{perf?.latency_ms?.toFixed(0)}ms</div>
                      <div className="overlay-badge">{frameProcessor.trackedWorkers.length} workers</div>
                    </div>
                  )}
                </>
              )}

              {uploadState === 'uploaded' && (
                <div style={{
                  position: 'absolute',
                  top: 12,
                  left: 12,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  padding: '6px 12px',
                  background: 'rgba(16,185,129,0.15)',
                  border: '1px solid rgba(16,185,129,0.3)',
                  borderRadius: 'var(--radius-md)',
                  fontSize: 12,
                  color: 'var(--accent-green)',
                }}>
                  <CheckCircle size={14} />
                  Ready for analysis
                </div>
              )}
            </div>

            {error && (
              <div style={{
                padding: '8px 16px',
                background: 'rgba(239,68,68,0.1)',
                border: '1px solid rgba(239,68,68,0.3)',
                borderRadius: 'var(--radius-md)',
                color: 'var(--accent-red)',
                fontSize: 13,
                marginTop: 8,
              }}>
                {error}
              </div>
            )}
          </div>

          {/* Right: Analytics Sidebar */}
          <div className="live-sidebar">
            <WorkerPanel
              workers={frameProcessor.trackedWorkers}
              activeViolations={safety?.active_violations ?? 0}
              ppeCompliance={safety?.ppe_compliance_percentage ?? 0}
            />
            <OverlayControls settings={overlaySettings} onChange={setOverlaySettings} />
          </div>
        </div>
      </div>
    </>
  );
}
