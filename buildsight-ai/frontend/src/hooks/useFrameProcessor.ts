import { useRef, useState, useCallback, useEffect } from 'react';
import type { AnalyticsMessage, TrackedWorker } from '../types';

const DEFAULT_FPS = 10;
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';

interface FrameProcessorOptions {
  fps?: number;
  onAnalytics?: (data: AnalyticsMessage) => void;
}

export function useFrameProcessor(options: FrameProcessorOptions = {}) {
  const { fps = DEFAULT_FPS, onAnalytics } = options;

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const frameWsRef = useRef<WebSocket | null>(null);
  const analyticsWsRef = useRef<WebSocket | null>(null);
  const captureIntervalRef = useRef<number | null>(null);
  const fpsCounterRef = useRef({ frames: 0, startTime: performance.now() });

  const [isCapturing, setIsCapturing] = useState(false);
  const [captureFps, setCaptureFps] = useState(0);
  const [trackedWorkers, setTrackedWorkers] = useState<TrackedWorker[]>([]);
  const [latestAnalytics, setLatestAnalytics] = useState<AnalyticsMessage | null>(null);
  const [frameWsStatus, setFrameWsStatus] = useState<string>('disconnected');

  // Connect frame WebSocket
  const connectFrameWs = useCallback(() => {
    if (frameWsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(`${WS_URL}/ws/frames`);
    frameWsRef.current = ws;

    ws.onopen = () => setFrameWsStatus('connected');
    ws.onclose = () => setFrameWsStatus('disconnected');
    ws.onerror = () => setFrameWsStatus('error');
  }, []);

  // Connect analytics WebSocket
  const connectAnalyticsWs = useCallback(() => {
    if (analyticsWsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(`${WS_URL}/ws/analytics`);
    analyticsWsRef.current = ws;

    ws.onopen = () => {
      // Start processing on backend
      ws.send(JSON.stringify({ type: 'start_processing', source: 'webcam' }));
    };

    ws.onmessage = (event) => {
      try {
        const data: AnalyticsMessage = JSON.parse(event.data);
        if (data.type === 'analytics_update') {
          setLatestAnalytics(data);
          setTrackedWorkers(data.tracked_workers || []);
          onAnalytics?.(data);
        }
      } catch {
        // Ignore non-JSON
      }
    };

    ws.onclose = () => {
      // Reconnect after delay
      setTimeout(() => connectAnalyticsWs(), 2000);
    };
  }, [onAnalytics]);

  // Capture frame from video element
  const captureFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ws = frameWsRef.current;

    if (!video || !canvas || !ws || ws.readyState !== WebSocket.OPEN) return;
    if (video.readyState < 2) return; // Not enough data

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size to match full native video resolution
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;

    // Enable high-fidelity bilinear image scaling
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    // Draw current frame at full clarity
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to high-definition base64 JPEG (92% quality)
    const frameData = canvas.toDataURL('image/jpeg', 0.92);

    // Update FPS counter
    fpsCounterRef.current.frames++;
    const elapsed = (performance.now() - fpsCounterRef.current.startTime) / 1000;
    if (elapsed >= 1) {
      setCaptureFps(Math.round(fpsCounterRef.current.frames / elapsed));
      fpsCounterRef.current = { frames: 0, startTime: performance.now() };
    }

    // Send frame
    try {
      ws.send(JSON.stringify({
        frame: frameData,
        capture_fps: captureFps,
      }));
    } catch {
      // WebSocket buffer full, skip frame
    }
  }, [captureFps]);

  // Start capturing frames
  const startCapture = useCallback((videoElement: HTMLVideoElement) => {
    videoRef.current = videoElement;

    // Create hidden canvas for frame capture
    if (!canvasRef.current) {
      canvasRef.current = document.createElement('canvas');
    }

    // Connect WebSockets
    connectFrameWs();
    connectAnalyticsWs();

    // Start capture interval
    const interval = 1000 / fps;
    captureIntervalRef.current = window.setInterval(captureFrame, interval);
    setIsCapturing(true);
  }, [fps, captureFrame, connectFrameWs, connectAnalyticsWs]);

  // Stop capturing
  const stopCapture = useCallback(() => {
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }

    // Stop processing on backend
    if (analyticsWsRef.current?.readyState === WebSocket.OPEN) {
      analyticsWsRef.current.send(JSON.stringify({ type: 'stop_processing' }));
    }

    // Close WebSockets
    frameWsRef.current?.close();
    analyticsWsRef.current?.close();
    frameWsRef.current = null;
    analyticsWsRef.current = null;

    setIsCapturing(false);
    setCaptureFps(0);
    setTrackedWorkers([]);
    setLatestAnalytics(null);
    setFrameWsStatus('disconnected');
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current);
      }
      frameWsRef.current?.close();
      analyticsWsRef.current?.close();
    };
  }, []);

  return {
    isCapturing,
    captureFps,
    trackedWorkers,
    latestAnalytics,
    frameWsStatus,
    startCapture,
    stopCapture,
  };
}
