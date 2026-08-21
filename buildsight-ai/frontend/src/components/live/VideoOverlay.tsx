import { useEffect, useRef, useCallback } from 'react';
import type { TrackedWorker, OverlaySettings } from '../../types';

interface VideoOverlayProps {
  videoElement: HTMLVideoElement | null;
  workers: TrackedWorker[];
  overlaySettings: OverlaySettings;
  isMirrored?: boolean;
}

const RISK_COLORS: Record<string, string> = {
  SAFE: '#10b981',
  LOW: '#3b82f6',
  MEDIUM: '#f59e0b',
  HIGH: '#f97316',
  CRITICAL: '#ef4444',
};

export function VideoOverlay({ videoElement, workers, overlaySettings, isMirrored = false }: VideoOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number>(0);

  const drawOverlays = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoElement;
    if (!canvas || !video) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Match canvas to video display size with high-DPI crisp scaling
    const rect = video.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    // Calculate scale factors and letterbox offsets
    const videoWidth = video.videoWidth || 1280;
    const videoHeight = video.videoHeight || 720;

    let renderWidth = rect.width;
    let renderHeight = rect.height;
    let offsetX = 0;
    let offsetY = 0;

    if (rect.width > 0 && rect.height > 0 && videoWidth > 0 && videoHeight > 0) {
      const containerRatio = rect.width / rect.height;
      const videoRatio = videoWidth / videoHeight;

      if (containerRatio > videoRatio) {
        // Pillarboxed (black bars left/right)
        renderWidth = rect.height * videoRatio;
        offsetX = (rect.width - renderWidth) / 2;
      } else {
        // Letterboxed (black bars top/bottom)
        renderHeight = rect.width / videoRatio;
        offsetY = (rect.height - renderHeight) / 2;
      }
    }

    const scaleX = renderWidth / videoWidth;
    const scaleY = renderHeight / videoHeight;

    ctx.clearRect(0, 0, rect.width, rect.height);

    for (const worker of workers) {
      const {
        bbox, worker_id, confidence, helmet, vest, risk_level,
        permanent_worker_id, name, identity_status, face_bbox,
      } = worker;

      const isRegistered = identity_status === 'REGISTERED' && permanent_worker_id;
      const color = RISK_COLORS[risk_level] || RISK_COLORS.SAFE;

      const rawX1 = offsetX + bbox.x1 * scaleX;
      const rawX2 = offsetX + bbox.x2 * scaleX;
      const x1 = isMirrored ? (rect.width - rawX2) : rawX1;
      const x2 = isMirrored ? (rect.width - rawX1) : rawX2;
      const y1 = offsetY + bbox.y1 * scaleY;
      const y2 = offsetY + bbox.y2 * scaleY;
      const w = x2 - x1;
      const h = y2 - y1;

      // 1. Person Bounding Box
      if (overlaySettings.showBoundingBoxes) {
        ctx.strokeStyle = color;
        ctx.lineWidth = isRegistered ? 2.5 : 1.8;
        ctx.strokeRect(x1, y1, w, h);

        // Corner accents
        const cornerLen = Math.min(14, w / 4, h / 4);
        ctx.lineWidth = 3;
        // Top-left
        ctx.beginPath(); ctx.moveTo(x1, y1 + cornerLen); ctx.lineTo(x1, y1); ctx.lineTo(x1 + cornerLen, y1); ctx.stroke();
        // Top-right
        ctx.beginPath(); ctx.moveTo(x2 - cornerLen, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + cornerLen); ctx.stroke();
        // Bottom-left
        ctx.beginPath(); ctx.moveTo(x1, y2 - cornerLen); ctx.lineTo(x1, y2); ctx.lineTo(x1 + cornerLen, y2); ctx.stroke();
        // Bottom-right
        ctx.beginPath(); ctx.moveTo(x2 - cornerLen, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2 - cornerLen, y2); ctx.stroke();
      }

      // 2. Face Bounding Box & Face Worker ID Tag (if detected)
      if (face_bbox && (overlaySettings.showFaceBoxes ?? true)) {
        const rawFx1 = offsetX + face_bbox.x1 * scaleX;
        const rawFx2 = offsetX + face_bbox.x2 * scaleX;
        const fx1 = isMirrored ? (rect.width - rawFx2) : rawFx1;
        const fx2 = isMirrored ? (rect.width - rawFx1) : rawFx2;
        const fy1 = offsetY + face_bbox.y1 * scaleY;
        const fy2 = offsetY + face_bbox.y2 * scaleY;
        const fw = fx2 - fx1;
        const fh = fy2 - fy1;

        ctx.strokeStyle = isRegistered ? '#06b6d4' : 'rgba(255, 255, 255, 0.7)';
        ctx.lineWidth = 2.0;
        ctx.setLineDash(isRegistered ? [] : [3, 3]);
        ctx.strokeRect(fx1, fy1, fw, fh);
        ctx.setLineDash([]);

        // Small Face Tag above face box
        const faceTag = isRegistered ? `✓ ${permanent_worker_id}` : `? Face`;
        ctx.font = 'bold 10px Inter, sans-serif';
        const ftMetrics = ctx.measureText(faceTag);
        const ftW = ftMetrics.width + 8;
        const ftH = 15;

        ctx.fillStyle = isRegistered ? 'rgba(8, 145, 178, 0.95)' : 'rgba(51, 65, 85, 0.85)';
        ctx.fillRect(fx1, Math.max(0, fy1 - ftH - 2), ftW, ftH);
        ctx.fillStyle = '#ffffff';
        ctx.fillText(faceTag, fx1 + 4, Math.max(11, fy1 - 4));
      }

      // 3. Permanent Worker ID + Track ID Top Banner
      let idLabel = '';
      if (isRegistered) {
        idLabel = `🆔 ${permanent_worker_id}${name ? ` · ${name}` : ''} (Track #${worker_id})`;
      } else {
        idLabel = `❓ UNKNOWN · Track #${worker_id}`;
      }

      if (overlaySettings.showConfidence) {
        idLabel += ` · ${(confidence * 100).toFixed(0)}%`;
      }

      ctx.font = 'bold 11px Inter, sans-serif';
      const metrics = ctx.measureText(idLabel);
      const labelW = metrics.width + 16;
      const labelH = 22;

      // Header Tag Background
      ctx.fillStyle = isRegistered ? '#0f766e' : '#374151';
      ctx.globalAlpha = 0.95;
      ctx.fillRect(x1, Math.max(0, y1 - labelH - 3), labelW, labelH);
      ctx.globalAlpha = 1.0;

      // Left Color Marker
      ctx.fillStyle = isRegistered ? '#06b6d4' : color;
      ctx.fillRect(x1, Math.max(0, y1 - labelH - 3), 4, labelH);

      // Text
      ctx.fillStyle = '#ffffff';
      ctx.fillText(idLabel, x1 + 8, Math.max(14, y1 - 8));

      // 4. PPE & Risk Status below box
      if (overlaySettings.showPPEStatus && (helmet !== null || vest !== null || worker.gloves !== null || worker.face_mask !== null)) {
        const ppeItems: string[] = [];
        if (helmet !== null) ppeItems.push(helmet ? '🪖 Helmet ✓' : '🪖 No Helmet ✗');
        if (vest !== null) ppeItems.push(vest ? '🦺 Vest ✓' : '🦺 No Vest ✗');
        if (worker.gloves !== null && worker.gloves !== undefined) ppeItems.push(worker.gloves ? '🧤 Gloves ✓' : '🧤 No Gloves ✗');
        if (worker.face_mask !== null && worker.face_mask !== undefined) ppeItems.push(worker.face_mask ? '😷 Mask ✓' : '😷 No Mask ✗');

        if (overlaySettings.showRiskLabels) {
          ppeItems.push(`Risk: ${risk_level}`);
        }

        const ppeText = ppeItems.join('  |  ');
        ctx.font = 'bold 11px Inter, sans-serif';
        const ppeMetrics = ctx.measureText(ppeText);
        const ppeW = ppeMetrics.width + 16;
        const ppeH = 20;

        ctx.fillStyle = 'rgba(15, 23, 42, 0.90)';
        ctx.fillRect(x1, y2 + 3, ppeW, ppeH);

        // Render with color highlighting
        ctx.fillStyle = '#f8fafc';
        ctx.fillText(ppeText, x1 + 8, y2 + 17);
      }
    }

    animFrameRef.current = requestAnimationFrame(drawOverlays);
  }, [videoElement, workers, overlaySettings, isMirrored]);

  useEffect(() => {
    animFrameRef.current = requestAnimationFrame(drawOverlays);
    return () => {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, [drawOverlays]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
      }}
    />
  );
}
