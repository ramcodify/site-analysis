import { useRef, useState, useCallback, useEffect } from 'react';

interface WebcamOptions {
  defaultFacingMode?: 'user' | 'environment';
}

interface CameraDevice {
  deviceId: string;
  label: string;
}

export type Resolution = { width: number; height: number; label: string };

export const RESOLUTIONS: Resolution[] = [
  { width: 1920, height: 1080, label: '1080p (Full HD)' },
  { width: 1280, height: 720, label: '720p (HD)' },
  { width: 2560, height: 1440, label: '2K (QHD)' },
  { width: 640, height: 480, label: '480p (Fast)' },
];

export function useWebcam(options: WebcamOptions = {}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [isActive, setIsActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cameras, setCameras] = useState<CameraDevice[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<string>('');
  const [selectedResolution, setSelectedResolution] = useState<Resolution>(RESOLUTIONS[0]); // Default to 1080p Full HD

  // Enumerate cameras
  const enumerateCameras = useCallback(async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices
        .filter((d) => d.kind === 'videoinput')
        .map((d, i) => ({
          deviceId: d.deviceId,
          label: d.label || `HD Camera ${i + 1}`,
        }));
      setCameras(videoDevices);
      if (videoDevices.length > 0 && !selectedCamera) {
        setSelectedCamera(videoDevices[0].deviceId);
      }
    } catch {
      setError('Cannot enumerate cameras');
    }
  }, [selectedCamera]);

  // Start webcam with high-definition optical constraints
  const start = useCallback(async () => {
    setError(null);

    try {
      // Request permission first if no cameras listed
      if (cameras.length === 0) {
        const tempStream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1920 },
            height: { ideal: 1080 },
          },
        });
        tempStream.getTracks().forEach((t) => t.stop());
        await enumerateCameras();
      }

      const constraints: MediaStreamConstraints = {
        video: {
          deviceId: selectedCamera ? { exact: selectedCamera } : undefined,
          width: { ideal: selectedResolution.width, min: Math.min(selectedResolution.width, 640) },
          height: { ideal: selectedResolution.height, min: Math.min(selectedResolution.height, 480) },
          frameRate: { ideal: 30, max: 60 },
          aspectRatio: { ideal: 1.7777777778 },
          facingMode: options.defaultFacingMode || 'user',
        },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.setAttribute('playsinline', 'true');
        await videoRef.current.play();
      }

      setIsActive(true);

      // Re-enumerate after permission grant
      await enumerateCameras();
    } catch (err: unknown) {
      const error = err as Error;
      if (error.name === 'NotAllowedError') {
        setError('Camera permission denied. Please allow camera access in your browser settings.');
      } else if (error.name === 'NotFoundError') {
        setError('No camera found. Please connect a camera and try again.');
      } else if (error.name === 'NotReadableError') {
        setError('Camera is in use by another application.');
      } else {
        // Fallback to basic constraints if HD resolution fails on specific hardware
        try {
          const fallbackStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
          streamRef.current = fallbackStream;
          if (videoRef.current) {
            videoRef.current.srcObject = fallbackStream;
            await videoRef.current.play();
          }
          setIsActive(true);
        } catch {
          setError(`Camera error: ${error.message}`);
          setIsActive(false);
        }
      }
    }
  }, [cameras.length, selectedCamera, selectedResolution, options.defaultFacingMode, enumerateCameras]);

  // Stop webcam
  const stop = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsActive(false);
  }, []);

  // Restart with new settings
  const restart = useCallback(async () => {
    stop();
    await new Promise((r) => setTimeout(r, 200));
    await start();
  }, [stop, start]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
    };
  }, []);

  // Initial camera enumeration
  useEffect(() => {
    enumerateCameras();
  }, [enumerateCameras]);

  return {
    videoRef,
    isActive,
    error,
    cameras,
    selectedCamera,
    setSelectedCamera,
    resolutions: RESOLUTIONS,
    selectedResolution,
    setSelectedResolution,
    start,
    stop,
    restart,
  };
}
