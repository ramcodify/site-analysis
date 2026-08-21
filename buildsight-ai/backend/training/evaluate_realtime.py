"""BuildSight AI — Real-Time Inference Latency and Hardware Performance Benchmark

Benchmarks:
  - System hardware specifications (CPU, RAM, OS, GPU/CUDA)
  - Latency across multiple video resolutions (640x480, 1280x720, 1920x1080)
  - Average, median, and P95 latency per inference
  - End-to-end processing FPS

Outputs:
  - backend/experiments/realtime_benchmark.json
"""

import os
import sys
import time
import json
import platform
import logging
from pathlib import Path
import psutil
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "data" / "models"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

PPE_MODEL_PATH = MODELS_DIR / "ppe_model.pt"


def get_hardware_info():
    """Extract actual host system hardware metrics."""
    cpu_name = platform.processor() or "x86_64"
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    cpu_name = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    os_info = f"{platform.system()} {platform.release()}"

    gpu_info = "None (CPU Execution)"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return {
        "cpu_model": cpu_name,
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_total_gb": ram_gb,
        "operating_system": os_info,
        "python_version": sys.version.split()[0],
        "accelerator": gpu_info,
    }


def benchmark_pipeline(n_warmup=10, n_trials=50):
    """Run measured latency benchmarks across resolutions."""
    from ultralytics import YOLO

    hardware = get_hardware_info()
    logger.info(f"Hardware Detected: CPU={hardware['cpu_model']} | RAM={hardware['ram_total_gb']}GB | OS={hardware['operating_system']}")

    if not PPE_MODEL_PATH.exists():
        logger.error(f"PPE model not found at {PPE_MODEL_PATH}")
        return None

    logger.info(f"Loading model: {PPE_MODEL_PATH}")
    model = YOLO(str(PPE_MODEL_PATH))

    resolutions = [
        ("640x480 (SD)", (480, 640, 3)),
        ("1280x720 (HD)", (720, 1280, 3)),
        ("1920x1080 (FHD)", (1080, 1920, 3)),
    ]

    benchmark_results = {}

    for res_name, (h, w, c) in resolutions:
        logger.info(f"Benchmarking resolution: {res_name} ({w}x{h})...")
        dummy_frame = np.random.randint(0, 255, (h, w, c), dtype=np.uint8)

        # Warmup
        for _ in range(n_warmup):
            _ = model(dummy_frame, verbose=False)

        latencies_ms = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            _ = model(dummy_frame, verbose=False)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        latencies_ms = np.array(latencies_ms)
        avg_lat = float(np.mean(latencies_ms))
        med_lat = float(np.median(latencies_ms))
        p95_lat = float(np.percentile(latencies_ms, 95))
        min_lat = float(np.min(latencies_ms))
        max_lat = float(np.max(latencies_ms))
        fps = float(1000.0 / avg_lat) if avg_lat > 0 else 0.0

        benchmark_results[res_name] = {
            "resolution": f"{w}x{h}",
            "trials_measured": n_trials,
            "average_latency_ms": round(avg_lat, 2),
            "median_latency_ms": round(med_lat, 2),
            "p95_latency_ms": round(p95_lat, 2),
            "min_latency_ms": round(min_lat, 2),
            "max_latency_ms": round(max_lat, 2),
            "measured_fps": round(fps, 1),
        }
        logger.info(f"  ✓ {res_name} -> Avg: {avg_lat:.2f}ms | Median: {med_lat:.2f}ms | P95: {p95_lat:.2f}ms | FPS: {fps:.1f}")

    output_data = {
        "hardware": hardware,
        "model_file": str(PPE_MODEL_PATH),
        "model_size_mb": round(os.path.getsize(str(PPE_MODEL_PATH)) / (1024 * 1024), 2),
        "benchmarks": benchmark_results,
    }

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = EXPERIMENTS_DIR / "realtime_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"✓ Benchmark report saved to {out_file}")
    return output_data


if __name__ == "__main__":
    benchmark_pipeline()
