"""BuildSight AI — FastAPI Application Entry Point (Complete)"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager

# pyrefly: ignore [missing-import]
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database.session import init_db
from app.services.video_processor import video_processor
from app.services.websocket_manager import ws_manager
from app.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown."""
    logger.info("=" * 60)
    logger.info("  BuildSight AI — Starting Up")
    logger.info("=" * 60)

    # 1. Initialize database
    init_db()

    # 2. Auto-download models (non-blocking via executor)
    from app.ai.model_downloader import download_all_models
    loop = asyncio.get_event_loop()
    model_paths = await loop.run_in_executor(None, download_all_models)

    # Apply downloaded model paths to video_processor before initialize()
    if model_paths.get("ppe_model_path"):
        video_processor.ppe_detector.model_path = model_paths["ppe_model_path"]
        logger.info(f"  ✓ PPE model path set: {model_paths['ppe_model_path']}")

    # 3. Initialize all AI models
    status = video_processor.initialize()
    for model_name, model_status in status.items():
        loaded = model_status.get("loaded", False)
        icon = "✓" if loaded else "ℹ"
        detail = model_status.get("model") or model_status.get("error", "")
        logger.info(f"  {icon} {model_name}: {detail}")

    # 4. Start background processing
    await video_processor.start_processing(source="webcam")
    logger.info("  ✓ Video processing pipeline ready")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("BuildSight AI — Shutting down...")
    await video_processor.stop_processing()
    from app.database.mongodb import close_mongo_connection
    close_mongo_connection()


app = FastAPI(
    title="BuildSight AI",
    description="AI-Powered Construction Site Intelligence",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os
# pyrefly: ignore [missing-import]
from fastapi.staticfiles import StaticFiles

# Ensure storage directories exist
os.makedirs(settings.evidence_dir, exist_ok=True)
os.makedirs(settings.profile_images_dir, exist_ok=True)
os.makedirs("data/photos", exist_ok=True)
os.makedirs("data/snapshots", exist_ok=True)

app.mount("/data/evidence", StaticFiles(directory=settings.evidence_dir), name="evidence")
app.mount("/data/profiles", StaticFiles(directory=settings.profile_images_dir), name="profiles")
app.mount("/data/photos", StaticFiles(directory="data/photos"), name="photos")
app.mount("/data/snapshots", StaticFiles(directory="data/snapshots"), name="snapshots")

app.include_router(router)


# ── Analytics WebSocket ───────────────────────────────────────────

@app.websocket("/ws/analytics")
async def analytics_ws(websocket: WebSocket):
    """Clients connect here to receive real-time analytics pushes."""
    await ws_manager.connect(websocket)
    logger.info("Analytics WS connected")
    try:
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                msg = json.loads(raw)

                if msg.get("type") == "start_processing":
                    source = msg.get("source", "webcam")
                    if not video_processor.is_processing:
                        await video_processor.start_processing(source=source)

                elif msg.get("type") == "stop_processing":
                    await video_processor.stop_processing()

                elif msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

            except asyncio.TimeoutError:
                try:
                    await websocket.send_json({"type": "keepalive"})
                except Exception:
                    break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"Analytics WS error: {e}")
    finally:
        await ws_manager.disconnect(websocket)
        logger.info("Analytics WS disconnected")


# ── Frame Ingestion WebSocket ─────────────────────────────────────

@app.websocket("/ws/frames")
async def frames_ws(websocket: WebSocket):
    """Browser sends base64 JPEG frames here for real-time AI analysis."""
    await websocket.accept()
    logger.info("Frame WS connected")
    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            frame_data = msg.get("frame", "")
            capture_fps = float(msg.get("capture_fps", 0))

            if frame_data:
                await video_processor.submit_frame(frame_data, capture_fps)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"Frame WS error: {e}")
    finally:
        logger.info("Frame WS disconnected")
