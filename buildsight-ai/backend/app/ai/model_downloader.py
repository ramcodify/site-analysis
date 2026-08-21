"""BuildSight AI — Model Auto-Downloader

Downloads AI models from Hugging Face Hub on first run.
Models are cached to data/models/ for offline use.
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "models"


def _model_path(filename: str) -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return str(DATA_DIR / filename)


def download_ppe_model() -> str:
    """
    Download a real PPE detection model from Hugging Face Hub.

    Uses keremberke/yolov8n-hard-hat-detection — a YOLOv8n model
    trained on the Hard Hat Workers dataset detecting:
      0: hardhat, 1: NO-Hardhat, 2: NO-Safety Vest, 3: NO-mask,
      4: Person, 5: Safety Cone, 6: Safety Vest/Jacket, 7: machinery, 8: vehicle

    No API key required. MIT license.
    """
    local_path = _model_path("ppe_model.pt")

    if os.path.exists(local_path) and os.path.getsize(local_path) > 1_000_000:
        logger.info(f"✓ PPE model found (cached): {local_path}")
        return local_path

    logger.info("⬇  Downloading PPE model from Hugging Face Hub...")
    logger.info("   Model: keremberke/yolov8n-hard-hat-detection")

    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id="keremberke/yolov8n-hard-hat-detection",
            filename="best.pt",
            local_dir=str(DATA_DIR),
            local_dir_use_symlinks=False,
        )
        # Move / copy to expected path
        import shutil
        shutil.copy2(downloaded, local_path)
        size_mb = os.path.getsize(local_path) / 1_048_576
        logger.info(f"✓ PPE model downloaded ({size_mb:.1f} MB) → {local_path}")
        return local_path
    except Exception as e:
        logger.error(f"PPE model download failed: {e}")
        return ""


def download_all_models() -> dict:
    """Download all required models and return their paths."""
    results = {}

    # PPE Model
    ppe_path = download_ppe_model()
    results["ppe_model_path"] = ppe_path

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    paths = download_all_models()
    for k, v in paths.items():
        print(f"{k}: {v}")
