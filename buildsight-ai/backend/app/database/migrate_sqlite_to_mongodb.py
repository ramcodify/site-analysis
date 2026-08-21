"""BuildSight AI — SQLite to MongoDB Data Migration Utility"""

import os
import sqlite3
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from app.database.mongodb import get_db, init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SQLITE_PATH = Path(__file__).resolve().parents[2] / "buildsight.db"


def migrate_sqlite_to_mongodb():
    """Migrate records from SQLite to MongoDB if SQLite file exists."""
    init_db()
    db = get_db()

    if not SQLITE_PATH.exists():
        logger.info("No existing SQLite database found to migrate.")
        return

    logger.info(f"Migrating SQLite data from {SQLITE_PATH} to MongoDB database '{db.name}'...")
    conn = sqlite3.connect(str(SQLITE_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 1. Migrate Registered Workers
    try:
        cursor.execute("SELECT * FROM registered_workers")
        rows = cursor.fetchall()
        for r in rows:
            wcode = r["worker_code"]
            embs = json.loads(r["biometric_embeddings"]) if r["biometric_embeddings"] else []
            existing = db.registered_workers.find_one({"worker_code": wcode})
            if not existing:
                db.registered_workers.insert_one({
                    "worker_code": wcode,
                    "name": r["name"],
                    "employee_number": r["employee_number"],
                    "department": r["department"],
                    "role": r["role"],
                    "profile_image_path": r["profile_image_path"],
                    "biometric_embeddings": embs,
                    "registration_date": datetime.now(timezone.utc),
                    "active_status": r["active_status"] or "ACTIVE",
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                })
        logger.info(f"✓ Migrated {len(rows)} registered workers")
    except Exception as e:
        logger.debug(f"Registered workers migration notice: {e}")

    # 2. Migrate Danger Zones
    try:
        cursor.execute("SELECT * FROM danger_zones")
        rows = cursor.fetchall()
        for r in rows:
            poly = json.loads(r["polygon_data"]) if r["polygon_data"] else []
            zid = r["zone_id"]
            db.danger_zones.replace_one(
                {"zone_id": zid},
                {
                    "zone_id": zid,
                    "name": r["name"],
                    "zone_type": r["zone_type"],
                    "polygon_data": poly,
                    "risk_weight": r["risk_weight"] or 30.0,
                },
                upsert=True
            )
        logger.info(f"✓ Migrated {len(rows)} danger zones")
    except Exception as e:
        logger.debug(f"Danger zones migration notice: {e}")

    conn.close()
    logger.info("✓ SQLite to MongoDB migration complete!")


if __name__ == "__main__":
    migrate_sqlite_to_mongodb()
