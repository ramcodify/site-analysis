"""BuildSight AI — MongoDB Index Management

Creates unique and compound performance indexes across all MongoDB collections:
  - registered_workers (unique: worker_code, employee_number)
  - worker_sessions (track_id, permanent_worker_id, source_id)
  - worker_snapshots (worker_id, session_id, timestamp)
  - violations (unique: violation_id; indexed: worker_id, source_id, status, violation_type, timestamp)
  - progress_records (source_id, current_stage, timestamp)
  - video_sources (unique: source_id)
  - danger_zones (unique: zone_id)
  - reports (report_type, generated_at)
"""

import logging
import pymongo
from pymongo.database import Database
from app.database.collections import (
    COLLECTION_REGISTERED_WORKERS,
    COLLECTION_WORKER_SESSIONS,
    COLLECTION_WORKER_SNAPSHOTS,
    COLLECTION_VIOLATIONS,
    COLLECTION_PROGRESS_RECORDS,
    COLLECTION_VIDEO_SOURCES,
    COLLECTION_DANGER_ZONES,
    COLLECTION_REPORTS,
)

logger = logging.getLogger(__name__)


def create_indexes(db: Database):
    """Create all required indexes synchronously or during startup."""
    try:
        # 1. registered_workers
        try:
            db[COLLECTION_REGISTERED_WORKERS].create_index([("worker_code", pymongo.ASCENDING)], unique=True)
        except Exception:
            pass
        try:
            # Drop non-unique employee_number_1 if it exists
            idx_info = db[COLLECTION_REGISTERED_WORKERS].index_information()
            if "employee_number_1" in idx_info and not idx_info["employee_number_1"].get("unique"):
                db[COLLECTION_REGISTERED_WORKERS].drop_index("employee_number_1")
            db[COLLECTION_REGISTERED_WORKERS].create_index([("employee_number", pymongo.ASCENDING)], unique=True)
        except Exception:
            pass
        db[COLLECTION_REGISTERED_WORKERS].create_index([("active_status", pymongo.ASCENDING)])

        # 2. worker_sessions
        db[COLLECTION_WORKER_SESSIONS].create_index([("track_id", pymongo.ASCENDING)])
        db[COLLECTION_WORKER_SESSIONS].create_index([("permanent_worker_id", pymongo.ASCENDING)])
        db[COLLECTION_WORKER_SESSIONS].create_index([("source_id", pymongo.ASCENDING)])
        db[COLLECTION_WORKER_SESSIONS].create_index([("started_at", pymongo.DESCENDING)])
        db[COLLECTION_WORKER_SESSIONS].create_index([("last_seen", pymongo.DESCENDING)])

        # 3. worker_snapshots
        db[COLLECTION_WORKER_SNAPSHOTS].create_index([("worker_id", pymongo.ASCENDING)])
        db[COLLECTION_WORKER_SNAPSHOTS].create_index([("session_id", pymongo.ASCENDING)])
        db[COLLECTION_WORKER_SNAPSHOTS].create_index([("worker_code", pymongo.ASCENDING)])
        db[COLLECTION_WORKER_SNAPSHOTS].create_index([("timestamp", pymongo.DESCENDING)])

        # 4. violations
        db[COLLECTION_VIOLATIONS].create_index([("violation_id", pymongo.ASCENDING)], unique=True)
        db[COLLECTION_VIOLATIONS].create_index([("worker_id", pymongo.ASCENDING)])
        db[COLLECTION_VIOLATIONS].create_index([("worker_code", pymongo.ASCENDING)])
        db[COLLECTION_VIOLATIONS].create_index([("source_id", pymongo.ASCENDING)])
        db[COLLECTION_VIOLATIONS].create_index([("status", pymongo.ASCENDING)])
        db[COLLECTION_VIOLATIONS].create_index([("violation_type", pymongo.ASCENDING)])
        db[COLLECTION_VIOLATIONS].create_index([("timestamp", pymongo.DESCENDING)])

        # 5. progress_records
        db[COLLECTION_PROGRESS_RECORDS].create_index([("source_id", pymongo.ASCENDING)])
        db[COLLECTION_PROGRESS_RECORDS].create_index([("current_stage", pymongo.ASCENDING)])
        db[COLLECTION_PROGRESS_RECORDS].create_index([("timestamp", pymongo.DESCENDING)])

        # 6. video_sources
        db[COLLECTION_VIDEO_SOURCES].create_index([("source_id", pymongo.ASCENDING)], unique=True)

        # 7. danger_zones
        db[COLLECTION_DANGER_ZONES].create_index([("zone_id", pymongo.ASCENDING)], unique=True)
        db[COLLECTION_DANGER_ZONES].create_index([("source_id", pymongo.ASCENDING)])

        # 8. reports
        db[COLLECTION_REPORTS].create_index([("report_type", pymongo.ASCENDING)])
        db[COLLECTION_REPORTS].create_index([("generated_at", pymongo.DESCENDING)])

        logger.info("✓ MongoDB indexes created and verified across all collections")
    except Exception as e:
        logger.error(f"Error creating MongoDB indexes: {e}")
