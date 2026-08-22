"""BuildSight AI — MongoDB Repository & Aggregation Layer

Central persistent database layer powered by MongoDB:
  - registered_workers
  - worker_sessions
  - worker_snapshots
  - violations
  - progress_records
  - video_sources
  - danger_zones
  - reports
"""

import os
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
# pyrefly: ignore [missing-import]
import pymongo

from app.config import settings
from app.database.mongodb import get_db, get_async_db
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
from app.database.utils import serialize_mongo_doc, serialize_mongo_docs, to_object_id

logger = logging.getLogger(__name__)


# ── 1. Registered Worker Repository ───────────────────────────────

class RegisteredWorkerRepository:
    """CRUD, Biometric Cache, and Aggregation for permanent registered workers."""

    @staticmethod
    def get_next_worker_code() -> str:
        """Generate the next sequential permanent worker ID (e.g., W001, W002)."""
        db = get_db()
        cursor = db[COLLECTION_REGISTERED_WORKERS].find({}, {"worker_code": 1})
        max_num = 0
        for doc in cursor:
            code = doc.get("worker_code", "")
            if code and code.startswith("W"):
                try:
                    num = int(code[1:])
                    if num > max_num:
                        max_num = num
                except ValueError:
                    pass
        return f"W{max_num + 1:03d}"

    @staticmethod
    def get_next_employee_number() -> str:
        """Generate the next sequential employee ID starting from EMP-001."""
        db = get_db()
        cursor = db[COLLECTION_REGISTERED_WORKERS].find({}, {"employee_number": 1})
        max_num = 0
        for doc in cursor:
            emp = str(doc.get("employee_number", ""))
            digits = "".join(filter(str.isdigit, emp))
            if digits:
                try:
                    num = int(digits)
                    if num > max_num:
                        max_num = num
                except ValueError:
                    pass
        return f"EMP-{max_num + 1:03d}"

    @staticmethod
    def create(
        name: str,
        employee_number: str,
        department: str,
        role: str,
        embeddings: List[List[float]],
        worker_code: Optional[str] = None,
        profile_image_path: Optional[str] = None,
        active_status: str = "ACTIVE",
    ) -> Dict[str, Any]:
        """Register a new permanent worker with multi-sample biometric embeddings."""
        db = get_db()
        code = worker_code or RegisteredWorkerRepository.get_next_worker_code()

        # Enforce unique worker_code and employee_number
        existing = db[COLLECTION_REGISTERED_WORKERS].find_one({
            "$or": [{"worker_code": code}, {"employee_number": employee_number}]
        })
        if existing:
            raise ValueError(f"Worker with code {code} or employee number {employee_number} already exists")

        now = datetime.now(timezone.utc)
        doc = {
            "worker_code": code,
            "name": name,
            "employee_number": employee_number,
            "department": department,
            "role": role,
            "profile_image_path": profile_image_path,
            "biometric_embeddings": embeddings,
            "registration_date": now,
            "active_status": active_status,
            "last_recognized": None,
            "created_at": now,
            "updated_at": now,
        }
        res = db[COLLECTION_REGISTERED_WORKERS].insert_one(doc)

        return {
            "id": str(res.inserted_id),
            "worker_code": code,
            "name": name,
            "employee_number": employee_number,
            "department": department,
            "role": role,
            "profile_image_path": profile_image_path,
            "registration_date": now.isoformat(),
            "active_status": active_status,
            "total_embeddings": len(embeddings),
        }

    @staticmethod
    def get_all(active_only: bool = False) -> List[Dict[str, Any]]:
        """List all registered workers (never exposes raw biometric embeddings to frontend)."""
        db = get_db()
        query = {"active_status": "ACTIVE"} if active_only else {}
        cursor = db[COLLECTION_REGISTERED_WORKERS].find(query).sort("worker_code", pymongo.ASCENDING)

        results = []
        for doc in cursor:
            embs = doc.get("biometric_embeddings", [])
            embs_count = len(embs) if isinstance(embs, list) else 0
            reg_date = doc.get("registration_date")
            created = doc.get("created_at")
            updated = doc.get("updated_at")

            results.append({
                "id": str(doc["_id"]),
                "worker_code": doc.get("worker_code"),
                "name": doc.get("name"),
                "employee_number": doc.get("employee_number"),
                "department": doc.get("department"),
                "role": doc.get("role"),
                "profile_image_path": doc.get("profile_image_path"),
                "registration_date": reg_date.isoformat() if isinstance(reg_date, datetime) else (str(reg_date) if reg_date else None),
                "active_status": doc.get("active_status", "ACTIVE"),
                "created_at": created.isoformat() if isinstance(created, datetime) else (str(created) if created else None),
                "updated_at": updated.isoformat() if isinstance(updated, datetime) else (str(updated) if updated else None),
                "total_embeddings": embs_count,
            })
        return results

    @staticmethod
    def get_by_code(worker_code: str) -> Optional[Dict[str, Any]]:
        """Get registered worker metadata by worker_code (no raw embeddings)."""
        db = get_db()
        doc = db[COLLECTION_REGISTERED_WORKERS].find_one({"worker_code": worker_code})
        if not doc:
            return None

        embs = doc.get("biometric_embeddings", [])
        embs_count = len(embs) if isinstance(embs, list) else 0
        reg_date = doc.get("registration_date")
        created = doc.get("created_at")
        updated = doc.get("updated_at")

        return {
            "id": str(doc["_id"]),
            "worker_code": doc.get("worker_code"),
            "name": doc.get("name"),
            "employee_number": doc.get("employee_number"),
            "department": doc.get("department"),
            "role": doc.get("role"),
            "profile_image_path": doc.get("profile_image_path"),
            "registration_date": reg_date.isoformat() if isinstance(reg_date, datetime) else (str(reg_date) if reg_date else None),
            "active_status": doc.get("active_status", "ACTIVE"),
            "created_at": created.isoformat() if isinstance(created, datetime) else (str(created) if created else None),
            "updated_at": updated.isoformat() if isinstance(updated, datetime) else (str(updated) if updated else None),
            "total_embeddings": embs_count,
        }

    @staticmethod
    def get_all_raw_for_biometric_cache() -> List[Dict[str, Any]]:
        """Only called by backend AI service on startup to populate memory matching cache."""
        db = get_db()
        cursor = db[COLLECTION_REGISTERED_WORKERS].find(
            {"active_status": "ACTIVE", "biometric_embeddings": {"$exists": True, "$ne": None}},
            {"worker_code": 1, "name": 1, "employee_number": 1, "biometric_embeddings": 1}
        )
        return [
            {
                "worker_code": doc.get("worker_code"),
                "name": doc.get("name"),
                "employee_number": doc.get("employee_number"),
                "biometric_embeddings": doc.get("biometric_embeddings"),
            }
            for doc in cursor
        ]

    @staticmethod
    def update(worker_code: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update registered worker metadata."""
        db = get_db()
        fields_to_set = {}
        for key in ["name", "department", "role", "active_status", "profile_image_path"]:
            if key in update_data and update_data[key] is not None:
                fields_to_set[key] = update_data[key]

        if fields_to_set:
            fields_to_set["updated_at"] = datetime.now(timezone.utc)
            db[COLLECTION_REGISTERED_WORKERS].update_one({"worker_code": worker_code}, {"$set": fields_to_set})

        return RegisteredWorkerRepository.get_by_code(worker_code)

    @staticmethod
    def delete(worker_code: str) -> bool:
        """Permanently delete a registered worker from MongoDB."""
        db = get_db()
        res = db[COLLECTION_REGISTERED_WORKERS].delete_one({"worker_code": worker_code})
        return res.deleted_count > 0

    @staticmethod
    def clear_all_registered_workers() -> int:
        """Permanently delete all registered workers from MongoDB."""
        db = get_db()
        res = db[COLLECTION_REGISTERED_WORKERS].delete_many({})
        return res.deleted_count

    @staticmethod
    def get_historical_stats(worker_code: str) -> Dict[str, Any]:
        """
        Merge lifetime tracking duration, violation count, and PPE compliance
        across all tracking sessions for a permanent worker via MongoDB aggregation.
        """
        db = get_db()
        # 1. Total violations count
        vcount = db[COLLECTION_VIOLATIONS].count_documents({"worker_code": worker_code})

        # 2. Lifetime tracking duration summed across all worker sessions
        pipeline = [
            {"$match": {"worker_code": worker_code}},
            {"$group": {"_id": None, "total_duration": {"$sum": "$total_tracking_duration"}}}
        ]
        dur_result = list(db[COLLECTION_WORKER_SESSIONS].aggregate(pipeline))
        dur_sum = dur_result[0]["total_duration"] if dur_result else 0.0

        # 3. Latest snapshot
        latest_snap = db[COLLECTION_WORKER_SNAPSHOTS].find_one(
            {"worker_code": worker_code},
            sort=[("timestamp", pymongo.DESCENDING)]
        )

        # 4. Average PPE compliance aggregation
        ppe_pipeline = [
            {"$match": {"worker_code": worker_code}},
            {"$group": {"_id": None, "avg_compliance": {"$avg": "$ppe_compliance"}}}
        ]
        ppe_result = list(db[COLLECTION_WORKER_SNAPSHOTS].aggregate(ppe_pipeline))
        avg_compliance = ppe_result[0]["avg_compliance"] if ppe_result else 100.0

        # 5. Last seen from sessions or registered worker record
        last_session = db[COLLECTION_WORKER_SESSIONS].find_one(
            {"worker_code": worker_code},
            sort=[("last_seen", pymongo.DESCENDING)]
        )
        last_seen_val = last_session.get("last_seen") if last_session else None

        return {
            "total_violations_count": vcount,
            "lifetime_tracking_duration": round(float(dur_sum), 1),
            "avg_ppe_compliance": round(float(avg_compliance), 1),
            "latest_risk_score": latest_snap.get("risk_score", 0.0) if latest_snap else 0.0,
        "latest_risk_level": latest_snap.get("risk_level", "SAFE") if latest_snap else "SAFE",
            "last_recognized": last_seen_val.isoformat() if isinstance(last_seen_val, datetime) else (str(last_seen_val) if last_seen_val else None),
        }


# ── 2. Worker Tracking & Session Repository ───────────────────────

class WorkerRepository:
    """CRUD for ByteTrack tracking sessions in MongoDB (`worker_sessions`)."""

    @staticmethod
    def upsert_worker(
        track_id: int,
        source_id: str,
        worker_code: Optional[str] = None,
        photo_url: Optional[str] = None,
        helmet: Optional[bool] = None,
        vest: Optional[bool] = None,
        gloves: Optional[bool] = None,
        face_mask: Optional[bool] = None,
        missing_ppe: Optional[list] = None,
        risk_score: Optional[float] = None,
        risk_level: Optional[str] = None,
        compliance_status: Optional[str] = None,
    ) -> dict:
        db = get_db()
        now = datetime.now(timezone.utc)
        existing = db[COLLECTION_WORKER_SESSIONS].find_one({"track_id": track_id})

        if not existing:
            doc = {
                "worker_id": track_id,
                "track_id": track_id,
                "worker_code": worker_code,
                "permanent_worker_id": worker_code,
                "source_id": source_id,
                "photo_url": photo_url,
                "helmet": helmet,
                "vest": vest,
                "gloves": gloves,
                "face_mask": face_mask,
                "missing_ppe": missing_ppe or ([] if (helmet and vest) else ["Helmet / Hardhat", "High-Visibility Safety Vest"]),
                "risk_score": risk_score if risk_score is not None else (0.0 if (helmet and vest) else 55.0),
                "risk_level": risk_level or ("SAFE" if (helmet and vest) else "MEDIUM"),
                "compliance_status": compliance_status or ("COMPLIANT" if (helmet and vest) else "NON_COMPLIANT"),
                "identity_status": "REGISTERED" if worker_code else "UNKNOWN",
                "started_at": now,
                "first_seen": now,
                "last_seen": now,
                "total_tracking_duration": 0.0,
                "is_live": True,
            }
            res = db[COLLECTION_WORKER_SESSIONS].insert_one(doc)
            doc["_id"] = res.inserted_id
            return doc
        else:
            update = {
                "$set": {
                    "last_seen": now,
                    "is_live": True,
                }
            }
            if photo_url:
                update["$set"]["photo_url"] = photo_url
            if worker_code and not existing.get("worker_code"):
                update["$set"]["worker_code"] = worker_code
                update["$set"]["permanent_worker_id"] = worker_code
                update["$set"]["identity_status"] = "REGISTERED"
            if helmet is not None:
                update["$set"]["helmet"] = helmet
            if vest is not None:
                update["$set"]["vest"] = vest
            if gloves is not None:
                update["$set"]["gloves"] = gloves
            if face_mask is not None:
                update["$set"]["face_mask"] = face_mask
            if missing_ppe is not None:
                update["$set"]["missing_ppe"] = missing_ppe
            if risk_score is not None:
                update["$set"]["risk_score"] = risk_score
            if risk_level is not None:
                update["$set"]["risk_level"] = risk_level
            if compliance_status is not None:
                update["$set"]["compliance_status"] = compliance_status

            db[COLLECTION_WORKER_SESSIONS].update_one({"track_id": track_id}, update)
            return db[COLLECTION_WORKER_SESSIONS].find_one({"track_id": track_id})

    @staticmethod
    def update_worker_duration(
        track_id: int,
        duration: float,
        worker_code: Optional[str] = None,
        photo_url: Optional[str] = None,
        helmet: Optional[bool] = None,
        vest: Optional[bool] = None,
        risk_score: Optional[float] = None,
        risk_level: Optional[str] = None,
        missing_ppe: Optional[list] = None,
        compliance_status: Optional[str] = None,
    ):
        db = get_db()
        now = datetime.now(timezone.utc)
        update = {"$set": {"last_seen": now, "total_tracking_duration": duration}}
        if photo_url:
            update["$set"]["photo_url"] = photo_url
        if worker_code:
            update["$set"]["worker_code"] = worker_code
            update["$set"]["permanent_worker_id"] = worker_code
            update["$set"]["identity_status"] = "REGISTERED"
        if helmet is not None:
            update["$set"]["helmet"] = helmet
        if vest is not None:
            update["$set"]["vest"] = vest
        if risk_score is not None:
            update["$set"]["risk_score"] = risk_score
        if risk_level is not None:
            update["$set"]["risk_level"] = risk_level
        if missing_ppe is not None:
            update["$set"]["missing_ppe"] = missing_ppe
        if compliance_status is not None:
            update["$set"]["compliance_status"] = compliance_status

        db[COLLECTION_WORKER_SESSIONS].update_one({"track_id": track_id}, update)

    @staticmethod
    def add_snapshot(
        worker_id: int,
        helmet: Optional[bool],
        vest: Optional[bool],
        risk_score: float,
        risk_level: str,
        bbox: tuple,
        worker_code: Optional[str] = None,
        activity: Optional[str] = None,
        gloves: Optional[bool] = None,
        face_mask: Optional[bool] = None,
        ppe_compliance: Optional[float] = None,
        photo_url: Optional[str] = None,
    ):
        db = get_db()
        now = datetime.now(timezone.utc)

        # Calculate PPE compliance percentage if not provided
        if ppe_compliance is None:
            items = [helmet, vest, gloves, face_mask]
            valid_items = [i for i in items if i is not None]
            compliant = [i for i in valid_items if i is True]
            ppe_compliance = (len(compliant) / max(1, len(valid_items))) * 100.0

        doc = {
            "worker_id": worker_id,
            "session_id": worker_id,
            "worker_code": worker_code,
            "timestamp": now,
            "helmet": helmet,
            "safety_vest": vest,
            "vest": vest,
            "gloves": gloves,
            "face_mask": face_mask,
            "ppe_compliance": round(float(ppe_compliance), 1),
            "risk_score": round(float(risk_score), 1),
            "risk_level": risk_level,
            "photo_url": photo_url,
            "bbox_x1": bbox[0] if bbox else None,
            "bbox_y1": bbox[1] if bbox else None,
            "bbox_x2": bbox[2] if bbox else None,
            "bbox_y2": bbox[3] if bbox else None,
            "activity": activity,
        }
        db[COLLECTION_WORKER_SNAPSHOTS].insert_one(doc)

    @staticmethod
    def get_all_workers() -> list[dict]:
        import os
        db = get_db()
        workers = list(db[COLLECTION_WORKER_SESSIONS].find().sort("last_seen", pymongo.DESCENDING))
        result = []

        for w in workers:
            tid = w.get("track_id", w.get("worker_id"))
            wcode = w.get("worker_code")
            vcount = db[COLLECTION_VIOLATIONS].count_documents({
                "$or": [{"worker_id": tid}, {"worker_code": wcode}]
            }) if (tid or wcode) else 0

            snap = db[COLLECTION_WORKER_SNAPSHOTS].find_one(
                {"$or": [{"worker_id": tid}, {"worker_code": wcode}]} if (tid or wcode) else {"worker_id": tid},
                sort=[("timestamp", pymongo.DESCENDING)]
            )

            rw = None
            if wcode:
                rw = db[COLLECTION_REGISTERED_WORKERS].find_one({"worker_code": wcode})

            # Multi-tier photo resolution (Guarantees every worker has a photo)
            p_url = None
            if rw and rw.get("profile_image_path"):
                fn = rw["profile_image_path"].replace("\\", "/").split("/")[-1]
                p_url = f"/data/profiles/{fn}"
            elif w.get("photo_url"):
                p_url = w.get("photo_url")
            elif snap and snap.get("photo_url"):
                p_url = snap.get("photo_url")
            elif tid and os.path.exists(f"data/snapshots/worker_{tid}.jpg"):
                p_url = f"/data/snapshots/worker_{tid}.jpg"
            else:
                # Check if there is any violation evidence snapshot for this worker
                v_snap = db[COLLECTION_VIOLATIONS].find_one(
                    {"$or": [{"worker_id": tid}, {"worker_code": wcode}]} if (tid or wcode) else {"worker_id": tid},
                    sort=[("timestamp", pymongo.DESCENDING)]
                )
                if v_snap:
                    p_url = v_snap.get("evidence_url") or (f"/data/evidence/{v_snap['evidence_path'].split('/')[-1]}" if v_snap.get("evidence_path") else None)

            first_seen_val = w.get("first_seen") or w.get("started_at")
            last_seen_val = w.get("last_seen")

            result.append({
                "id": str(w["_id"]),
                "worker_id": tid,
                "track_id": tid,
                "permanent_worker_id": wcode,
                "worker_code": wcode,
                "name": rw.get("name") if rw else (w.get("name") or f"Unknown Worker (Track #{tid})"),
                "identity_status": "REGISTERED" if rw else "UNKNOWN",
                "source_id": w.get("source_id", "webcam"),
                "first_seen": first_seen_val.isoformat() if isinstance(first_seen_val, datetime) else (str(first_seen_val) if first_seen_val else None),
                "last_seen": last_seen_val.isoformat() if isinstance(last_seen_val, datetime) else (str(last_seen_val) if last_seen_val else None),
                "tracking_duration": w.get("total_tracking_duration", 0.0),
                "helmet": snap.get("helmet") if snap else None,
                "vest": snap.get("safety_vest") if snap else (snap.get("vest") if snap else None),
                "gloves": snap.get("gloves") if snap else None,
                "face_mask": snap.get("face_mask") if snap else None,
                "ppe_compliance": snap.get("ppe_compliance", 100.0) if snap else 100.0,
                "risk_score": snap.get("risk_score", 0.0) if snap else 0.0,
                "risk_level": snap.get("risk_level", "SAFE") if snap else "SAFE",
                "risk_factors": [],
                "violation_count": vcount,
                "photo_url": p_url,
                "is_live": w.get("is_live", False),
            })
        return result

    @staticmethod
    def clear_all_worker_sessions():
        """Delete all temporary tracking sessions and snapshots from MongoDB."""
        db = get_db()
        db[COLLECTION_WORKER_SESSIONS].delete_many({})
        db[COLLECTION_WORKER_SNAPSHOTS].delete_many({})
        logger.info("✓ Cleared all worker sessions and snapshots from MongoDB")

    @staticmethod
    def delete_worker_session(track_id: int) -> bool:
        """Delete a specific worker tracking session and snapshots from MongoDB."""
        db = get_db()
        res1 = db[COLLECTION_WORKER_SESSIONS].delete_many({"track_id": track_id})
        res2 = db[COLLECTION_WORKER_SNAPSHOTS].delete_many({"track_id": track_id})
        logger.info(f"✓ Deleted worker tracking session #{track_id} from MongoDB")
        return bool(res1.deleted_count > 0 or res2.deleted_count > 0)

    @staticmethod
    def get_worker_detail(track_id: int) -> Optional[dict]:
        db = get_db()
        worker = db[COLLECTION_WORKER_SESSIONS].find_one({"track_id": track_id})
        if not worker:
            return None

        wcode = worker.get("worker_code")
        violations = list(db[COLLECTION_VIOLATIONS].find({
            "$or": [{"worker_id": track_id}, {"worker_code": wcode}]
        }).sort("timestamp", pymongo.DESCENDING))

        snap = db[COLLECTION_WORKER_SNAPSHOTS].find_one(
            {"$or": [{"worker_id": track_id}, {"worker_code": wcode}]},
            sort=[("timestamp", pymongo.DESCENDING)]
        )

        rw = None
        if wcode:
            rw = db[COLLECTION_REGISTERED_WORKERS].find_one({"worker_code": wcode})

        first_seen_val = worker.get("first_seen") or worker.get("started_at")
        last_seen_val = worker.get("last_seen")

        # Multi-tier photo resolution
        p_url = None
        if rw and rw.get("profile_image_path"):
            fn = rw["profile_image_path"].replace("\\", "/").split("/")[-1]
            p_url = f"/data/profiles/{fn}"
        elif worker.get("photo_url"):
            p_url = worker.get("photo_url")
        elif snap and snap.get("photo_url"):
            p_url = snap.get("photo_url")
        elif os.path.exists(f"data/snapshots/worker_{track_id}.jpg"):
            p_url = f"/data/snapshots/worker_{track_id}.jpg"
        elif violations:
            v_first = violations[0]
            if v_first.get("evidence_path"):
                p_url = f"/data/evidence/{v_first['evidence_path'].split('/')[-1]}"
            elif v_first.get("evidence_url"):
                p_url = v_first.get("evidence_url")

        return {
            "id": str(worker["_id"]),
            "track_id": track_id,
            "permanent_worker_id": wcode,
            "worker_code": wcode,
            "name": rw.get("name") if rw else (worker.get("name") or f"Unknown Worker (Track #{track_id})"),
            "identity_status": "REGISTERED" if rw else "UNKNOWN",
            "source_id": worker.get("source_id", "webcam"),
            "first_seen": first_seen_val.isoformat() if isinstance(first_seen_val, datetime) else (str(first_seen_val) if first_seen_val else None),
            "last_seen": last_seen_val.isoformat() if isinstance(last_seen_val, datetime) else (str(last_seen_val) if last_seen_val else None),
            "tracking_duration": worker.get("total_tracking_duration", 0.0),
            "helmet": snap.get("helmet") if snap else None,
            "vest": snap.get("safety_vest") if snap else (snap.get("vest") if snap else None),
            "gloves": snap.get("gloves") if snap else None,
            "face_mask": snap.get("face_mask") if snap else None,
            "risk_score": snap.get("risk_score", 0.0) if snap else 0.0,
            "risk_level": snap.get("risk_level", "SAFE") if snap else "SAFE",
            "photo_url": p_url,
            "violations": [
                {
                    "violation_id": v.get("violation_id"),
                    "violation_type": v.get("violation_type"),
                    "severity": v.get("severity"),
                    "status": v.get("status"),
                    "timestamp": v.get("timestamp").isoformat() if isinstance(v.get("timestamp"), datetime) else str(v.get("timestamp")),
                }
                for v in violations
            ],
        }


# ── 3. Violation Repository ───────────────────────────────────────

class ViolationRepository:
    @staticmethod
    def save_violation(violation: dict, worker_db_id: Optional[int] = None, worker_code: Optional[str] = None):
        db = get_db()
        vid = violation.get("violation_id", str(uuid.uuid4()))
        now = datetime.now(timezone.utc)

        # Lookup registered worker info if available
        code = worker_code or violation.get("worker_code")
        worker_name = violation.get("worker_name")
        emp_no = violation.get("employee_number")
        if code and (not worker_name or not emp_no):
            reg = db[COLLECTION_REGISTERED_WORKERS].find_one({"worker_code": code})
            if reg:
                worker_name = reg.get("name")
                emp_no = reg.get("employee_number")

        existing = db[COLLECTION_VIOLATIONS].find_one({"violation_id": vid})
        if existing:
            update = {
                "$set": {
                    "duration_seconds": violation.get("duration_seconds", 0.0),
                    "risk_score": violation.get("risk_score", existing.get("risk_score", 0.0)),
                    "updated_at": now,
                }
            }
            if code and not existing.get("worker_code"):
                update["$set"]["worker_code"] = code
            if worker_name and not existing.get("worker_name"):
                update["$set"]["worker_name"] = worker_name
            if emp_no and not existing.get("employee_number"):
                update["$set"]["employee_number"] = emp_no
            db[COLLECTION_VIOLATIONS].update_one({"violation_id": vid}, update)
            return

        ts = violation.get("timestamp")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except Exception:
                ts = now
        elif not isinstance(ts, datetime):
            ts = now

        doc = {
            "violation_id": vid,
            "worker_id": violation.get("worker_id", worker_db_id),
            "worker_code": code,
            "worker_name": worker_name,
            "employee_number": emp_no,
            "session_id": violation.get("session_id", worker_db_id),
            "source_id": violation.get("source_id", "webcam"),
            "violation_type": violation.get("violation_type", "UNKNOWN"),
            "missing_items": violation.get("missing_items", []),
            "severity": violation.get("severity", "MEDIUM"),
            "risk_score": violation.get("risk_score", 0.0),
            "status": violation.get("status", "OPEN"),
            "timestamp": ts,
            "duration_seconds": violation.get("duration_seconds", 0.0),
            "evidence_path": violation.get("evidence_path"),
            "evidence_url": violation.get("evidence_url") or violation.get("evidence_path"),
            "snapshot_base64": violation.get("snapshot_base64"),
            "description": violation.get("description"),
            "created_at": now,
            "updated_at": now,
        }
        db[COLLECTION_VIOLATIONS].insert_one(doc)

    @staticmethod
    def resolve_worker_violations(
        worker_code: Optional[str] = None,
        worker_id: Optional[int] = None,
        except_violation_id: Optional[str] = None
    ):
        """Mark previous open violations as resolved when worker compliance state changes."""
        db = get_db()
        now = datetime.now(timezone.utc)
        query = {"status": "OPEN"}
        if worker_code:
            query["worker_code"] = worker_code
        elif worker_id is not None:
            query["worker_id"] = worker_id
        else:
            return

        if except_violation_id:
            query["violation_id"] = {"$ne": except_violation_id}

        db[COLLECTION_VIOLATIONS].update_many(
            query,
            {"$set": {"status": "RESOLVED", "resolved_at": now, "updated_at": now}}
        )

    @staticmethod
    def get_all() -> list[dict]:
        db = get_db()
        violations = list(db[COLLECTION_VIOLATIONS].find().sort("timestamp", pymongo.DESCENDING))
        result = []
        for v in violations:
            ts = v.get("timestamp")
            resolved_at = v.get("resolved_at")
            wcode = v.get("worker_code")
            wid = v.get("worker_id")

            ev_path = v.get("evidence_path")
            ev_url = v.get("evidence_url") or ev_path
            snap_b64 = v.get("snapshot_base64")

            # Fallback to registered worker photo or live snapshot if evidence photo was not explicitly saved
            if not ev_url and not snap_b64:
                if wcode:
                    rw = db[COLLECTION_REGISTERED_WORKERS].find_one({"worker_code": wcode})
                    if rw:
                        if rw.get("profile_image_path"):
                            fn = rw["profile_image_path"].replace("\\", "/").split("/")[-1]
                            ev_url = f"/data/profiles/{fn}"
                            ev_path = ev_url
                        elif rw.get("profile_image_base64"):
                            snap_b64 = rw.get("profile_image_base64")
                if not ev_url and not snap_b64 and wid is not None:
                    snap = db[COLLECTION_WORKER_SNAPSHOTS].find_one({"track_id": wid}, sort=[("timestamp", -1)])
                    if snap:
                        ev_url = snap.get("photo_url")
                        ev_path = ev_url
                        snap_b64 = snap.get("face_crop_base64")

            result.append({
                "id": str(v["_id"]),
                "violation_id": v.get("violation_id"),
                "worker_id": v.get("worker_id"),
                "permanent_worker_id": v.get("worker_code"),
                "worker_code": v.get("worker_code"),
                "worker_name": v.get("worker_name"),
                "employee_number": v.get("employee_number"),
                "source_id": v.get("source_id", "webcam"),
                "violation_type": v.get("violation_type"),
                "missing_items": v.get("missing_items", []),
                "severity": v.get("severity"),
                "risk_score": v.get("risk_score", 0.0),
                "status": v.get("status", "OPEN"),
                "timestamp": ts.isoformat() if isinstance(ts, datetime) else (str(ts) if ts else None),
                "resolved_at": resolved_at.isoformat() if isinstance(resolved_at, datetime) else (str(resolved_at) if resolved_at else None),
                "duration_seconds": v.get("duration_seconds", 0.0),
                "evidence_path": ev_path,
                "evidence_url": ev_url,
                "snapshot_base64": snap_b64,
                "description": v.get("description"),
            })
        return result

    @staticmethod
    def update_status(violation_id: str, status: str) -> Optional[dict]:
        db = get_db()
        now = datetime.now(timezone.utc)
        res = db[COLLECTION_VIOLATIONS].find_one_and_update(
            {"violation_id": violation_id},
            {"$set": {"status": status, "updated_at": now}},
            return_document=pymongo.ReturnDocument.AFTER
        )
        if res:
            return {"violation_id": res.get("violation_id"), "status": res.get("status")}
        return None

    @staticmethod
    def delete_violation(violation_id: str) -> bool:
        """Delete a violation from MongoDB and remove its stored evidence photo from disk."""
        db = get_db()
        v = db[COLLECTION_VIOLATIONS].find_one({"violation_id": violation_id})
        if not v:
            return False

        # Delete evidence photo file from disk if present
        ev_path = v.get("evidence_path")
        if ev_path:
            clean_path = ev_path.lstrip("/")
            possible_paths = [
                clean_path,
                os.path.join(settings.evidence_dir, os.path.basename(ev_path)),
                os.path.join("data/evidence", os.path.basename(ev_path)),
            ]
            for p in possible_paths:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                        logger.info(f"Removed violation evidence photo: {p}")
                except Exception as e:
                    logger.debug(f"Could not remove evidence file {p}: {e}")

        res = db[COLLECTION_VIOLATIONS].delete_one({"violation_id": violation_id})
        return res.deleted_count > 0

    @staticmethod
    def delete_all_violations() -> int:
        """Delete all violations from MongoDB and purge evidence images."""
        db = get_db()
        try:
            if os.path.exists(settings.evidence_dir):
                for fname in os.listdir(settings.evidence_dir):
                    if fname.startswith("violation_") and fname.endswith(".jpg"):
                        try:
                            os.remove(os.path.join(settings.evidence_dir, fname))
                        except Exception:
                            pass
        except Exception as e:
            logger.debug(f"Error purging evidence directory: {e}")

        res = db[COLLECTION_VIOLATIONS].delete_many({})
        return res.deleted_count

    @staticmethod
    def count_active() -> int:
        db = get_db()
        return db[COLLECTION_VIOLATIONS].count_documents({"status": "OPEN"})

    @staticmethod
    def count_total() -> int:
        db = get_db()
        return db[COLLECTION_VIOLATIONS].count_documents({})


# ── 4. Progress Repository ────────────────────────────────────────

class ProgressRepository:
    """CRUD and History for Construction Progress in MongoDB (`progress_records`)."""

    @staticmethod
    def save(record: dict):
        db = get_db()
        now = datetime.now(timezone.utc)
        doc = {
            "source_id": record.get("source_id", "webcam"),
            "timestamp": now,
            "current_stage": record.get("current_stage", "Not Started"),
            "stage_confidence": record.get("stage_confidence", 0.0),
            "stage_completion_percentage": record.get("stage_completion", record.get("stage_completion_percentage", 0.0)),
            "overall_progress_percentage": record.get("overall_progress", record.get("overall_progress_percentage", 0.0)),
            "project_status": record.get("project_status", "ON_TRACK"),
            "created_at": now,
        }
        db[COLLECTION_PROGRESS_RECORDS].insert_one(doc)

    @staticmethod
    def get_history(limit: int = 100) -> list[dict]:
        db = get_db()
        records = list(db[COLLECTION_PROGRESS_RECORDS].find().sort("timestamp", pymongo.DESCENDING).limit(limit))
        return [
            {
                "id": str(r["_id"]),
                "timestamp": r.get("timestamp").isoformat() if isinstance(r.get("timestamp"), datetime) else str(r.get("timestamp")),
                "source_id": r.get("source_id", "webcam"),
                "current_stage": r.get("current_stage"),
                "stage_confidence": r.get("stage_confidence", 0.0),
                "stage_completion_percentage": r.get("stage_completion_percentage", 0.0),
                "overall_progress_percentage": r.get("overall_progress_percentage", 0.0),
                "project_status": r.get("project_status", "ON_TRACK"),
            }
            for r in records
        ]


# ── 5. Danger Zone & Video Source & Reports ───────────────────────

class DangerZoneRepository:
    """CRUD for danger zones in MongoDB (`danger_zones`)."""

    @staticmethod
    def get_all() -> list[dict]:
        db = get_db()
        zones = list(db[COLLECTION_DANGER_ZONES].find())
        return serialize_mongo_docs(zones)

    @staticmethod
    def save(zone: dict):
        db = get_db()
        now = datetime.now(timezone.utc)
        doc = {
            "zone_id": zone.get("zone_id", str(uuid.uuid4())),
            "source_id": zone.get("source_id", "webcam"),
            "name": zone.get("name", "Restricted Area"),
            "zone_type": zone.get("zone_type", "RESTRICTED"),
            "polygon_data": zone.get("polygon_data", []),
            "risk_weight": zone.get("risk_weight", 30.0),
            "active": zone.get("active", True),
            "created_at": now,
        }
        db[COLLECTION_DANGER_ZONES].replace_one({"zone_id": doc["zone_id"]}, doc, upsert=True)

    @staticmethod
    def delete(zone_id: str):
        db = get_db()
        db[COLLECTION_DANGER_ZONES].delete_one({"zone_id": zone_id})


class VideoSourceRepository:
    """CRUD for video and RTSP sources in MongoDB (`video_sources`)."""

    @staticmethod
    def get_all() -> list[dict]:
        db = get_db()
        sources = list(db[COLLECTION_VIDEO_SOURCES].find())
        # Sanitise RTSP passwords from frontend return
        sanitized = []
        for s in sources:
            item = serialize_mongo_doc(s)
            config = item.get("configuration", {})
            if "rtsp_url" in config:
                import re
                config["rtsp_url"] = re.sub(r"://([^:]+):([^@]+)@", "://***:***@", config["rtsp_url"])
            sanitized.append(item)
        return sanitized

    @staticmethod
    def save(source: dict):
        db = get_db()
        now = datetime.now(timezone.utc)
        doc = {
            "source_id": source.get("source_id", str(uuid.uuid4())),
            "name": source.get("name", "Camera Source"),
            "source_type": source.get("source_type", "WEBCAM"),
            "configuration": source.get("configuration", {}),
            "status": source.get("status", "ACTIVE"),
            "created_at": now,
            "updated_at": now,
        }
        db[COLLECTION_VIDEO_SOURCES].replace_one({"source_id": doc["source_id"]}, doc, upsert=True)


class ReportRepository:
    """CRUD for generated safety and progress reports in MongoDB (`reports`)."""

    @staticmethod
    def save_report(report_type: str, period_start: str, period_end: str, file_path: str, data: dict = None) -> dict:
        db = get_db()
        now = datetime.now(timezone.utc)
        doc = {
            "report_type": report_type,
            "period_start": period_start,
            "period_end": period_end,
            "file_path": file_path,
            "data": data or {},
            "generated_at": now,
        }
        res = db[COLLECTION_REPORTS].insert_one(doc)
        doc["_id"] = res.inserted_id
        return serialize_mongo_doc(doc)

    @staticmethod
    def get_all(report_type: Optional[str] = None) -> list[dict]:
        db = get_db()
        query = {"report_type": report_type} if report_type else {}
        reports = list(db[COLLECTION_REPORTS].find(query).sort("generated_at", pymongo.DESCENDING))
        return serialize_mongo_docs(reports)


# ── 6. Advanced MongoDB Analytics & Aggregations ──────────────────

class AnalyticsAggregationRepository:
    """High-performance MongoDB aggregation pipelines for Dashboards and Reports."""

    @staticmethod
    def get_dashboard_metrics() -> Dict[str, Any]:
        """Aggregate real-time metrics for the main Dashboard."""
        db = get_db()

        # 1. Total violations
        total_viols = db[COLLECTION_VIOLATIONS].count_documents({})
        active_viols = db[COLLECTION_VIOLATIONS].count_documents({"status": "OPEN"})

        # 2. Risk distribution aggregation
        risk_pipeline = [
            {"$group": {"_id": "$risk_level", "count": {"$sum": 1}}}
        ]
        risk_res = list(db[COLLECTION_WORKER_SNAPSHOTS].aggregate(risk_pipeline))
        risk_dist = {"SAFE": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        for r in risk_res:
            lvl = r.get("_id")
            if lvl in risk_dist:
                risk_dist[lvl] = r.get("count", 0)

        # 3. Overall PPE compliance average
        ppe_pipeline = [
            {"$group": {"_id": None, "avg_ppe": {"$avg": "$ppe_compliance"}}}
        ]
        ppe_res = list(db[COLLECTION_WORKER_SNAPSHOTS].aggregate(ppe_pipeline))
        avg_ppe = round(ppe_res[0]["avg_ppe"], 1) if ppe_res else 100.0

        # 4. Latest progress record
        latest_progress = db[COLLECTION_PROGRESS_RECORDS].find_one(
            {}, sort=[("timestamp", pymongo.DESCENDING)]
        )

        return {
            "active_violations": active_viols,
            "total_violations": total_viols,
            "average_ppe_compliance": avg_ppe,
            "risk_distribution": risk_dist,
            "latest_stage": latest_progress.get("current_stage", "Site Preparation") if latest_progress else "Site Preparation",
            "overall_progress_percentage": latest_progress.get("overall_progress_percentage", 0.0) if latest_progress else 0.0,
        }

    @staticmethod
    def get_safety_analytics_data() -> Dict[str, Any]:
        """Aggregate violation breakdowns and safety trends."""
        db = get_db()

        # Violation type distribution
        type_pipeline = [
            {"$group": {"_id": "$violation_type", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        type_res = list(db[COLLECTION_VIOLATIONS].aggregate(type_pipeline))
        types = [{"type": r["_id"], "count": r["count"]} for r in type_res if r["_id"]]

        # Severity breakdown
        sev_pipeline = [
            {"$group": {"_id": "$severity", "count": {"$sum": 1}}}
        ]
        sev_res = list(db[COLLECTION_VIOLATIONS].aggregate(sev_pipeline))
        severities = {r["_id"]: r["count"] for r in sev_res if r["_id"]}

        return {
            "violation_types": types,
            "severities": severities,
            "total_violations": db[COLLECTION_VIOLATIONS].count_documents({}),
            "active_violations": db[COLLECTION_VIOLATIONS].count_documents({"status": "OPEN"}),
        }
