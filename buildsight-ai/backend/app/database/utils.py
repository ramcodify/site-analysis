"""BuildSight AI — MongoDB ObjectId Helpers and Serialization Utilities"""

from bson import ObjectId
from datetime import datetime
from typing import Any, Optional, Dict, List
from fastapi import HTTPException


def to_object_id(id_str: str) -> ObjectId:
    """Validate and convert a string to a MongoDB ObjectId, raising 400 on invalid input."""
    if not id_str or not ObjectId.is_valid(id_str):
        raise HTTPException(status_code=400, detail=f"Invalid ObjectId format: '{id_str}'")
    return ObjectId(id_str)


def is_valid_object_id(id_str: str) -> bool:
    """Check if string is a valid 24-hex character ObjectId."""
    return bool(id_str and ObjectId.is_valid(id_str))


def serialize_mongo_doc(doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Convert a MongoDB document into a JSON-serializable dictionary with string `id`."""
    if doc is None:
        return None
    res = dict(doc)
    if "_id" in res:
        res["id"] = str(res.pop("_id"))
    for k, v in res.items():
        if isinstance(v, ObjectId):
            res[k] = str(v)
        elif isinstance(v, datetime):
            res[k] = v.isoformat()
    return res


def serialize_mongo_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert a list of MongoDB documents into JSON-serializable dictionaries."""
    return [serialize_mongo_doc(d) for d in docs if d is not None]
