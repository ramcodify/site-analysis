"""BuildSight AI — MongoDB Connection Manager (Motor Async + PyMongo Sync)"""

import logging
from typing import Optional
from pymongo import MongoClient
from pymongo.database import Database as SyncDatabase
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import settings
from app.database.indexes import create_indexes

logger = logging.getLogger(__name__)

# Singletons
_sync_client: Optional[MongoClient] = None
_sync_db: Optional[SyncDatabase] = None
_async_client: Optional[AsyncIOMotorClient] = None
_async_db: Optional[AsyncIOMotorDatabase] = None


def get_mongo_client() -> MongoClient:
    """Get synchronous PyMongo client."""
    global _sync_client
    if _sync_client is None:
        _sync_client = MongoClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=3000,
            connectTimeoutMS=3000,
            maxPoolSize=50,
        )
    return _sync_client


def get_db() -> SyncDatabase:
    """Get synchronous PyMongo database."""
    global _sync_db
    if _sync_db is None:
        client = get_mongo_client()
        _sync_db = client[settings.mongodb_database]
    return _sync_db


def get_async_client() -> AsyncIOMotorClient:
    """Get asynchronous Motor client for FastAPI."""
    global _async_client
    if _async_client is None:
        _async_client = AsyncIOMotorClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=3000,
            connectTimeoutMS=3000,
            maxPoolSize=50,
        )
    return _async_client


def get_async_db() -> AsyncIOMotorDatabase:
    """Get asynchronous Motor database."""
    global _async_db
    if _async_db is None:
        client = get_async_client()
        _async_db = client[settings.mongodb_database]
    return _async_db


def init_db():
    """Initialize MongoDB database, test connectivity, and create all indexes."""
    try:
        db = get_db()
        # Test connection ping
        db.command("ping")
        # Create all collection indexes
        create_indexes(db)
        logger.info(f"✓ Connected to MongoDB database '{settings.mongodb_database}' ({settings.mongodb_uri})")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB at '{settings.mongodb_uri}': {e}")
        raise


def is_mongo_connected() -> bool:
    """Ping check for GET /health endpoint."""
    try:
        db = get_db()
        db.command("ping")
        return True
    except Exception:
        return False


def close_mongo_connection():
    """Graceful disconnect on application shutdown."""
    global _sync_client, _sync_db, _async_client, _async_db
    if _sync_client is not None:
        _sync_client.close()
        _sync_client = None
        _sync_db = None
    if _async_client is not None:
        _async_client.close()
        _async_client = None
        _async_db = None
    logger.info("✓ MongoDB connections closed.")
