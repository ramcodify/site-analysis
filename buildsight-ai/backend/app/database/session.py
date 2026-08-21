"""BuildSight AI — Database Session & MongoDB Integration"""

import logging
from app.database.mongodb import get_db, get_mongo_client, init_db as init_mongo_db, close_mongo_connection

logger = logging.getLogger(__name__)


def init_db():
    """Initialize database collections and indexes in MongoDB."""
    init_mongo_db()
