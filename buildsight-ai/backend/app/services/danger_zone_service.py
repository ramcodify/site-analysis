"""BuildSight AI — Danger Zone Service

Polygon-based danger zone detection for worker safety.
"""

import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class DangerZoneService:
    """Manages danger zones and checks worker positions against them."""

    def __init__(self):
        self._zones: list[dict] = []

    def add_zone(self, zone: dict):
        """Add a danger zone."""
        self._zones.append(zone)

    def remove_zone(self, zone_id: int):
        """Remove a danger zone by ID."""
        self._zones = [z for z in self._zones if z.get("id") != zone_id]

    def get_zones(self) -> list[dict]:
        """Get all active danger zones."""
        return [z for z in self._zones if z.get("is_active", True)]

    def check_worker_in_zone(
        self,
        worker_bbox: tuple[float, float, float, float],
        frame_width: int = 640,
        frame_height: int = 480,
    ) -> Optional[dict]:
        """Check if a worker's bounding box center intersects any danger zone.

        Uses point-in-polygon test (ray casting algorithm).
        """
        if not self._zones:
            return None

        # Calculate worker center point
        x1, y1, x2, y2 = worker_bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        for zone in self._zones:
            if not zone.get("is_active", True):
                continue

            polygon = zone.get("polygon_data", [])
            if len(polygon) < 3:
                continue

            if self._point_in_polygon(cx, cy, polygon):
                return zone

        return None

    @staticmethod
    def _point_in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
        """Ray casting algorithm for point-in-polygon test."""
        n = len(polygon)
        inside = False
        j = n - 1

        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]

            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i

        return inside
