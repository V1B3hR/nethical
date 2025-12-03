"""
GPS/GNSS Positioning and Tracking

Provides real-time positioning using multiple GNSS constellations:
- GPS (USA)
- GLONASS (Russia)
- Galileo (EU)
- BeiDou (China)

Features:
- Real-time position tracking
- Geofencing capabilities
- Location-aware routing decisions
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class GNSSConstellation(Enum):
    """GNSS constellation types."""

    GPS = "gps"
    GLONASS = "glonass"
    GALILEO = "galileo"
    BEIDOU = "beidou"
    ALL = "all"


class GeofenceType(Enum):
    """Geofence geometry types."""

    CIRCLE = "circle"
    POLYGON = "polygon"
    RECTANGLE = "rectangle"


class GeofenceEvent(Enum):
    """Geofence event types."""

    ENTER = "enter"
    EXIT = "exit"
    DWELL = "dwell"


@dataclass
class Position:
    """GPS/GNSS position data."""

    latitude: float = 0.0
    longitude: float = 0.0
    altitude_m: float = 0.0

    # Accuracy metrics
    horizontal_accuracy_m: float = 0.0
    vertical_accuracy_m: float = 0.0

    # Speed and heading
    speed_mps: float = 0.0
    heading_deg: float = 0.0

    # Fix information
    fix_type: str = "none"  # none, 2d, 3d, dgps, rtk
    satellites_used: int = 0
    satellites_visible: int = 0
    hdop: float = 0.0
    vdop: float = 0.0
    pdop: float = 0.0

    # Timestamp
    timestamp: Optional[datetime] = None

    # Source constellation
    constellation: GNSSConstellation = GNSSConstellation.GPS

    def distance_to(self, other: "Position") -> float:
        """
        Calculate distance to another position using Haversine formula.

        Args:
            other: Target position

        Returns:
            Distance in meters
        """
        R = 6371000  # Earth's radius in meters

        lat1 = math.radians(self.latitude)
        lat2 = math.radians(other.latitude)
        dlat = math.radians(other.latitude - self.latitude)
        dlon = math.radians(other.longitude - self.longitude)

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def bearing_to(self, other: "Position") -> float:
        """
        Calculate bearing to another position.

        Args:
            other: Target position

        Returns:
            Bearing in degrees (0-360)
        """
        lat1 = math.radians(self.latitude)
        lat2 = math.radians(other.latitude)
        dlon = math.radians(other.longitude - self.longitude)

        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(
            dlon
        )

        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude_m": self.altitude_m,
            "horizontal_accuracy_m": self.horizontal_accuracy_m,
            "vertical_accuracy_m": self.vertical_accuracy_m,
            "speed_mps": self.speed_mps,
            "heading_deg": self.heading_deg,
            "fix_type": self.fix_type,
            "satellites_used": self.satellites_used,
            "constellation": self.constellation.value,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


@dataclass
class Geofence:
    """Geofence definition."""

    fence_id: str
    name: str
    fence_type: GeofenceType
    enabled: bool = True

    # Circle parameters
    center_lat: float = 0.0
    center_lon: float = 0.0
    radius_m: float = 0.0

    # Polygon parameters (list of lat/lon tuples)
    vertices: List[Tuple[float, float]] = field(default_factory=list)

    # Rectangle parameters
    north_lat: float = 0.0
    south_lat: float = 0.0
    east_lon: float = 0.0
    west_lon: float = 0.0

    # Event callbacks
    on_enter: Optional[Callable] = None
    on_exit: Optional[Callable] = None
    on_dwell: Optional[Callable] = None
    dwell_time_seconds: float = 60.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def contains(self, position: Position) -> bool:
        """
        Check if position is within geofence.

        Args:
            position: Position to check

        Returns:
            True if position is inside geofence
        """
        if not self.enabled:
            return False

        if self.fence_type == GeofenceType.CIRCLE:
            return self._contains_circle(position)
        elif self.fence_type == GeofenceType.RECTANGLE:
            return self._contains_rectangle(position)
        elif self.fence_type == GeofenceType.POLYGON:
            return self._contains_polygon(position)

        return False

    def _contains_circle(self, position: Position) -> bool:
        """Check if position is within circular geofence."""
        center = Position(latitude=self.center_lat, longitude=self.center_lon)
        distance = position.distance_to(center)
        return distance <= self.radius_m

    def _contains_rectangle(self, position: Position) -> bool:
        """Check if position is within rectangular geofence."""
        return (
            self.south_lat <= position.latitude <= self.north_lat
            and self.west_lon <= position.longitude <= self.east_lon
        )

    def _contains_polygon(self, position: Position) -> bool:
        """Check if position is within polygon geofence (ray casting)."""
        if len(self.vertices) < 3:
            return False

        n = len(self.vertices)
        inside = False
        x, y = position.longitude, position.latitude

        p1x, p1y = self.vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = self.vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside


class GPSTracker:
    """
    GPS/GNSS positioning and tracking system.

    Provides real-time position tracking with support for multiple
    GNSS constellations and geofencing capabilities.
    """

    def __init__(
        self,
        constellations: Optional[List[GNSSConstellation]] = None,
        update_interval_seconds: float = 1.0,
    ):
        """
        Initialize GPS tracker.

        Args:
            constellations: GNSS constellations to use
            update_interval_seconds: Position update interval
        """
        self._constellations = constellations or [GNSSConstellation.GPS]
        self._update_interval = update_interval_seconds

        self._current_position: Optional[Position] = None
        self._position_history: List[Position] = []
        self._max_history_size = 1000

        self._geofences: Dict[str, Geofence] = {}
        self._geofence_states: Dict[str, bool] = {}  # Inside or outside

        self._is_tracking = False
        self._callbacks: Dict[str, List[Callable]] = {
            "on_position_update": [],
            "on_geofence_enter": [],
            "on_geofence_exit": [],
        }

    @property
    def current_position(self) -> Optional[Position]:
        """Get current position."""
        return self._current_position

    @property
    def is_tracking(self) -> bool:
        """Check if tracking is active."""
        return self._is_tracking

    @property
    def geofences(self) -> Dict[str, Geofence]:
        """Get all geofences."""
        return self._geofences

    async def start_tracking(self) -> bool:
        """
        Start GPS tracking.

        Returns:
            True if tracking started successfully
        """
        try:
            self._is_tracking = True
            logger.info(
                f"GPS tracking started with constellations: "
                f"{[c.value for c in self._constellations]}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to start GPS tracking: {e}")
            return False

    async def stop_tracking(self) -> bool:
        """
        Stop GPS tracking.

        Returns:
            True if tracking stopped successfully
        """
        self._is_tracking = False
        logger.info("GPS tracking stopped")
        return True

    async def get_position(self) -> Optional[Position]:
        """
        Get current GPS position.

        Returns:
            Current position or None if unavailable
        """
        if not self._is_tracking:
            return None

        # In a real implementation, this would read from GNSS hardware
        # For now, return simulated position
        position = Position(
            latitude=37.7749,
            longitude=-122.4194,
            altitude_m=10.0,
            horizontal_accuracy_m=3.0,
            vertical_accuracy_m=5.0,
            speed_mps=0.0,
            heading_deg=0.0,
            fix_type="3d",
            satellites_used=12,
            satellites_visible=15,
            hdop=1.2,
            vdop=1.8,
            pdop=2.2,
            timestamp=datetime.utcnow(),
            constellation=self._constellations[0],
        )

        self._update_position(position)
        return position

    def _update_position(self, position: Position):
        """Update current position and check geofences."""
        self._current_position = position

        # Store in history
        self._position_history.append(position)
        if len(self._position_history) > self._max_history_size:
            self._position_history.pop(0)

        # Trigger callbacks
        for callback in self._callbacks["on_position_update"]:
            try:
                callback(position)
            except Exception as e:
                logger.error(f"Error in position callback: {e}")

        # Check geofences
        self._check_geofences(position)

    def _check_geofences(self, position: Position):
        """Check position against all geofences."""
        for fence_id, fence in self._geofences.items():
            if not fence.enabled:
                continue

            is_inside = fence.contains(position)
            was_inside = self._geofence_states.get(fence_id, False)

            if is_inside and not was_inside:
                # Entered geofence
                self._geofence_states[fence_id] = True
                logger.info(f"Entered geofence: {fence.name}")
                if fence.on_enter:
                    fence.on_enter(fence, position)
                for callback in self._callbacks["on_geofence_enter"]:
                    callback(fence, position)

            elif not is_inside and was_inside:
                # Exited geofence
                self._geofence_states[fence_id] = False
                logger.info(f"Exited geofence: {fence.name}")
                if fence.on_exit:
                    fence.on_exit(fence, position)
                for callback in self._callbacks["on_geofence_exit"]:
                    callback(fence, position)

    def add_geofence(self, geofence: Geofence):
        """
        Add a geofence.

        Args:
            geofence: Geofence to add
        """
        self._geofences[geofence.fence_id] = geofence
        self._geofence_states[geofence.fence_id] = False
        logger.info(f"Added geofence: {geofence.name} ({geofence.fence_type.value})")

    def remove_geofence(self, fence_id: str) -> bool:
        """
        Remove a geofence.

        Args:
            fence_id: Geofence ID to remove

        Returns:
            True if removed successfully
        """
        if fence_id in self._geofences:
            del self._geofences[fence_id]
            del self._geofence_states[fence_id]
            return True
        return False

    def create_circular_geofence(
        self,
        fence_id: str,
        name: str,
        center_lat: float,
        center_lon: float,
        radius_m: float,
        **kwargs,
    ) -> Geofence:
        """
        Create and add a circular geofence.

        Args:
            fence_id: Unique identifier
            name: Human-readable name
            center_lat: Center latitude
            center_lon: Center longitude
            radius_m: Radius in meters
            **kwargs: Additional geofence parameters

        Returns:
            Created geofence
        """
        geofence = Geofence(
            fence_id=fence_id,
            name=name,
            fence_type=GeofenceType.CIRCLE,
            center_lat=center_lat,
            center_lon=center_lon,
            radius_m=radius_m,
            **kwargs,
        )
        self.add_geofence(geofence)
        return geofence

    def register_callback(self, event: str, callback: Callable):
        """Register a callback for position or geofence events."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def get_position_history(
        self, limit: Optional[int] = None
    ) -> List[Position]:
        """
        Get position history.

        Args:
            limit: Maximum number of positions to return

        Returns:
            List of historical positions
        """
        if limit:
            return self._position_history[-limit:]
        return self._position_history.copy()

    def get_travel_distance_m(self) -> float:
        """
        Calculate total travel distance from position history.

        Returns:
            Total distance in meters
        """
        if len(self._position_history) < 2:
            return 0.0

        total = 0.0
        for i in range(1, len(self._position_history)):
            total += self._position_history[i - 1].distance_to(
                self._position_history[i]
            )
        return total

    def get_status(self) -> Dict[str, Any]:
        """
        Get tracker status.

        Returns:
            Status dictionary
        """
        return {
            "is_tracking": self._is_tracking,
            "constellations": [c.value for c in self._constellations],
            "update_interval_seconds": self._update_interval,
            "current_position": (
                self._current_position.to_dict() if self._current_position else None
            ),
            "position_history_size": len(self._position_history),
            "geofence_count": len(self._geofences),
            "active_geofences": [
                f.name for f in self._geofences.values() if f.enabled
            ],
        }
