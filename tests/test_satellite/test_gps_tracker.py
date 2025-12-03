"""Tests for GPS tracker module."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

from nethical.connectivity.satellite.gps_tracker import (
    GPSTracker,
    Position,
    GNSSConstellation,
    GeofenceType,
    Geofence,
)


class TestPosition:
    """Tests for Position dataclass."""

    def test_default_values(self):
        """Test default position values."""
        pos = Position()
        assert pos.latitude == 0.0
        assert pos.longitude == 0.0
        assert pos.altitude_m == 0.0

    def test_custom_position(self):
        """Test custom position."""
        pos = Position(
            latitude=37.7749,
            longitude=-122.4194,
            altitude_m=10.0,
            horizontal_accuracy_m=5.0,
        )
        assert pos.latitude == 37.7749
        assert pos.longitude == -122.4194
        assert pos.altitude_m == 10.0
        assert pos.horizontal_accuracy_m == 5.0

    def test_distance_to(self):
        """Test distance calculation."""
        # San Francisco
        sf = Position(latitude=37.7749, longitude=-122.4194)
        # Los Angeles
        la = Position(latitude=34.0522, longitude=-118.2437)

        distance = sf.distance_to(la)
        # ~559km according to haversine
        assert 550000 < distance < 570000  # meters

    def test_distance_to_self(self):
        """Test distance to same location is zero."""
        pos = Position(latitude=37.7749, longitude=-122.4194)
        distance = pos.distance_to(pos)
        assert distance == pytest.approx(0.0, abs=0.01)

    def test_bearing_to(self):
        """Test bearing calculation."""
        # Due north
        pos1 = Position(latitude=0.0, longitude=0.0)
        pos2 = Position(latitude=1.0, longitude=0.0)

        bearing = pos1.bearing_to(pos2)
        assert bearing == pytest.approx(0.0, abs=1.0)  # 0° = North

    def test_bearing_to_east(self):
        """Test bearing calculation to east."""
        pos1 = Position(latitude=0.0, longitude=0.0)
        pos2 = Position(latitude=0.0, longitude=1.0)

        bearing = pos1.bearing_to(pos2)
        assert bearing == pytest.approx(90.0, abs=1.0)  # 90° = East


class TestGNSSConstellation:
    """Tests for GNSS constellation enum."""

    def test_constellation_values(self):
        """Test constellation enum values."""
        assert GNSSConstellation.GPS.value == "gps"
        assert GNSSConstellation.GLONASS.value == "glonass"
        assert GNSSConstellation.GALILEO.value == "galileo"
        assert GNSSConstellation.BEIDOU.value == "beidou"


class TestGeofence:
    """Tests for Geofence class."""

    def test_circular_geofence(self):
        """Test circular geofence creation."""
        fence = Geofence(
            fence_id="test-1",
            name="Test Zone",
            fence_type=GeofenceType.CIRCLE,
            center_lat=37.7749,
            center_lon=-122.4194,
            radius_m=1000.0,
        )

        assert fence.fence_id == "test-1"
        assert fence.name == "Test Zone"
        assert fence.fence_type == GeofenceType.CIRCLE
        assert fence.radius_m == 1000.0

    def test_point_inside_circular_fence(self):
        """Test point inside circular geofence."""
        fence = Geofence(
            fence_id="test-1",
            name="Test Zone",
            fence_type=GeofenceType.CIRCLE,
            center_lat=37.7749,
            center_lon=-122.4194,
            radius_m=1000.0,  # 1km radius
        )

        # Point very close to center
        inside = Position(latitude=37.775, longitude=-122.419)
        assert fence.contains(inside) is True

        # Point far outside
        outside = Position(latitude=38.0, longitude=-122.0)
        assert fence.contains(outside) is False


class TestGPSTracker:
    """Tests for GPSTracker."""

    @pytest.fixture
    def tracker(self):
        """Create GPS tracker."""
        return GPSTracker(
            constellations=[
                GNSSConstellation.GPS,
                GNSSConstellation.GLONASS,
            ],
            update_interval_seconds=1.0,
        )

    def test_initial_state(self, tracker):
        """Test initial tracker state."""
        assert tracker._is_tracking is False
        assert len(tracker._constellations) == 2
        assert tracker._current_position is None

    @pytest.mark.asyncio
    async def test_start_tracking(self, tracker):
        """Test starting tracking."""
        await tracker.start_tracking()
        assert tracker._is_tracking is True
        await tracker.stop_tracking()

    @pytest.mark.asyncio
    async def test_stop_tracking(self, tracker):
        """Test stopping tracking."""
        await tracker.start_tracking()
        await tracker.stop_tracking()
        assert tracker._is_tracking is False

    @pytest.mark.asyncio
    async def test_get_position(self, tracker):
        """Test getting position."""
        await tracker.start_tracking()

        # Simulate position update
        tracker._current_position = Position(
            latitude=37.7749,
            longitude=-122.4194,
            altitude_m=10.0,
        )

        position = await tracker.get_position()
        assert position.latitude == 37.7749
        assert position.longitude == -122.4194
        await tracker.stop_tracking()

    def test_create_circular_geofence(self, tracker):
        """Test creating circular geofence."""
        fence = tracker.create_circular_geofence(
            fence_id="zone-1",
            name="Restricted Zone",
            center_lat=37.7749,
            center_lon=-122.4194,
            radius_m=500.0,
        )

        assert fence is not None
        # Check it's in the geofences list
        assert len(tracker._geofences) == 1

    def test_remove_geofence(self, tracker):
        """Test removing geofence."""
        fence = tracker.create_circular_geofence(
            fence_id="zone-1",
            name="Test Zone",
            center_lat=37.7749,
            center_lon=-122.4194,
            radius_m=500.0,
        )

        assert len(tracker._geofences) == 1
        tracker.remove_geofence("zone-1")
        assert len(tracker._geofences) == 0

    def test_callback_registration(self, tracker):
        """Test callback registration."""
        callback = MagicMock()
        tracker.register_callback("on_position_update", callback)
        assert callback in tracker._callbacks["on_position_update"]

    @pytest.mark.asyncio
    async def test_get_status(self, tracker):
        """Test getting tracker status."""
        await tracker.start_tracking()

        tracker._current_position = Position(
            latitude=37.7749,
            longitude=-122.4194,
            satellites_used=12,
            satellites_visible=18,
        )

        status = tracker.get_status()
        assert status["is_tracking"] is True
        await tracker.stop_tracking()
