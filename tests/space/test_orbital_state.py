# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Tests for orbital mechanics and state modeling (tests.space.test_orbital_state)."""

from datetime import datetime, timezone
import math
import pytest

from nethical.space.models import (
    EARTH_EQUATORIAL_RADIUS_KM,
    MU_EARTH_KM3_S2,
    Covariance3D,
    OrbitalRegime,
    OrbitalState,
    Vector3D,
)


class TestVector3D:
    """Test suite for 3D astrodynamical vector mathematics."""

    def test_vector_basic_operations(self) -> None:
        v1 = Vector3D(x=3.0, y=0.0, z=4.0)
        assert v1.magnitude == 5.0

        v2 = Vector3D(x=0.0, y=2.0, z=0.0)
        assert v1.dot(v2) == 0.0

        cross = v1.cross(v2)
        assert cross.x == -8.0
        assert cross.y == 0.0
        assert cross.z == 6.0

        dist = v1.distance_to(v2)
        assert pytest.approx(dist, rel=1e-3) == math.sqrt(9.0 + 4.0 + 16.0)

        norm = v1.normalized()
        assert pytest.approx(norm.magnitude, rel=1e-5) == 1.0

        scaled = v1.scale(2.0)
        assert scaled.x == 6.0 and scaled.z == 8.0

        added = v1.add(v2)
        assert added.x == 3.0 and added.y == 2.0 and added.z == 4.0

        subtracted = v1.subtract(v2)
        assert subtracted.x == 3.0 and subtracted.y == -2.0 and subtracted.z == 4.0


class TestOrbitalState:
    """Test suite for Keplerian orbital state vectors and regimes."""

    def test_leo_circular_orbit_properties(self) -> None:
        """Verify LEO circular orbit at 500 km altitude."""
        r_mag = EARTH_EQUATORIAL_RADIUS_KM + 500.0  # 6878.137 km
        v_circ = math.sqrt(MU_EARTH_KM3_S2 / r_mag)  # ~7.61 km/s

        state = OrbitalState(
            satellite_id="SAT_LEO_01",
            epoch=datetime.now(timezone.utc),
            position_eci_km=Vector3D(x=r_mag, y=0.0, z=0.0),
            velocity_eci_kms=Vector3D(x=0.0, y=v_circ, z=0.0),
        )

        assert pytest.approx(state.altitude_km, rel=1e-3) == 500.0
        assert pytest.approx(state.speed_kms, rel=1e-3) == v_circ
        assert state.regime == OrbitalRegime.LEO
        assert state.orbital_period_minutes > 90.0 and state.orbital_period_minutes < 100.0
        assert state.eccentricity is not None
        assert state.eccentricity < 0.01
        assert pytest.approx(state.perigee_altitude_km, abs=5.0) == 500.0
        assert pytest.approx(state.apogee_altitude_km, abs=5.0) == 500.0

    def test_geo_orbit_properties(self) -> None:
        """Verify Geostationary Earth Orbit (GEO) at ~35,786 km."""
        r_geo = EARTH_EQUATORIAL_RADIUS_KM + 35786.0  # 42164.137 km
        v_geo = math.sqrt(MU_EARTH_KM3_S2 / r_geo)  # ~3.07 km/s

        state = OrbitalState(
            satellite_id="SAT_GEO_01",
            position_eci_km=Vector3D(x=r_geo, y=0.0, z=0.0),
            velocity_eci_kms=Vector3D(x=0.0, y=v_geo, z=0.0),
        )

        assert state.regime == OrbitalRegime.GEO
        # Orbital period of GEO is 23h 56m (~1436 minutes)
        assert pytest.approx(state.orbital_period_minutes, abs=15.0) == 1436.0

    def test_meo_orbit_properties(self) -> None:
        """Verify Medium Earth Orbit (e.g. Galileo constellation at 23,222 km)."""
        r_meo = EARTH_EQUATORIAL_RADIUS_KM + 23222.0
        v_meo = math.sqrt(MU_EARTH_KM3_S2 / r_meo)

        state = OrbitalState(
            satellite_id="GALILEO_MEO_01",
            position_eci_km=Vector3D(x=r_meo, y=0.0, z=0.0),
            velocity_eci_kms=Vector3D(x=0.0, y=v_meo, z=0.0),
        )

        assert state.regime == OrbitalRegime.MEO

    def test_keplerian_propagation(self) -> None:
        """Verify two-body propagation forwards in time."""
        r_mag = EARTH_EQUATORIAL_RADIUS_KM + 600.0
        v_mag = math.sqrt(MU_EARTH_KM3_S2 / r_mag)

        initial = OrbitalState(
            satellite_id="SAT_PROP_01",
            position_eci_km=Vector3D(x=r_mag, y=0.0, z=0.0),
            velocity_eci_kms=Vector3D(x=0.0, y=v_mag, z=0.0),
        )

        # Propagate quarter orbit
        period_sec = initial.orbital_period_minutes * 60.0
        quarter_period = period_sec / 4.0

        propagated = initial.propagate_kepler(quarter_period)
        assert propagated.epoch.timestamp() == pytest.approx(initial.epoch.timestamp() + quarter_period, rel=1e-5)
        # In circular orbit, magnitude remains approximately constant
        assert pytest.approx(propagated.position_eci_km.magnitude, rel=1e-2) == r_mag
        # After 90 degrees, x is near 0 and y is near r_mag
        assert abs(propagated.position_eci_km.x) < 200.0
        assert pytest.approx(propagated.position_eci_km.y, rel=1e-2) == r_mag

    def test_tle_parsing(self) -> None:
        """Verify standard NORAD TLE parsing (ISS Zarya)."""
        # ISS (ZARYA) NORAD TLE sample
        line1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9001"
        line2 = "2 25544  51.6416 120.1234 0005432  65.1234  80.1234 15.49876543123456"

        state = OrbitalState.from_tle(line1, line2, satellite_id="ISS_ZARYA")
        assert state.satellite_id == "ISS_ZARYA"
        assert state.altitude_km > 350.0 and state.altitude_km < 450.0
        assert pytest.approx(state.inclination_deg, abs=0.1) == 51.64
        assert state.regime == OrbitalRegime.LEO
        assert state.epoch.tzinfo == timezone.utc

    def test_utc_epoch_coercion(self) -> None:
        """Ensure naive datetimes are automatically coerced to UTC."""
        naive_dt = datetime(2026, 9, 24, 12, 0, 0)
        state = OrbitalState(
            satellite_id="SAT_UTC_01",
            epoch=naive_dt,
            position_eci_km=Vector3D(x=7000.0, y=0.0, z=0.0),
            velocity_eci_kms=Vector3D(x=0.0, y=7.5, z=0.0),
        )
        assert state.epoch.tzinfo == timezone.utc
