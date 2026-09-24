# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Tests for constellation topology and orbital routing (tests.space.test_constellation_topology)."""

from datetime import datetime, timezone
import pytest

from nethical.space.models import (
    ConstellationTopology,
    GroundStationContact,
    InterSatelliteLink,
    ISLLinkStatus,
)


class TestConstellationTopology:
    """Test suite for orbital mesh routing and ground station visibility."""

    def test_alternate_route_bypassing_jammed_link(self) -> None:
        """Verify dynamic re-routing when a cross-link is jammed or severed."""
        topology = ConstellationTopology(
            satellite_id="SAT_PLANE1_01",
            constellation_name="SOVEREIGN_CONSTELLATION",
            orbital_plane=1,
            plane_slot=1,
            inter_satellite_links={
                "SAT_PLANE1_02": InterSatelliteLink(
                    target_satellite_id="SAT_PLANE1_02",
                    link_type="LASER_OPTICAL",
                    range_km=1200.0,
                    azimuth_deg=0.0,
                    elevation_deg=0.0,
                    status=ISLLinkStatus.JAMMED,  # Jammed primary link!
                    latency_ms=4.0,
                ),
                "SAT_PLANE2_01": InterSatelliteLink(
                    target_satellite_id="SAT_PLANE2_01",
                    link_type="LASER_OPTICAL",
                    range_km=1500.0,
                    azimuth_deg=90.0,
                    elevation_deg=0.0,
                    status=ISLLinkStatus.ACTIVE,  # Healthy cross-plane link
                    latency_ms=6.0,
                ),
            },
            active_route_table={
                "GATEWAY_EUROPE": ["SAT_PLANE1_02", "GATEWAY_EUROPE"],
            },
        )

        alternate = topology.find_alternate_route(
            destination="GATEWAY_EUROPE",
            failed_link_sat_id="SAT_PLANE1_02",
        )
        assert alternate is not None
        assert alternate[0] == "SAT_PLANE2_01"
        assert alternate[1] == "GATEWAY_EUROPE"

    def test_best_ground_station_selection(self) -> None:
        """Verify selection of ground station with highest elevation angle above horizon."""
        topology = ConstellationTopology(
            satellite_id="SAT_01",
            ground_station_contacts={
                "GS_SVALBARD": GroundStationContact(
                    station_id="GS_SVALBARD",
                    station_name="Svalbard Satellite Station",
                    latitude=78.22,
                    longitude=15.65,
                    elevation_angle_deg=12.5,
                    azimuth_angle_deg=340.0,
                    is_visible=True,
                ),
                "GS_REDZIKOWO": GroundStationContact(
                    station_id="GS_REDZIKOWO",
                    station_name="Redzikowo Gateway Station",
                    latitude=54.47,
                    longitude=17.05,
                    elevation_angle_deg=48.2,  # Best visibility / lowest atmospheric path
                    azimuth_angle_deg=180.0,
                    is_visible=True,
                ),
                "GS_HARWELL": GroundStationContact(
                    station_id="GS_HARWELL",
                    station_name="Harwell Space Gateway (UK)",
                    latitude=51.57,
                    longitude=-1.31,
                    elevation_angle_deg=5.0,  # Below 10 deg threshold
                    azimuth_angle_deg=220.0,
                    is_visible=True,
                ),
            },
        )

        best = topology.get_best_ground_station(min_elevation_deg=10.0)
        assert best is not None
        assert best.station_id == "GS_REDZIKOWO"
        assert best.elevation_angle_deg == 48.2
