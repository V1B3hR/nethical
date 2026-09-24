# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Tests for Space Situational Awareness and CCSDS CDM ingestion (tests.space.test_ssa_client)."""

from datetime import datetime, timezone
import pytest

from nethical.space.ssa_client import (
    CCSDSConjunctionDataMessage,
    SSAClient,
)


class TestSSAClient:
    """Test suite for CCSDS Conjunction Data Message (CDM) and CelesTrak parsing."""

    def test_cdm_from_dict_and_json(self) -> None:
        raw_cdm = {
            "message_id": "CDM_2026_09_TEST_01",
            "originator": "EU_SST_CA",
            "tca": "2026-09-24T12:00:00+00:00",
            "miss_distance_m": 420.5,
            "relative_speed_mps": 14200.0,
            "collision_probability": 2.4e-4,
            "primary_object_id": "SOVEREIGN_SAT_01",
            "secondary_object_id": "DEBRIS_COSMOS_2251",
            "primary_state": {
                "position_eci_km": {"x": 6878.0, "y": 0.0, "z": 0.0},
                "velocity_eci_kms": {"x": 0.0, "y": 7.6, "z": 0.0},
            },
            "secondary_state": {
                "position_eci_km": {"x": 6878.4, "y": 0.05, "z": 0.0},
                "velocity_eci_kms": {"x": 0.0, "y": -7.6, "z": 0.0},
            },
        }

        cdm = CCSDSConjunctionDataMessage.from_dict(raw_cdm)
        assert cdm.message_id == "CDM_2026_09_TEST_01"
        assert cdm.miss_distance_m == 420.5
        assert cdm.collision_probability == 2.4e-4
        assert cdm.primary_object_id == "SOVEREIGN_SAT_01"
        assert cdm.secondary_object_id == "DEBRIS_COSMOS_2251"
        assert cdm.primary_state.position_eci_km.x == 6878.0

    def test_cdm_from_ccsds_kvn_format(self) -> None:
        kvn_data = """CCSDS_CDM_VERS = 1.0;
COMMENT Conjunction Data Message from 18th Space Defense Squadron;
MESSAGE_ID = CDM_KVN_18SPCS_998;
ORIGINATOR = 18_SPCS;
TCA = 2026-09-24T18:30:00;
MISS_DISTANCE = 310.0;
RELATIVE_SPEED = 14500.0;
COLLISION_PROBABILITY = 3.5e-4;
OBJECT1_ID = SAT_POLSA_01;
OBJECT1_NAME = POLSA_EAGLE;
OBJECT1_X = 6900.0;
OBJECT1_Y = 0.0;
OBJECT1_Z = 0.0;
OBJECT1_X_DOT = 0.0;
OBJECT1_Y_DOT = 7.6;
OBJECT1_Z_DOT = 0.0;
OBJECT2_ID = FENGYUN_DEBRIS_44;
OBJECT2_NAME = FY1C_DEBRIS;
OBJECT2_X = 6900.3;
OBJECT2_Y = 0.0;
OBJECT2_Z = 0.0;
OBJECT2_X_DOT = 0.0;
OBJECT2_Y_DOT = -7.6;
OBJECT2_Z_DOT = 0.0;
"""
        cdm = CCSDSConjunctionDataMessage.from_kvn(kvn_data)
        assert cdm.message_id == "CDM_KVN_18SPCS_998"
        assert cdm.miss_distance_m == 310.0
        assert cdm.primary_object_id == "SAT_POLSA_01"
        assert cdm.secondary_object_id == "FENGYUN_DEBRIS_44"
        assert cdm.collision_probability == 3.5e-4

    def test_ssa_client_registration_and_filtering(self) -> None:
        client = SSAClient()
        raw = {
            "message_id": "CDM_TEST",
            "tca": "2026-09-24T12:00:00+00:00",
            "miss_distance_m": 500.0,
            "primary_object_id": "SAT_MY_FLEET",
            "secondary_object_id": "DEBRIS_99",
        }
        cdm = CCSDSConjunctionDataMessage.from_dict(raw)
        client.register_cdm(cdm)

        my_cdms = client.get_active_cdms("SAT_MY_FLEET")
        assert len(my_cdms) == 1
        assert my_cdms[0].message_id == "CDM_TEST"

        other_cdms = client.get_active_cdms("SAT_OTHER")
        assert len(other_cdms) == 0

    def test_celestrak_multi_tle_parsing(self) -> None:
        client = SSAClient()
        catalog_sample = """ISS (ZARYA)
1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9001
2 25544  51.6416 120.1234 0005432  65.1234  80.1234 15.49876543123456
TIANGONG
1 48274U 21035A   24001.50000000  .00021432  00000-0  14560-3 0  9002
2 48274  41.4721 145.4321 0003456  45.1234  90.4321 15.61234567123456
"""
        states = client.parse_celestrak_tle_catalog(catalog_sample)
        assert len(states) == 2
        assert states[0].satellite_id == "ISS (ZARYA)"
        assert states[1].satellite_id == "TIANGONG"
        assert states[0].altitude_km > 350.0
        assert states[1].altitude_km > 350.0
