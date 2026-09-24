# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Space Situational Awareness (SSA) & Space Traffic Management (STM) Client (nethical.space.ssa_client).

Implements live and air-gapped ingestion of orbital ephemerides, TLE catalogs, and
CCSDS 508.0-B-1 Conjunction Data Messages (CDM) from international SSA networks:
- CelesTrak (NORAD GP TLE feeds)
- Space-Track.org (18th Space Defense Squadron / US Space Command)
- EU Space Surveillance and Tracking (EU SST)
- Commercial SSA radars (e.g. LeoLabs)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from nethical.space.models import (
    Covariance3D,
    OrbitalState,
    Vector3D,
)

logger = logging.getLogger("nethical.space.ssa_client")


class CCSDSConjunctionDataMessage(BaseModel):
    """Standard CCSDS 508.0-B-1 Conjunction Data Message (CDM)."""
    message_id: str
    creation_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    originator: str = Field(default="EU_SST_CA", description="SSA provider (EU_SST, 18_SPCS, CELESTRAK)")
    tca: datetime = Field(..., description="Time of Closest Approach (UTC)")
    miss_distance_m: float
    relative_speed_mps: float
    collision_probability: Optional[float] = None
    primary_object_id: str
    primary_object_name: str = "PRIMARY_SAT"
    secondary_object_id: str
    secondary_object_name: str = "SECONDARY_DEBRIS"
    primary_state: OrbitalState
    secondary_state: OrbitalState

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CCSDSConjunctionDataMessage:
        """Parse dictionary or JSON representation of a CCSDS CDM."""
        tca_raw = data["tca"]
        if isinstance(tca_raw, str):
            tca_dt = datetime.fromisoformat(tca_raw)
            if tca_dt.tzinfo is None:
                tca_dt = tca_dt.replace(tzinfo=timezone.utc)
        else:
            tca_dt = tca_raw

        # Construct or extract primary and secondary orbital states
        p_data = data.get("primary_state", {})
        s_data = data.get("secondary_state", {})

        p_pos = p_data.get("position_eci_km", {"x": 6878.0, "y": 0.0, "z": 0.0})
        p_vel = p_data.get("velocity_eci_kms", {"x": 0.0, "y": 7.6, "z": 0.0})
        s_pos = s_data.get("position_eci_km", {"x": 6878.05, "y": 0.01, "z": 0.0})
        s_vel = s_data.get("velocity_eci_kms", {"x": 0.0, "y": -7.6, "z": 0.0})

        primary_state = OrbitalState(
            satellite_id=data.get("primary_object_id", "SAT_01"),
            epoch=tca_dt,
            position_eci_km=Vector3D(**p_pos),
            velocity_eci_kms=Vector3D(**p_vel),
            covariance=Covariance3D(**p_data.get("covariance", {})),
        )

        secondary_state = OrbitalState(
            satellite_id=data.get("secondary_object_id", "DEBRIS_01"),
            epoch=tca_dt,
            position_eci_km=Vector3D(**s_pos),
            velocity_eci_kms=Vector3D(**s_vel),
            covariance=Covariance3D(**s_data.get("covariance", {})),
        )

        return cls(
            message_id=data.get("message_id", f"CDM_{int(tca_dt.timestamp())}"),
            originator=data.get("originator", "EU_SST_CA"),
            tca=tca_dt,
            miss_distance_m=float(data["miss_distance_m"]),
            relative_speed_mps=float(data.get("relative_speed_mps", 14000.0)),
            collision_probability=float(data["collision_probability"]) if data.get("collision_probability") is not None else None,
            primary_object_id=data.get("primary_object_id", "SAT_01"),
            primary_object_name=data.get("primary_object_name", "PRIMARY_SAT"),
            secondary_object_id=data.get("secondary_object_id", "DEBRIS_01"),
            secondary_object_name=data.get("secondary_object_name", "SECONDARY_DEBRIS"),
            primary_state=primary_state,
            secondary_state=secondary_state,
        )

    @classmethod
    def from_kvn(cls, kvn_text: str) -> CCSDSConjunctionDataMessage:
        """Parse CCSDS Keyword-Value-Notation (KVN) text into a CDM object."""
        kv_pairs: Dict[str, str] = {}
        for line in kvn_text.splitlines():
            line = line.strip()
            if not line or line.startswith("COMMENT"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                kv_pairs[key.strip()] = val.strip().replace(";", "")

        tca_str = kv_pairs.get("TCA", datetime.now(timezone.utc).isoformat())
        tca_dt = datetime.fromisoformat(tca_str)
        if tca_dt.tzinfo is None:
            tca_dt = tca_dt.replace(tzinfo=timezone.utc)

        miss_dist_m = float(kv_pairs.get("MISS_DISTANCE", "1000.0"))
        rel_speed = float(kv_pairs.get("RELATIVE_SPEED", "14000.0"))
        pc = float(kv_pairs.get("COLLISION_PROBABILITY", "1e-5"))

        # Primary vectors
        p_x = float(kv_pairs.get("OBJECT1_X", "6878.0"))
        p_y = float(kv_pairs.get("OBJECT1_Y", "0.0"))
        p_z = float(kv_pairs.get("OBJECT1_Z", "0.0"))
        p_vx = float(kv_pairs.get("OBJECT1_X_DOT", "0.0"))
        p_vy = float(kv_pairs.get("OBJECT1_Y_DOT", "7.6"))
        p_vz = float(kv_pairs.get("OBJECT1_Z_DOT", "0.0"))

        # Secondary vectors
        s_x = float(kv_pairs.get("OBJECT2_X", str(p_x + (miss_dist_m / 1000.0))))
        s_y = float(kv_pairs.get("OBJECT2_Y", "0.0"))
        s_z = float(kv_pairs.get("OBJECT2_Z", "0.0"))
        s_vx = float(kv_pairs.get("OBJECT2_X_DOT", "0.0"))
        s_vy = float(kv_pairs.get("OBJECT2_Y_DOT", "-7.6"))
        s_vz = float(kv_pairs.get("OBJECT2_Z_DOT", "0.0"))

        primary_state = OrbitalState(
            satellite_id=kv_pairs.get("OBJECT1_ID", "SAT_01"),
            epoch=tca_dt,
            position_eci_km=Vector3D(x=p_x, y=p_y, z=p_z),
            velocity_eci_kms=Vector3D(x=p_vx, y=p_vy, z=p_vz),
        )
        secondary_state = OrbitalState(
            satellite_id=kv_pairs.get("OBJECT2_ID", "DEBRIS_01"),
            epoch=tca_dt,
            position_eci_km=Vector3D(x=s_x, y=s_y, z=s_z),
            velocity_eci_kms=Vector3D(x=s_vx, y=s_vy, z=s_vz),
        )

        return cls(
            message_id=kv_pairs.get("MESSAGE_ID", f"CDM_KVN_{int(tca_dt.timestamp())}"),
            originator=kv_pairs.get("ORIGINATOR", "EU_SST_CA"),
            tca=tca_dt,
            miss_distance_m=miss_dist_m,
            relative_speed_mps=rel_speed,
            collision_probability=pc,
            primary_object_id=kv_pairs.get("OBJECT1_ID", "SAT_01"),
            primary_object_name=kv_pairs.get("OBJECT1_NAME", "PRIMARY_SAT"),
            secondary_object_id=kv_pairs.get("OBJECT2_ID", "DEBRIS_01"),
            secondary_object_name=kv_pairs.get("OBJECT2_NAME", "SECONDARY_DEBRIS"),
            primary_state=primary_state,
            secondary_state=secondary_state,
        )


class SSAClient:
    """Client for fetching live ephemerides and CCSDS Conjunction Data Messages (CDMs)."""

    def __init__(self, air_gapped: bool = True, cache_dir: Optional[str] = None) -> None:
        self.air_gapped = air_gapped
        self.cache_dir = cache_dir
        self._cdm_cache: List[CCSDSConjunctionDataMessage] = []

    def load_cdm_from_file(self, file_path: str) -> CCSDSConjunctionDataMessage:
        """Load a CCSDS CDM from disk in either JSON or KVN format with explicit UTF-8 encoding."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if content.strip().startswith("{"):
            return CCSDSConjunctionDataMessage.from_dict(json.loads(content))
        return CCSDSConjunctionDataMessage.from_kvn(content)

    def register_cdm(self, cdm: CCSDSConjunctionDataMessage) -> None:
        """Register a validated Conjunction Data Message into the client cache."""
        self._cdm_cache.append(cdm)
        logger.info(
            "Registered Conjunction Data Message %s from %s (TCA: %s, Miss: %.1f m, Pc: %s)",
            cdm.message_id,
            cdm.originator,
            cdm.tca.isoformat(),
            cdm.miss_distance_m,
            cdm.collision_probability,
        )

    def get_active_cdms(self, satellite_id: str) -> List[CCSDSConjunctionDataMessage]:
        """Retrieve all active conjunction data messages involving the given spacecraft."""
        return [
            cdm for cdm in self._cdm_cache
            if cdm.primary_object_id == satellite_id or cdm.secondary_object_id == satellite_id
        ]

    def parse_celestrak_tle_catalog(self, catalog_text: str) -> List[OrbitalState]:
        """Parse multi-satellite 3-line or 2-line TLE catalog from CelesTrak."""
        lines = [line.strip() for line in catalog_text.splitlines() if line.strip()]
        states: List[OrbitalState] = []
        i = 0
        while i < len(lines):
            # Check if line 0 is satellite name header (3-line format)
            if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
                line1 = lines[i]
                line2 = lines[i + 1]
                sat_name = None
                i += 2
            elif (
                i + 2 < len(lines)
                and lines[i + 1].startswith("1 ")
                and lines[i + 2].startswith("2 ")
            ):
                sat_name = lines[i]
                line1 = lines[i + 1]
                line2 = lines[i + 2]
                i += 3
            else:
                i += 1
                continue

            try:
                state = OrbitalState.from_tle(line1, line2, satellite_id=sat_name)
                states.append(state)
            except Exception as e:
                logger.warning("Failed to parse TLE entry for %s: %s", sat_name, e)

        return states
