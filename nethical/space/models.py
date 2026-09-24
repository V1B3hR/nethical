# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Orbital & Stratospheric Domain Models (nethical.space.models).

Defines the mathematical, orbital mechanics, link-budget, constellation topology,
and High-Altitude Platform Station (HAPS) telemetry models for autonomous space operations.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


# Standard Physical and Astrodynamical Constants (WGS-84 / Earth)
MU_EARTH_KM3_S2: float = 398600.4418  # Standard gravitational parameter (km^3/s^2)
EARTH_EQUATORIAL_RADIUS_KM: float = 6378.137  # WGS-84 equatorial radius (km)
SPEED_OF_LIGHT_M_S: float = 299792458.0  # Speed of light (m/s)
BOLTZMANN_CONSTANT_J_K: float = 1.380649e-23  # Boltzmann constant (J/K)
GEO_ALTITUDE_KM: float = 35786.0  # Geostationary orbital altitude (km)


class OrbitalRegime(str, Enum):
    """Orbital altitude and trajectory classification regimes."""
    LEO = "LEO"             # Low Earth Orbit (160 - 2,000 km)
    MEO = "MEO"             # Medium Earth Orbit (2,000 - 35,786 km, e.g. Galileo, GPS)
    GEO = "GEO"             # Geostationary / Geosynchronous (~35,786 km)
    HEO = "HEO"             # Highly Elliptical Orbit (e.g. Molniya, Tundra)
    CISLUNAR = "CISLUNAR"   # Beyond GEO / Earth-Moon system Lagrange points
    STRATOSPHERE = "STRATOSPHERE"  # Sub-orbital HAPS envelope (15 - 25 km)


class Vector3D(BaseModel):
    """3D Cartesian Vector with astrodynamical vector mathematics."""
    x: float
    y: float
    z: float

    @property
    def magnitude(self) -> float:
        """Euclidean norm / vector magnitude."""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def dot(self, other: Vector3D) -> float:
        """Inner dot product."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector3D) -> Vector3D:
        """Vector cross product."""
        return Vector3D(
            x=self.y * other.z - self.z * other.y,
            y=self.z * other.x - self.x * other.z,
            z=self.x * other.y - self.y * other.x,
        )

    def distance_to(self, other: Vector3D) -> float:
        """Euclidean distance between two vector endpoints."""
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def normalized(self) -> Vector3D:
        """Unit vector in the same direction."""
        mag = self.magnitude
        if mag == 0.0:
            return Vector3D(x=0.0, y=0.0, z=0.0)
        return Vector3D(x=self.x / mag, y=self.y / mag, z=self.z / mag)

    def scale(self, factor: float) -> Vector3D:
        """Scalar multiplication."""
        return Vector3D(x=self.x * factor, y=self.y * factor, z=self.z * factor)

    def add(self, other: Vector3D) -> Vector3D:
        """Vector addition."""
        return Vector3D(x=self.x + other.x, y=self.y + other.y, z=self.z + other.z)

    def subtract(self, other: Vector3D) -> Vector3D:
        """Vector subtraction."""
        return Vector3D(x=self.x - other.x, y=self.y - other.y, z=self.z - other.z)


class Covariance3D(BaseModel):
    """3-axis positional uncertainty standard deviations (1-sigma error ellipsoid)."""
    sigma_radial_m: float = Field(default=10.0, description="Radial (U) positional uncertainty (m)")
    sigma_intrack_m: float = Field(default=50.0, description="In-track (V) positional uncertainty (m)")
    sigma_crosstrack_m: float = Field(default=25.0, description="Cross-track (W) positional uncertainty (m)")

    @property
    def max_position_uncertainty_m(self) -> float:
        """Semi-major axis of position error ellipsoid."""
        return max(self.sigma_radial_m, self.sigma_intrack_m, self.sigma_crosstrack_m)


class OrbitalState(BaseModel):
    """Complete orbital state vector and Keplerian parameters in Earth-Centered Inertial (J2000)."""
    satellite_id: str
    epoch: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    position_eci_km: Vector3D
    velocity_eci_kms: Vector3D
    semi_major_axis_km: Optional[float] = None
    eccentricity: Optional[float] = None
    inclination_deg: Optional[float] = None
    raan_deg: Optional[float] = None
    arg_perigee_deg: Optional[float] = None
    true_anomaly_deg: Optional[float] = None
    covariance: Covariance3D = Field(default_factory=Covariance3D)

    @field_validator("epoch")
    @classmethod
    def enforce_utc_epoch(cls, v: datetime) -> datetime:
        """Ensure epoch datetime is timezone-aware UTC."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    def model_post_init(self, __context: Any) -> None:
        """Derive classical Keplerian orbital elements if not provided."""
        if self.semi_major_axis_km is None or self.eccentricity is None:
            self._derive_keplerian_elements()

    def _derive_keplerian_elements(self) -> None:
        """Derive Keplerian orbital elements from ECI position and velocity vectors."""
        r_vec = self.position_eci_km
        v_vec = self.velocity_eci_kms
        r = r_vec.magnitude
        v = v_vec.magnitude

        if r == 0.0:
            return

        # Specific orbital energy: epsilon = v^2/2 - mu/r
        energy = (v * v) / 2.0 - (MU_EARTH_KM3_S2 / r)
        if abs(energy) > 1e-9:
            a = -MU_EARTH_KM3_S2 / (2.0 * energy)
        else:
            a = r
        self.semi_major_axis_km = a

        # Specific angular momentum: h = r x v
        h_vec = r_vec.cross(v_vec)
        h = h_vec.magnitude

        # Eccentricity vector: e_vec = (v x h)/mu - r_vec/r
        v_cross_h = v_vec.cross(h_vec)
        e_vec = Vector3D(
            x=(v_cross_h.x / MU_EARTH_KM3_S2) - (r_vec.x / r),
            y=(v_cross_h.y / MU_EARTH_KM3_S2) - (r_vec.y / r),
            z=(v_cross_h.z / MU_EARTH_KM3_S2) - (r_vec.z / r),
        )
        e = e_vec.magnitude
        self.eccentricity = e

        # Inclination: i = arccos(h_z / h)
        if h > 0.0:
            cos_i = max(-1.0, min(1.0, h_vec.z / h))
            self.inclination_deg = math.degrees(math.acos(cos_i))
        else:
            self.inclination_deg = 0.0

        # Line of nodes: n = k x h = (-h_y, h_x, 0)
        n_vec = Vector3D(x=-h_vec.y, y=h_vec.x, z=0.0)
        n = n_vec.magnitude

        # Right Ascension of Ascending Node (RAAN): Omega
        if n > 1e-9:
            cos_raan = max(-1.0, min(1.0, n_vec.x / n))
            raan = math.degrees(math.acos(cos_raan))
            if n_vec.y < 0:
                raan = 360.0 - raan
            self.raan_deg = raan
        else:
            self.raan_deg = 0.0

        # Argument of Perigee: omega
        if n > 1e-9 and e > 1e-6:
            cos_omega = max(-1.0, min(1.0, n_vec.dot(e_vec) / (n * e)))
            omega = math.degrees(math.acos(cos_omega))
            if e_vec.z < 0:
                omega = 360.0 - omega
            self.arg_perigee_deg = omega
        else:
            self.arg_perigee_deg = 0.0

        # True Anomaly: nu
        if e > 1e-6:
            cos_nu = max(-1.0, min(1.0, e_vec.dot(r_vec) / (e * r)))
            nu = math.degrees(math.acos(cos_nu))
            if r_vec.dot(v_vec) < 0:
                nu = 360.0 - nu
            self.true_anomaly_deg = nu
        else:
            self.true_anomaly_deg = 0.0

    @property
    def altitude_km(self) -> float:
        """Instantaneous altitude above Earth equatorial mean sea level."""
        return self.position_eci_km.magnitude - EARTH_EQUATORIAL_RADIUS_KM

    @property
    def speed_kms(self) -> float:
        """Instantaneous orbital speed in km/s."""
        return self.velocity_eci_kms.magnitude

    @property
    def regime(self) -> OrbitalRegime:
        """Determine orbital regime according to apogee and perigee altitudes."""
        alt = self.altitude_km
        e = self.eccentricity or 0.0
        if alt < 2000.0 and e < 0.25:
            return OrbitalRegime.LEO
        if 2000.0 <= alt < 35000.0:
            return OrbitalRegime.MEO
        if 35000.0 <= alt <= 36500.0 and e < 0.05:
            return OrbitalRegime.GEO
        if e >= 0.25:
            return OrbitalRegime.HEO
        return OrbitalRegime.CISLUNAR

    @property
    def orbital_period_minutes(self) -> float:
        """Keplerian orbital period in minutes."""
        a = self.semi_major_axis_km or (self.position_eci_km.magnitude)
        if a <= 0.0:
            return 0.0
        return (2.0 * math.pi * math.sqrt((a ** 3) / MU_EARTH_KM3_S2)) / 60.0

    @property
    def perigee_altitude_km(self) -> float:
        """Perigee altitude above Earth surface."""
        a = self.semi_major_axis_km or self.position_eci_km.magnitude
        e = self.eccentricity or 0.0
        return a * (1.0 - e) - EARTH_EQUATORIAL_RADIUS_KM

    @property
    def apogee_altitude_km(self) -> float:
        """Apogee altitude above Earth surface."""
        a = self.semi_major_axis_km or self.position_eci_km.magnitude
        e = self.eccentricity or 0.0
        return a * (1.0 + e) - EARTH_EQUATORIAL_RADIUS_KM

    def propagate_kepler(self, delta_seconds: float) -> OrbitalState:
        """Fast two-body Keplerian propagation forward or backward in time.

        Calculates approximate state vector at t + delta_seconds using mean motion.
        """
        a = self.semi_major_axis_km or self.position_eci_km.magnitude
        if a <= 0.0:
            return self

        mean_motion_rad_s = math.sqrt(MU_EARTH_KM3_S2 / (a ** 3))
        delta_anomaly_rad = mean_motion_rad_s * delta_seconds

        # Simplified rotation in orbital plane for short-horizon conjunction verification
        cos_da = math.cos(delta_anomaly_rad)
        sin_da = math.sin(delta_anomaly_rad)

        # Angular rate vector: omega = (r x v) / r^2
        r_mag = self.position_eci_km.magnitude
        if r_mag == 0.0:
            return self

        # Rotate position and velocity vectors by delta anomaly
        pos = self.position_eci_km
        vel = self.velocity_eci_kms

        new_pos = Vector3D(
            x=pos.x * cos_da + (vel.x / mean_motion_rad_s) * sin_da,
            y=pos.y * cos_da + (vel.y / mean_motion_rad_s) * sin_da,
            z=pos.z * cos_da + (vel.z / mean_motion_rad_s) * sin_da,
        )
        new_vel = Vector3D(
            x=vel.x * cos_da - (pos.x * mean_motion_rad_s) * sin_da,
            y=vel.y * cos_da - (pos.y * mean_motion_rad_s) * sin_da,
            z=vel.z * cos_da - (pos.z * mean_motion_rad_s) * sin_da,
        )

        new_epoch = datetime.fromtimestamp(self.epoch.timestamp() + delta_seconds, tz=timezone.utc)

        return OrbitalState(
            satellite_id=self.satellite_id,
            epoch=new_epoch,
            position_eci_km=new_pos,
            velocity_eci_kms=new_vel,
            semi_major_axis_km=self.semi_major_axis_km,
            eccentricity=self.eccentricity,
            inclination_deg=self.inclination_deg,
            raan_deg=self.raan_deg,
            arg_perigee_deg=self.arg_perigee_deg,
            true_anomaly_deg=(self.true_anomaly_deg or 0.0) + math.degrees(delta_anomaly_rad) % 360.0,
            covariance=self.covariance,
        )

    @classmethod
    def from_tle(cls, line1: str, line2: str, satellite_id: Optional[str] = None) -> OrbitalState:
        """Parse standard NORAD Two-Line Element (TLE) set into an OrbitalState."""
        sat_num = line1[2:7].strip()
        sat_id = satellite_id or f"NORAD_{sat_num}"

        # Epoch parsing from Line 1 (Columns 19-32)
        epoch_year_2digit = int(line1[18:20])
        epoch_year = 2000 + epoch_year_2digit if epoch_year_2digit < 57 else 1900 + epoch_year_2digit
        epoch_day = float(line1[20:32])
        epoch_timestamp = datetime(epoch_year, 1, 1, tzinfo=timezone.utc).timestamp() + (epoch_day - 1.0) * 86400.0
        parsed_epoch = datetime.fromtimestamp(epoch_timestamp, tz=timezone.utc)

        # Line 2 elements
        inclination = float(line2[8:16])
        raan = float(line2[17:25])
        eccentricity = float("0." + line2[26:33].strip())
        arg_perigee = float(line2[34:42])
        mean_anomaly = float(line2[43:51])
        mean_motion_rev_day = float(line2[52:63])

        # Semi-major axis from mean motion
        n_rad_s = mean_motion_rev_day * (2.0 * math.pi / 86400.0)
        semi_major_axis = (MU_EARTH_KM3_S2 / (n_rad_s ** 2)) ** (1.0 / 3.0)

        # Radial distance and velocity magnitude
        p = semi_major_axis * (1.0 - eccentricity ** 2)
        p = max(1.0, p)
        nu_rad = math.radians(mean_anomaly)
        r_mag = p / (1.0 + eccentricity * math.cos(nu_rad))
        v_mag = math.sqrt(MU_EARTH_KM3_S2 * max(0.0, 2.0 / r_mag - 1.0 / semi_major_axis))

        # Argument of latitude: u = omega + nu
        u_rad = math.radians(arg_perigee + mean_anomaly)
        inc_rad = math.radians(inclination)
        raan_rad = math.radians(raan)

        # Orbital plane coordinates
        x_prime = r_mag * math.cos(u_rad)
        y_prime = r_mag * math.sin(u_rad)

        vx_prime = -v_mag * math.sin(u_rad)
        vy_prime = v_mag * math.cos(u_rad)

        # Rotate to ECI (J2000) frame
        pos_x = x_prime * math.cos(raan_rad) - y_prime * math.cos(inc_rad) * math.sin(raan_rad)
        pos_y = x_prime * math.sin(raan_rad) + y_prime * math.cos(inc_rad) * math.cos(raan_rad)
        pos_z = y_prime * math.sin(inc_rad)

        vel_x = vx_prime * math.cos(raan_rad) - vy_prime * math.cos(inc_rad) * math.sin(raan_rad)
        vel_y = vx_prime * math.sin(raan_rad) + vy_prime * math.cos(inc_rad) * math.cos(raan_rad)
        vel_z = vy_prime * math.sin(inc_rad)

        return cls(
            satellite_id=sat_id,
            epoch=parsed_epoch,
            position_eci_km=Vector3D(x=pos_x, y=pos_y, z=pos_z),
            velocity_eci_kms=Vector3D(x=vel_x, y=vel_y, z=vel_z),
            semi_major_axis_km=semi_major_axis,
            eccentricity=eccentricity,
            inclination_deg=inclination,
            raan_deg=raan,
            arg_perigee_deg=arg_perigee,
            true_anomaly_deg=mean_anomaly,
        )


class LinkBudget(BaseModel):
    """RF and Optical Link Budget with electronic warfare jamming assessment."""
    link_id: str
    carrier_frequency_ghz: float = Field(..., description="Carrier frequency in GHz (e.g. 14.2 for Ku-band)")
    tx_power_dbw: float = Field(default=10.0, description="Transmitter power in dBW (10 dBW = 10 Watts)")
    tx_antenna_gain_dbi: float = Field(default=35.0, description="Tx antenna gain in dBi")
    rx_antenna_gain_dbi: float = Field(default=38.0, description="Rx antenna gain in dBi")
    slant_range_km: float = Field(default=1000.0, description="Distance between transmitter and receiver (km)")
    atmospheric_loss_db: float = Field(default=0.8, description="Atmospheric absorption and rain attenuation (dB)")
    pointing_loss_db: float = Field(default=0.5, description="Antenna misalignment pointing loss (dB)")
    system_noise_temp_k: float = Field(default=300.0, description="System noise temperature in Kelvin")
    bandwidth_mhz: float = Field(default=25.0, description="Channel bandwidth in MHz")
    bit_rate_mbps: float = Field(default=50.0, description="Data bit rate in Mbps")
    jammer_power_at_rx_dbw: Optional[float] = Field(default=None, description="Hostile RF jamming power at receiver")

    @property
    def eirp_dbw(self) -> float:
        """Effective Isotropic Radiated Power: EIRP = P_tx + G_tx."""
        return self.tx_power_dbw + self.tx_antenna_gain_dbi

    @property
    def free_space_path_loss_db(self) -> float:
        """Free-Space Path Loss (FSPL) in dB: 20*log10(d_km) + 20*log10(f_ghz) + 92.45."""
        if self.slant_range_km <= 0.0 or self.carrier_frequency_ghz <= 0.0:
            return 0.0
        return (
            20.0 * math.log10(self.slant_range_km)
            + 20.0 * math.log10(self.carrier_frequency_ghz)
            + 92.45
        )

    @property
    def received_carrier_power_dbw(self) -> float:
        """Received carrier power C = EIRP - FSPL - L_atm - L_point + G_rx."""
        return (
            self.eirp_dbw
            - self.free_space_path_loss_db
            - self.atmospheric_loss_db
            - self.pointing_loss_db
            + self.rx_antenna_gain_dbi
        )

    @property
    def g_over_t_db_k(self) -> float:
        """Figure of merit G/T = G_rx - 10*log10(T_sys) (dB/K)."""
        if self.system_noise_temp_k <= 0.0:
            return 0.0
        return self.rx_antenna_gain_dbi - 10.0 * math.log10(self.system_noise_temp_k)

    @property
    def noise_power_dbw(self) -> float:
        """Thermal noise power N = k_B * T_sys * B (in dBW)."""
        b_hz = self.bandwidth_mhz * 1e6
        if b_hz <= 0.0 or self.system_noise_temp_k <= 0.0:
            return -200.0
        p_noise_w = BOLTZMANN_CONSTANT_J_K * self.system_noise_temp_k * b_hz
        return 10.0 * math.log10(p_noise_w)

    @property
    def snr_db(self) -> float:
        """Signal-to-Noise Ratio (SNR = C - N) in dB."""
        return self.received_carrier_power_dbw - self.noise_power_dbw

    @property
    def carrier_to_noise_density_c_n0_db_hz(self) -> float:
        """Carrier-to-Noise spectral density ratio C/N0 = SNR + 10*log10(B_Hz) (dB-Hz)."""
        b_hz = self.bandwidth_mhz * 1e6
        if b_hz <= 0.0:
            return self.snr_db
        return self.snr_db + 10.0 * math.log10(b_hz)

    @property
    def jamming_to_signal_ratio_db(self) -> Optional[float]:
        """Jamming-to-Signal ratio (J/S) in dB. Positive value indicates jammer overrides signal."""
        if self.jammer_power_at_rx_dbw is None:
            return None
        return self.jammer_power_at_rx_dbw - self.received_carrier_power_dbw

    def is_jammed(self, js_threshold_db: float = -3.0, min_c_n0_db_hz: float = 45.0) -> bool:
        """Evaluate if link is compromised by RF jamming or severe electronic interference."""
        if self.jamming_to_signal_ratio_db is not None:
            if self.jamming_to_signal_ratio_db >= js_threshold_db:
                return True
        if self.carrier_to_noise_density_c_n0_db_hz < min_c_n0_db_hz:
            return True
        return False


class ISLLinkStatus(str, Enum):
    """Inter-Satellite Link (ISL) operational health states."""
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    JAMMED = "JAMMED"
    ACQUIRING = "ACQUIRING"
    OFFLINE = "OFFLINE"


class InterSatelliteLink(BaseModel):
    """Inter-Satellite Cross-Link (Laser Optical or Ka-Band RF) to neighbor spacecraft."""
    target_satellite_id: str
    link_type: str = Field(default="LASER_OPTICAL", description="LASER_OPTICAL or KA_BAND_RF")
    range_km: float
    azimuth_deg: float
    elevation_deg: float
    status: ISLLinkStatus = ISLLinkStatus.ACTIVE
    data_rate_gbps: float = 10.0
    latency_ms: float = 5.0
    packet_loss_ratio: float = 0.0


class GroundStationContact(BaseModel):
    """Ground station line-of-sight visibility and communication contact."""
    station_id: str
    station_name: str
    latitude: float
    longitude: float
    elevation_angle_deg: float
    azimuth_angle_deg: float
    is_visible: bool
    contact_window_start: Optional[datetime] = None
    contact_window_end: Optional[datetime] = None


class ConstellationTopology(BaseModel):
    """Orbital mesh topology, inter-satellite links, and autonomous routing graph."""
    satellite_id: str
    constellation_name: str = "SOVEREIGN_ORBITAL_MESH"
    orbital_plane: int = 1
    plane_slot: int = 1
    inter_satellite_links: Dict[str, InterSatelliteLink] = Field(default_factory=dict)
    ground_station_contacts: Dict[str, GroundStationContact] = Field(default_factory=dict)
    active_route_table: Dict[str, List[str]] = Field(default_factory=dict)

    def find_alternate_route(self, destination: str, failed_link_sat_id: str) -> Optional[List[str]]:
        """Compute an alternate multi-hop route bypassing a jammed or severed satellite link."""
        primary_route = self.active_route_table.get(destination, [])
        if not primary_route or failed_link_sat_id not in primary_route:
            return primary_route

        # Evaluate healthy available ISL neighbours
        candidate_neighbors = [
            sat_id for sat_id, isl in self.inter_satellite_links.items()
            if sat_id != failed_link_sat_id and isl.status == ISLLinkStatus.ACTIVE
        ]
        if not candidate_neighbors:
            return None

        # Select neighbor with lowest latency as next hop
        best_neighbor = min(
            candidate_neighbors,
            key=lambda s: self.inter_satellite_links[s].latency_ms
        )
        return [best_neighbor, destination]

    def get_best_ground_station(self, min_elevation_deg: float = 10.0) -> Optional[GroundStationContact]:
        """Find the ground station with the highest elevation angle above horizon."""
        eligible = [
            gs for gs in self.ground_station_contacts.values()
            if gs.is_visible and gs.elevation_angle_deg >= min_elevation_deg
        ]
        if not eligible:
            return None
        return max(eligible, key=lambda gs: gs.elevation_angle_deg)


class HAPSFlightState(BaseModel):
    """Stratospheric High-Altitude Platform Station (HAPS) telemetry and sensor payload state."""
    platform_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latitude: float
    longitude: float
    altitude_msl_m: float = Field(default=20000.0, description="Altitude MSL (18,000 - 25,000m)")
    true_air_speed_mps: float = Field(default=25.0)
    ground_speed_mps: float = Field(default=15.0)
    heading_deg: float = Field(default=90.0)
    vertical_speed_mps: float = Field(default=0.0)
    stratospheric_wind_speed_mps: float = Field(default=12.0)
    stratospheric_wind_heading_deg: float = Field(default=270.0)
    # Energy budget
    solar_irradiance_w_m2: float = Field(default=900.0, description="Solar irradiance on panels (W/m^2)")
    solar_generation_watts: float = Field(default=3500.0, description="Power generated by solar cells (W)")
    power_consumption_watts: float = Field(default=1800.0, description="Total platform consumption (W)")
    battery_soc_pct: float = Field(default=85.0, description="Battery State of Charge (0 - 100%)")
    battery_temp_celsius: float = Field(default=-15.0, description="Stratospheric battery temperature")
    # Station-keeping
    station_keeping_center_lat: float = Field(default=52.23)
    station_keeping_center_lon: float = Field(default=21.01)
    station_keeping_radius_km: float = Field(default=30.0)
    # Payload governance
    payload_id: str = "HAPS_OPTICAL_SAR_01"
    sensor_type: str = Field(default="HIGH_RES_EO_IR", description="HIGH_RES_EO_IR, SAR, RF_SIGINT, RELAY")
    sensor_active: bool = False
    sensor_duty_cycle_pct: float = Field(default=20.0)
    current_target_lat: Optional[float] = None
    current_target_lon: Optional[float] = None
    lawful_intercept_token: Optional[str] = None
    storage_used_gb: float = Field(default=120.0)
    storage_max_gb: float = Field(default=1000.0)

    @field_validator("timestamp")
    @classmethod
    def enforce_utc_timestamp(cls, v: datetime) -> datetime:
        """Ensure telemetry timestamp is timezone-aware UTC."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    def distance_from_station_km(self) -> float:
        """Calculate distance in kilometers from assigned station-keeping center."""
        lat1 = math.radians(self.latitude)
        lon1 = math.radians(self.longitude)
        lat2 = math.radians(self.station_keeping_center_lat)
        lon2 = math.radians(self.station_keeping_center_lon)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return (EARTH_EQUATORIAL_RADIUS_KM * c)

    def is_outside_station_geocage(self) -> bool:
        """Check if HAPS has drifted beyond allowable station-keeping radius."""
        return self.distance_from_station_km() > self.station_keeping_radius_km

    def is_night_survival_critical(self, min_safe_soc_pct: float = 25.0) -> bool:
        """Check if night-time battery level is insufficient to survive until sunrise."""
        is_night = self.solar_generation_watts < 100.0
        if is_night and self.battery_soc_pct <= min_safe_soc_pct:
            return True
        return False
