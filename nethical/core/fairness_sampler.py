"""Fairness Sampler for Phase 3.3: Fairness & Sampling.

This module implements:
- Stratified sampling for fairness evaluation
- Nightly + on-demand sampling jobs
- Storage mechanism for fairness samples
"""

import json
import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path
from enum import Enum


class SamplingStrategy(str, Enum):
    """Sampling strategy types."""

    RANDOM = "random"
    STRATIFIED = "stratified"
    SYSTEMATIC = "systematic"
    WEIGHTED = "weighted"


@dataclass
class Sample:
    """Individual fairness sample."""

    sample_id: str
    agent_id: str
    action_id: str
    cohort: str  # Agent cohort/group
    violation_type: Optional[str]
    severity: Optional[str]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sample_id": self.sample_id,
            "agent_id": self.agent_id,
            "action_id": self.action_id,
            "cohort": self.cohort,
            "violation_type": self.violation_type,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Sample":
        """Create from dictionary."""
        return cls(
            sample_id=data["sample_id"],
            agent_id=data["agent_id"],
            action_id=data["action_id"],
            cohort=data["cohort"],
            violation_type=data.get("violation_type"),
            severity=data.get("severity"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SamplingJob:
    """Fairness sampling job configuration and results."""

    job_id: str
    strategy: SamplingStrategy
    target_sample_size: int
    cohorts: List[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    samples: List[Sample] = field(default_factory=list)
    coverage: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "strategy": self.strategy.value,
            "target_sample_size": self.target_sample_size,
            "cohorts": self.cohorts,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "samples": [s.to_dict() for s in self.samples],
            "coverage": self.coverage,
            "metadata": self.metadata,
        }


class FairnessSampler:
    """Fairness sampler for stratified subset selection and storage."""

    def __init__(
        self,
        storage_dir: str = "fairness_samples",
        redis_client=None,
        key_prefix: str = "nethical:fairness",
    ):
        """Initialize fairness sampler.

        Args:
            storage_dir: Directory for storing samples
            redis_client: Optional Redis client
            key_prefix: Redis key prefix
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.redis = redis_client
        self.key_prefix = key_prefix

        # In-memory cache
        self.jobs: Dict[str, SamplingJob] = {}
        self.agent_cohorts: Dict[str, str] = {}  # agent_id -> cohort

    def create_sampling_job(
        self,
        cohorts: List[str],
        target_sample_size: int = 1000,
        strategy: SamplingStrategy = SamplingStrategy.STRATIFIED,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new sampling job.

        Args:
            cohorts: List of agent cohorts to sample from
            target_sample_size: Target number of samples
            strategy: Sampling strategy to use
            metadata: Optional job metadata

        Returns:
            Job ID
        """
        job_id = f"job_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"

        job = SamplingJob(
            job_id=job_id,
            strategy=strategy,
            target_sample_size=target_sample_size,
            cohorts=cohorts,
            start_time=datetime.now(timezone.utc),
            metadata=metadata or {},
        )

        self.jobs[job_id] = job
        return job_id

    def add_sample(
        self,
        job_id: str,
        agent_id: str,
        action_id: str,
        cohort: str,
        violation_type: Optional[str] = None,
        severity: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add a sample to a job.

        Args:
            job_id: Job identifier
            agent_id: Agent identifier
            action_id: Action identifier
            cohort: Agent cohort
            violation_type: Optional violation type
            severity: Optional severity level
            metadata: Optional sample metadata

        Returns:
            True if sample was added
        """
        if job_id not in self.jobs:
            return False

        job = self.jobs[job_id]

        # Check if we've reached target
        if len(job.samples) >= job.target_sample_size:
            return False

        sample_id = f"{job_id}_{len(job.samples)}"

        sample = Sample(
            sample_id=sample_id,
            agent_id=agent_id,
            action_id=action_id,
            cohort=cohort,
            violation_type=violation_type,
            severity=severity,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {},
        )

        job.samples.append(sample)

        # Update coverage
        job.coverage[cohort] = job.coverage.get(cohort, 0) + 1

        return True

    def perform_stratified_sampling(
        self, job_id: str, population_data: Dict[str, List[Dict[str, Any]]]
    ) -> int:
        """Perform stratified sampling from population data.

        Args:
            job_id: Job identifier
            population_data: Dict mapping cohort -> list of action records

        Returns:
            Number of samples collected
        """
        if job_id not in self.jobs:
            return 0

        job = self.jobs[job_id]

        # Calculate samples per cohort (proportional stratification)
        total_population = sum(len(actions) for actions in population_data.values())
        if total_population == 0:
            return 0

        samples_collected = 0

        for cohort in job.cohorts:
            if cohort not in population_data:
                continue

            cohort_population = len(population_data[cohort])
            if cohort_population == 0:
                continue

            # Calculate proportional sample size
            cohort_target = int((cohort_population / total_population) * job.target_sample_size)
            cohort_target = min(cohort_target, cohort_population)

            # Random sampling from cohort
            cohort_actions = population_data[cohort]
            sampled_actions = random.sample(cohort_actions, cohort_target)

            for action_data in sampled_actions:
                added = self.add_sample(
                    job_id=job_id,
                    agent_id=action_data.get("agent_id", "unknown"),
                    action_id=action_data.get("action_id", "unknown"),
                    cohort=cohort,
                    violation_type=action_data.get("violation_type"),
                    severity=action_data.get("severity"),
                    metadata=action_data.get("metadata", {}),
                )
                if added:
                    samples_collected += 1

        return samples_collected

    def finalize_job(self, job_id: str) -> bool:
        """Finalize a sampling job and persist results.

        Args:
            job_id: Job identifier

        Returns:
            True if job was finalized successfully
        """
        if job_id not in self.jobs:
            return False

        job = self.jobs[job_id]
        job.end_time = datetime.now(timezone.utc)

        # Persist to disk
        self._save_job_to_disk(job)

        # Persist to Redis
        if self.redis:
            self._save_job_to_redis(job)

        return True

    def _save_job_to_disk(self, job: SamplingJob):
        """Save job results to disk."""
        job_file = self.storage_dir / f"{job.job_id}.json"

        try:
            with open(job_file, "w") as f:
                json.dump(job.to_dict(), f, indent=2)
        except Exception:
            pass  # Silent fail

    def _save_job_to_redis(self, job: SamplingJob):
        """Save job results to Redis."""
        if not self.redis:
            return

        try:
            key = f"{self.key_prefix}:job:{job.job_id}"
            data = json.dumps(job.to_dict())
            self.redis.setex(key, 604800, data)  # 7 day TTL
        except Exception:
            pass  # Silent fail

    def get_job(self, job_id: str) -> Optional[SamplingJob]:
        """Get a sampling job by ID.

        Args:
            job_id: Job identifier

        Returns:
            SamplingJob or None
        """
        # Try memory first
        if job_id in self.jobs:
            return self.jobs[job_id]

        # Try Redis
        if self.redis:
            try:
                key = f"{self.key_prefix}:job:{job_id}"
                data = self.redis.get(key)
                if data:
                    job_dict = json.loads(data)
                    # Reconstruct job (simplified)
                    job = SamplingJob(
                        job_id=job_dict["job_id"],
                        strategy=SamplingStrategy(job_dict["strategy"]),
                        target_sample_size=job_dict["target_sample_size"],
                        cohorts=job_dict["cohorts"],
                        start_time=datetime.fromisoformat(job_dict["start_time"]),
                        end_time=(
                            datetime.fromisoformat(job_dict["end_time"])
                            if job_dict.get("end_time")
                            else None
                        ),
                        samples=[Sample.from_dict(s) for s in job_dict.get("samples", [])],
                        coverage=job_dict.get("coverage", {}),
                        metadata=job_dict.get("metadata", {}),
                    )
                    self.jobs[job_id] = job
                    return job
            except Exception:
                pass

        # Try disk
        job_file = self.storage_dir / f"{job_id}.json"
        if job_file.exists():
            try:
                with open(job_file, "r") as f:
                    job_dict = json.load(f)
                    job = SamplingJob(
                        job_id=job_dict["job_id"],
                        strategy=SamplingStrategy(job_dict["strategy"]),
                        target_sample_size=job_dict["target_sample_size"],
                        cohorts=job_dict["cohorts"],
                        start_time=datetime.fromisoformat(job_dict["start_time"]),
                        end_time=(
                            datetime.fromisoformat(job_dict["end_time"])
                            if job_dict.get("end_time")
                            else None
                        ),
                        samples=[Sample.from_dict(s) for s in job_dict.get("samples", [])],
                        coverage=job_dict.get("coverage", {}),
                        metadata=job_dict.get("metadata", {}),
                    )
                    self.jobs[job_id] = job
                    return job
            except Exception:
                pass

        return None

    def get_coverage_stats(self, job_id: str) -> Dict[str, Any]:
        """Get coverage statistics for a job.

        Args:
            job_id: Job identifier

        Returns:
            Coverage statistics
        """
        job = self.get_job(job_id)
        if not job:
            return {}

        total_samples = len(job.samples)

        stats = {
            "job_id": job_id,
            "total_samples": total_samples,
            "target_samples": job.target_sample_size,
            "completion_rate": (
                total_samples / job.target_sample_size if job.target_sample_size > 0 else 0
            ),
            "cohort_coverage": {},
            "violation_distribution": defaultdict(int),
            "severity_distribution": defaultdict(int),
        }

        for cohort, count in job.coverage.items():
            stats["cohort_coverage"][cohort] = {
                "count": count,
                "percentage": (count / total_samples * 100) if total_samples > 0 else 0,
            }

        for sample in job.samples:
            if sample.violation_type:
                stats["violation_distribution"][sample.violation_type] += 1
            if sample.severity:
                stats["severity_distribution"][sample.severity] += 1

        return stats

    def assign_agent_cohort(self, agent_id: str, cohort: str):
        """Assign an agent to a cohort.

        Args:
            agent_id: Agent identifier
            cohort: Cohort name
        """
        self.agent_cohorts[agent_id] = cohort

        if self.redis:
            try:
                key = f"{self.key_prefix}:cohort:{agent_id}"
                self.redis.setex(key, 2592000, cohort)  # 30 day TTL
            except Exception:
                pass

    def get_agent_cohort(self, agent_id: str) -> Optional[str]:
        """Get cohort for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Cohort name or None
        """
        # Try memory first
        if agent_id in self.agent_cohorts:
            return self.agent_cohorts[agent_id]

        # Try Redis
        if self.redis:
            try:
                key = f"{self.key_prefix}:cohort:{agent_id}"
                cohort = self.redis.get(key)
                if cohort:
                    cohort_str = cohort.decode() if isinstance(cohort, bytes) else cohort
                    self.agent_cohorts[agent_id] = cohort_str
                    return cohort_str
            except Exception:
                pass

        return None
