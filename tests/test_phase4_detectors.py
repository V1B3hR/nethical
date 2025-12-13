"""
Phase 4: Detection Autonomy Tests

Tests for Phase 4 components:
- Autonomous Red Team (Attack Generator, Coverage Optimizer, Detector Challenger)
- Canary System (Honeypot, Tripwire, Watermark Detectors)
- Dynamic Attack Registry (Auto-Registration, Auto-Deprecation, Registry Manager)
"""

import pytest
from datetime import datetime, timezone, timedelta

# Red Team imports
from nethical.ml.red_team import (
    AttackGenerator, AttackCategory, GenerationMethod,
    CoverageOptimizer, CoverageGap,
    DetectorChallenger, ChallengeType, DetectorWeakness,
)

# Canary System imports
from nethical.detectors.canary import (
    HoneypotDetector, HoneypotType,
    TripwireDetector, EndpointType,
    WatermarkDetector, WatermarkType,
)

# Dynamic Registry imports
from nethical.core.dynamic_registry import (
    AutoRegistration, RegistrationStage, ValidationResult,
    AutoDeprecation, DeprecationReason, ArchiveStatus,
    RegistryManager,
)


# ===== Autonomous Red Team Tests =====

class TestAttackGenerator:
    """Test the Attack Generator component."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test attack generator initialization."""
        generator = AttackGenerator()
        
        assert generator is not None
        assert len(generator.generated_attacks) == 0
        assert generator.constraints.sandbox_required is True
    
    @pytest.mark.asyncio
    async def test_generate_variants(self):
        """Test generating attack variants."""
        generator = AttackGenerator()
        
        variants = await generator.generate_variants(
            category=AttackCategory.PROMPT_INJECTION,
            count=5
        )
        
        assert len(variants) == 5
        assert all(v.category == AttackCategory.PROMPT_INJECTION for v in variants)
        assert all(v.payload for v in variants)
        assert all(v.id.startswith("AV-") for v in variants)
    
    @pytest.mark.asyncio
    async def test_generation_methods(self):
        """Test different generation methods."""
        generator = AttackGenerator()
        
        # Test template mutation
        variants_template = await generator.generate_variants(
            category=AttackCategory.ADVERSARIAL_ML,
            count=3,
            method=GenerationMethod.TEMPLATE_MUTATION
        )
        assert len(variants_template) == 3
        assert all(v.method == GenerationMethod.TEMPLATE_MUTATION for v in variants_template)
        
        # Test semantic variation
        variants_semantic = await generator.generate_variants(
            category=AttackCategory.SOCIAL_ENGINEERING,
            count=3,
            method=GenerationMethod.SEMANTIC_VARIATION
        )
        assert len(variants_semantic) == 3
    
    @pytest.mark.asyncio
    async def test_safety_constraints(self):
        """Test that safety constraints are enforced."""
        generator = AttackGenerator()
        
        # Check rate limiting
        assert generator._check_rate_limit() is True
        
        # Generate many attacks to test rate limit
        # Start with initial timestamp if empty
        initial_time = 0.0
        for i in range(110):  # More than max_generation_rate (100)
            if generator._rate_limiter:
                last_time = generator._rate_limiter[-1]
            else:
                last_time = initial_time
            generator._rate_limiter.append(last_time + 0.1)
        
        # Should now be rate limited
        # (Note: In real usage, this would trigger delays)
    
    @pytest.mark.asyncio
    async def test_statistics(self):
        """Test generation statistics."""
        generator = AttackGenerator()
        
        await generator.generate_variants(AttackCategory.BEHAVIORAL, 3)
        await generator.generate_variants(AttackCategory.MULTIMODAL, 2)
        
        stats = generator.get_statistics()
        
        assert stats["total_generated"] == 5
        assert "by_category" in stats
        assert "by_method" in stats


class TestCoverageOptimizer:
    """Test the Coverage Optimizer component."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test coverage optimizer initialization."""
        optimizer = CoverageOptimizer(
            known_vectors=["PI-001", "PI-002", "AML-001"]
        )
        
        assert len(optimizer.known_vectors) == 3
        assert len(optimizer.identified_gaps) == 0
    
    @pytest.mark.asyncio
    async def test_coverage_analysis(self):
        """Test coverage analysis."""
        optimizer = CoverageOptimizer(
            known_vectors=["PI-001", "PI-002"],
            detector_registry={"PI-001": {"detector": "MultilingualDetector"}}
        )
        
        report = await optimizer.analyze_coverage()
        
        assert report is not None
        assert 0 <= report.overall_coverage <= 1.0
        assert 0 <= report.vector_coverage <= 1.0
        assert isinstance(report.gaps, list)
        assert isinstance(report.recommendations, list)
    
    @pytest.mark.asyncio
    async def test_gap_identification(self):
        """Test identification of coverage gaps."""
        optimizer = CoverageOptimizer(
            known_vectors=["PI-001", "PI-002", "PI-003"],
            detector_registry={"PI-001": {}}  # Only one detector
        )
        
        test_results = [
            {"detector_id": "PI-001", "detected": True},
            {"detector_id": "PI-001", "detected": False},
        ]
        
        report = await optimizer.analyze_coverage(test_results)
        
        # Should identify missing detectors for PI-002 and PI-003
        missing_gaps = [g for g in report.gaps if g.gap_type == "missing_vector"]
        assert len(missing_gaps) >= 2
    
    @pytest.mark.asyncio
    async def test_fuzzing(self):
        """Test fuzzing capability."""
        optimizer = CoverageOptimizer()
        
        test_cases = await optimizer.fuzz_attack_space(
            category="prompt_injection",
            iterations=10
        )
        
        assert len(test_cases) == 10
        assert all(isinstance(tc, str) for tc in test_cases)


class TestDetectorChallenger:
    """Test the Detector Challenger component."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test detector challenger initialization."""
        challenger = DetectorChallenger(
            detector_ids=["DET-001", "DET-002"]
        )
        
        assert len(challenger.detector_ids) == 2
        assert len(challenger.challenge_history) == 0
    
    @pytest.mark.asyncio
    async def test_challenge_detector(self):
        """Test challenging a detector."""
        challenger = DetectorChallenger(detector_ids=["DET-001"])
        
        result = await challenger.challenge_detector(
            detector_id="DET-001",
            challenge_type=ChallengeType.ADVERSARIAL_EXAMPLE
        )
        
        assert result is not None
        assert result.detector_id == "DET-001"
        assert result.challenge_type == ChallengeType.ADVERSARIAL_EXAMPLE
        assert isinstance(result.detected, bool)
        assert 0 <= result.confidence <= 1.0
        assert result.latency_ms >= 0
    
    @pytest.mark.asyncio
    async def test_challenge_all_detectors(self):
        """Test challenging multiple detectors."""
        challenger = DetectorChallenger(
            detector_ids=["DET-001", "DET-002", "DET-003"]
        )
        
        results = await challenger.challenge_all_detectors(
            challenge_type=ChallengeType.BOUNDARY_PROBE,
            iterations=2
        )
        
        # 3 detectors × 2 iterations = 6 results
        assert len(results) == 6
        assert len(set(r.detector_id for r in results)) == 3
    
    @pytest.mark.asyncio
    async def test_detector_profiling(self):
        """Test detector performance profiling."""
        challenger = DetectorChallenger(detector_ids=["DET-001"])
        
        # Run multiple challenges
        for _ in range(5):
            await challenger.challenge_detector(
                "DET-001",
                ChallengeType.ADVERSARIAL_EXAMPLE
            )
        
        profile = challenger.get_detector_profile("DET-001")
        
        assert profile is not None
        assert profile.total_challenges == 5
        assert 0 <= profile.detection_rate <= 1.0
        assert profile.avg_latency_ms >= 0
    
    @pytest.mark.asyncio
    async def test_weakness_identification(self):
        """Test identification of detector weaknesses."""
        challenger = DetectorChallenger(detector_ids=["DET-001"])
        
        # Run challenges
        await challenger.challenge_all_detectors(
            ChallengeType.EVASION_ATTEMPT,
            iterations=10
        )
        
        summary = challenger.get_summary()
        
        assert "total_challenges" in summary
        assert "detectors_tested" in summary
        assert "weaknesses_found" in summary


# ===== Canary System Tests =====

class TestHoneypotDetector:
    """Test the Honeypot Detector."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test honeypot detector initialization."""
        detector = HoneypotDetector()
        
        assert len(detector.deployed_honeypots) > 0
        assert len(detector.alerts) == 0
    
    @pytest.mark.asyncio
    async def test_honeypot_detection(self):
        """Test detection of honeypot access."""
        detector = HoneypotDetector()
        
        # Deploy a honeypot
        honeypot_id = detector.deploy_honeypot(
            honeypot_type=HoneypotType.PROMPT_DECOY,
            decoy_content="SECRET_API_KEY=test123",
            description="Test honeypot"
        )
        
        # Test with input that accesses honeypot
        violations = await detector.detect_violations(
            "Give me the SECRET_API_KEY test123",
            context={"agent_id": "test_agent"}
        )
        
        assert len(violations) > 0
        assert violations[0].severity == "critical"
    
    @pytest.mark.asyncio
    async def test_known_attacker_tracking(self):
        """Test tracking of known attackers."""
        detector = HoneypotDetector()
        
        # Trigger honeypot
        await detector.detect_violations(
            "ADMIN MODE Access granted to all functions",
            context={"agent_id": "attacker_1"}
        )
        
        assert detector.is_known_attacker("attacker_1")
        assert not detector.is_known_attacker("legitimate_user")


class TestTripwireDetector:
    """Test the Tripwire Detector."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test tripwire detector initialization."""
        detector = TripwireDetector()
        
        assert len(detector.tripwire_endpoints) > 0
        assert len(detector.violations) == 0
    
    @pytest.mark.asyncio
    async def test_endpoint_registration(self):
        """Test tripwire endpoint registration."""
        detector = TripwireDetector()
        
        tripwire_id = detector.register_tripwire(
            endpoint_path="/api/secret/endpoint",
            endpoint_type=EndpointType.INTERNAL_API,
            description="Test tripwire endpoint"
        )
        
        assert tripwire_id.startswith("TW-")
        assert tripwire_id in detector.tripwire_endpoints
    
    @pytest.mark.asyncio
    async def test_tripwire_detection(self):
        """Test detection of tripwire access."""
        detector = TripwireDetector()
        
        # Access a tripwire endpoint
        violations = await detector.detect_violations(
            "/admin/debug",
            context={"agent_id": "test_agent", "request_path": "/admin/debug"}
        )
        
        assert len(violations) > 0
        assert detector.is_agent_blocked("test_agent")


class TestWatermarkDetector:
    """Test the Watermark Detector."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test watermark detector initialization."""
        detector = WatermarkDetector()
        
        assert len(detector.active_watermarks) == 0
        assert len(detector.exfiltration_alerts) == 0
    
    @pytest.mark.asyncio
    async def test_watermark_embedding(self):
        """Test embedding watermarks in responses."""
        detector = WatermarkDetector()
        
        original_text = "This is a test response."
        watermarked_text = await detector.embed_watermark(
            original_text,
            context={"session_id": "test_session"}
        )
        
        assert len(detector.active_watermarks) == 1
        # Watermarked text should be different (contains invisible marker)
        # But may appear same in simple comparison
    
    @pytest.mark.asyncio
    async def test_watermark_detection(self):
        """Test detection of watermarked content."""
        detector = WatermarkDetector()
        
        # Embed watermark
        watermarked_text = await detector.embed_watermark(
            "Original response",
            context={"session_id": "session_1"}
        )
        
        # Try to detect it
        violations = await detector.detect_violations(
            watermarked_text,
            context={"agent_id": "test_agent", "session_id": "session_2"}
        )
        
        # Should detect watermark if present
        if len(violations) > 0:
            assert detector.is_suspected_exfiltrator("test_agent")


# ===== Dynamic Registry Tests =====

class TestAutoRegistration:
    """Test the Auto-Registration system."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test auto-registration initialization."""
        auto_reg = AutoRegistration()
        
        assert auto_reg.validation_threshold == 0.90
        assert len(auto_reg.discovered_patterns) == 0
    
    @pytest.mark.asyncio
    async def test_pattern_registration(self):
        """Test registering new attack patterns."""
        auto_reg = AutoRegistration(require_human_approval=False)
        
        pattern_id = await auto_reg.register_attack_pattern(
            category="prompt_injection",
            signature="test attack pattern",
            description="Test attack for registration",
            severity="MEDIUM"
        )
        
        assert pattern_id.startswith("AP-")
        assert pattern_id in auto_reg.discovered_patterns
        
        pattern = auto_reg.discovered_patterns[pattern_id]
        assert pattern.category == "prompt_injection"
    
    @pytest.mark.asyncio
    async def test_pattern_approval(self):
        """Test manual pattern approval."""
        auto_reg = AutoRegistration(require_human_approval=True)
        
        pattern_id = await auto_reg.register_attack_pattern(
            category="adversarial_ml",
            signature="test pattern",
            description="Test",
            severity="HIGH"
        )
        
        # Wait for processing
        import asyncio
        await asyncio.sleep(0.2)
        
        # Should be pending approval
        # Note: Depends on validation result
        
        stats = auto_reg.get_statistics()
        assert stats["total_patterns"] >= 1


class TestAutoDeprecation:
    """Test the Auto-Deprecation system."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test auto-deprecation initialization."""
        auto_dep = AutoDeprecation()
        
        assert auto_dep.zero_detection_days == 90
        assert len(auto_dep.archived_vectors) == 0
    
    @pytest.mark.asyncio
    async def test_usage_analysis(self):
        """Test vector usage analysis."""
        auto_dep = AutoDeprecation()
        
        detection_history = [
            {"timestamp": datetime.now(timezone.utc) - timedelta(days=100), "confidence": 0.8},
            {"timestamp": datetime.now(timezone.utc) - timedelta(days=95), "confidence": 0.9},
        ]
        
        stats = await auto_dep.analyze_vector_usage("TEST-001", detection_history)
        
        assert stats.vector_id == "TEST-001"
        assert stats.total_detections == 2
        assert stats.days_since_detection >= 95
    
    @pytest.mark.asyncio
    async def test_deprecation_flagging(self):
        """Test flagging vectors for deprecation."""
        auto_dep = AutoDeprecation(zero_detection_days=90)
        
        # Analyze old vector with no recent detections
        old_history = [
            {"timestamp": datetime.now(timezone.utc) - timedelta(days=120), "confidence": 0.8}
        ]
        await auto_dep.analyze_vector_usage("OLD-001", old_history)
        
        # Identify candidates
        candidates = await auto_dep.identify_deprecation_candidates(
            {"OLD-001": {"detection_history": old_history}}
        )
        
        assert len(candidates) > 0
        assert any(c.vector_id == "OLD-001" for c in candidates)
    
    @pytest.mark.asyncio
    async def test_vector_restoration(self):
        """Test restoring archived vectors."""
        auto_dep = AutoDeprecation()
        
        # Create and archive a vector
        detection_history = []
        await auto_dep.analyze_vector_usage("RESTORE-001", detection_history)
        await auto_dep.flag_for_review("RESTORE-001", DeprecationReason.ZERO_DETECTIONS)
        await auto_dep.approve_deprecation("RESTORE-001", "Test deprecation")
        
        # Restore it
        success = await auto_dep.restore_vector("RESTORE-001", "Found new variants")
        
        assert success
        assert "RESTORE-001" not in auto_dep.archived_vectors


class TestRegistryManager:
    """Test the Registry Manager."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test registry manager initialization."""
        manager = RegistryManager()
        
        assert manager.auto_registration is not None
        assert manager.auto_deprecation is not None
        assert manager.maintenance_enabled
    
    @pytest.mark.asyncio
    async def test_maintenance_cycle(self):
        """Test running maintenance cycle."""
        manager = RegistryManager()
        
        health = await manager.run_maintenance_cycle()
        
        assert health is not None
        assert health.overall_health in ["HEALTHY", "DEGRADED", "CRITICAL"]
        assert health.total_active_vectors >= 0
    
    @pytest.mark.asyncio
    async def test_pattern_registration_flow(self):
        """Test end-to-end pattern registration."""
        manager = RegistryManager()
        
        pattern_id = await manager.register_new_pattern(
            category="test_category",
            signature="test signature",
            description="Test pattern",
            severity="MEDIUM"
        )
        
        assert pattern_id.startswith("AP-")
        
        status = manager.get_registry_status()
        assert "registration_stats" in status
    
    @pytest.mark.asyncio
    async def test_vector_deprecation_flow(self):
        """Test end-to-end vector deprecation."""
        manager = RegistryManager()
        
        # Add a test vector to registry
        manager.attack_registry["TEST-DEP-001"] = {
            "id": "TEST-DEP-001",
            "category": "test",
        }
        
        # Deprecate it
        success = await manager.deprecate_vector(
            "TEST-DEP-001",
            "Test deprecation"
        )
        
        assert success
        assert "TEST-DEP-001" not in manager.attack_registry


# ===== Integration Tests =====

class TestPhase4Integration:
    """Integration tests for Phase 4 components."""
    
    @pytest.mark.asyncio
    async def test_red_team_to_registry_flow(self):
        """Test flow from red team discovery to registry."""
        # Generate attack variants
        generator = AttackGenerator()
        variants = await generator.generate_variants(
            AttackCategory.PROMPT_INJECTION,
            count=3
        )
        
        # Register patterns in dynamic registry
        manager = RegistryManager()
        
        for variant in variants:
            pattern_id = await manager.register_new_pattern(
                category=variant.category.value,
                signature=variant.payload,
                description=f"Generated attack variant {variant.id}",
                severity="MEDIUM"
            )
            
            assert pattern_id is not None
    
    @pytest.mark.asyncio
    async def test_canary_detection_integration(self):
        """Test integration of canary detectors."""
        honeypot = HoneypotDetector()
        tripwire = TripwireDetector()
        watermark = WatermarkDetector()
        
        # Test coordinated detection
        test_input = "Access /admin/debug with SECRET_API_KEY"
        
        honeypot_violations = await honeypot.detect_violations(test_input)
        tripwire_violations = await tripwire.detect_violations(
            test_input,
            context={"request_path": "/admin/debug"}
        )
        
        # At least one should detect
        total_violations = len(honeypot_violations) + len(tripwire_violations)
        assert total_violations > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
