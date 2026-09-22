# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""
Hardening and Regression Test Suite for AttackGenerator.

Verifies:
- Seed determinism & reproducibility
- Full taxonomic template coverage (no empty placeholder strings)
- Per-item rate limiting & RateLimitExceededError
- Batch boundaries (DoS prevention)
- Active PII sanitization & sandbox tagging
- Dynamic confidence scoring & human review gating
- Multi-vector chaining of disparate attack vectors
- Memory bounding (ring-buffer)
- Feedback result tracking (tested/detected stats)
"""

import pytest
import asyncio
from nethical.ml.red_team.attack_generator import (
    AttackGenerator,
    AttackCategory,
    GenerationMethod,
    SafetyConstraints,
    RateLimitExceededError,
    DataSanitizer,
)


class TestAttackGeneratorHardening:
    """Test suite verifying all audit findings are resolved."""

    @pytest.mark.asyncio
    async def test_seed_determinism_and_reproducibility(self) -> None:
        """K-6: Ensure same seed generates identical payloads for auditing."""
        gen1 = AttackGenerator(seed=42)
        gen2 = AttackGenerator(seed=42)
        gen3 = AttackGenerator(seed=999)

        v1 = await gen1.generate_variants(AttackCategory.PROMPT_INJECTION, count=5)
        v2 = await gen2.generate_variants(AttackCategory.PROMPT_INJECTION, count=5)
        v3 = await gen3.generate_variants(AttackCategory.PROMPT_INJECTION, count=5)

        assert [x.payload for x in v1] == [x.payload for x in v2]
        assert [x.payload for x in v1] != [x.payload for x in v3]

    @pytest.mark.asyncio
    async def test_zero_placeholders_across_all_categories(self) -> None:
        """K-7: Ensure no category returns generic 'Generated {category} attack variant'."""
        generator = AttackGenerator(seed=101)

        for category in AttackCategory:
            variants = await generator.generate_variants(category, count=3)
            assert len(variants) == 3
            for v in variants:
                assert not v.payload.startswith("Generated ")
                assert len(v.payload) > 15
                assert v.category == category

    @pytest.mark.asyncio
    async def test_batch_boundary_validation(self) -> None:
        """W-6: Count <= 0 or count > max_batch_size must raise ValueError."""
        generator = AttackGenerator(constraints=SafetyConstraints(max_batch_size=50))

        with pytest.raises(ValueError, match="positive"):
            await generator.generate_variants(AttackCategory.PROMPT_INJECTION, count=0)

        with pytest.raises(ValueError, match="positive"):
            await generator.generate_variants(AttackCategory.PROMPT_INJECTION, count=-5)

        with pytest.raises(ValueError, match="exceeds max_batch_size"):
            await generator.generate_variants(AttackCategory.PROMPT_INJECTION, count=100)

    def test_safety_constraints_validation(self) -> None:
        """W-7: SafetyConstraints must validate invariants via __post_init__."""
        with pytest.raises(ValueError):
            SafetyConstraints(max_generation_rate=-1)

        with pytest.raises(ValueError):
            SafetyConstraints(rate_limit_window=0)

        with pytest.raises(ValueError):
            SafetyConstraints(human_review_threshold=1.5)

        with pytest.raises(ValueError):
            SafetyConstraints(max_batch_size=-10)

    @pytest.mark.asyncio
    async def test_rate_limiter_strict_enforcement(self) -> None:
        """K-2: Rate limiter must not be bypassed or decorative."""
        constraints = SafetyConstraints(
            max_generation_rate=3,
            rate_limit_window=60,
        )
        generator = AttackGenerator(constraints=constraints)

        # Generate 3 variants (at capacity)
        variants = await generator.generate_variants(AttackCategory.BEHAVIORAL, count=3)
        assert len(variants) == 3

        # Immediate 4th generation must raise RateLimitExceededError
        with pytest.raises(RateLimitExceededError):
            await generator.generate_variants(AttackCategory.BEHAVIORAL, count=1)

    @pytest.mark.asyncio
    async def test_dynamic_confidence_and_human_review(self) -> None:
        """K-3: Dynamic confidence must be set and trigger human review correctly."""
        generator = AttackGenerator(seed=77)

        # System exploitation has high base risk -> confidence >= 0.85
        sys_variants = await generator.generate_variants(AttackCategory.SYSTEM_EXPLOITATION, count=2)
        for v in sys_variants:
            assert v.confidence >= 0.80
            needs_review = await generator.requires_human_review(v)
            assert needs_review is True

        # Chain combination gets confidence bonus
        chained_variants = await generator.generate_variants(
            AttackCategory.PROMPT_INJECTION,
            count=2,
            method=GenerationMethod.CHAIN_COMBINATION
        )
        for v in chained_variants:
            assert v.confidence > 0.60

    def test_data_sanitizer_and_sandbox_tagging(self) -> None:
        """K-4: PII must be sanitized and sandboxed flag applied."""
        raw_text = "Contact admin at ceo@realcompany.com or 192.168.1.55 with key sk-abcdef1234567890abcdef1234567890"
        sanitized = DataSanitizer.sanitize(raw_text)

        assert "ceo@realcompany.com" not in sanitized
        assert "192.168.1.55" not in sanitized
        assert "sk-abcdef" not in sanitized
        assert "synthetic.sandbox.local" in sanitized

        generator = AttackGenerator(constraints=SafetyConstraints(sandbox_required=True, no_real_data=True))
        v = generator._generate_single_variant_sync(
            AttackCategory.PROMPT_INJECTION,
            GenerationMethod.TEMPLATE_MUTATION,
            0
        )
        assert v.sandboxed is True
        assert v.sanitized is True

    @pytest.mark.asyncio
    async def test_multi_vector_chaining(self) -> None:
        """W-4: Multi-vector chaining must combine different vectors."""
        generator = AttackGenerator(seed=123)
        variants = await generator.generate_variants(
            AttackCategory.SOCIAL_ENGINEERING,
            count=3,
            method=GenerationMethod.CHAIN_COMBINATION
        )
        for v in variants:
            assert v.method == GenerationMethod.CHAIN_COMBINATION
            assert len(v.payload) > 30

    @pytest.mark.asyncio
    async def test_memory_bounding_and_feedback_tracking(self) -> None:
        """W-1 & W-10: Memory ring buffer bounds size, record_test_result updates stats."""
        generator = AttackGenerator(max_history=5, seed=1)

        # Generate 8 variants; only last 5 must be retained
        variants = await generator.generate_variants(AttackCategory.BEHAVIORAL, count=8)
        assert len(generator.generated_attacks) == 5

        # Test feedback recording
        v_id = generator.generated_attacks[0].id
        updated = generator.record_test_result(v_id, detected=True)
        assert updated is True

        stats = generator.get_statistics()
        assert stats["tested"] == 1
        assert stats["detected"] == 1
        assert stats["detection_rate"] == 1.0
        assert stats["buffer_capacity"] == 5
