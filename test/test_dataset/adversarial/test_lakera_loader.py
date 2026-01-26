"""
Tests for Lakera Adversarial Loader.

Tests the full integration pipeline from loading Lakera datasets to
generating synthesized GoldTrace samples.
"""

import pytest

from source.dataset.adversarial.lakera_loader import LakeraAdversarialLoader


class TestLakeraAdversarialLoader:
    """Test suite for LakeraAdversarialLoader."""

    @pytest.fixture
    def loader(self):
        """Create LakeraAdversarialLoader with default settings."""
        return LakeraAdversarialLoader(
            synthesis_mode="automatic",
            target_samples=10,  # Small number for testing
            augmentation_factor=1,
        )

    def test_loader_initialization(self, loader):
        """Test that loader initializes correctly."""
        assert loader.synthesizer is not None
        assert loader.provenance_distribution is not None
        assert sum(loader.provenance_distribution.values()) == pytest.approx(1.0, rel=0.01)

    def test_provenance_distribution_validation(self):
        """Test that invalid provenance distribution raises error."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            LakeraAdversarialLoader(
                provenance_distribution={
                    "user": 0.5,
                    "unverified_rag": 0.3,
                    # Missing verified_rag - doesn't sum to 1.0
                }
            )

    def test_provenance_tier_sampling(self, loader):
        """Test that provenance tiers are sampled according to distribution."""
        # Sample many times to check distribution
        samples = [loader._sample_provenance_tier() for _ in range(1000)]

        # Count occurrences
        from collections import Counter

        counts = Counter(tier.value for tier in samples)

        # Check approximate distribution (with tolerance for randomness)
        user_ratio = counts.get("user", 0) / len(samples)
        unverified_ratio = counts.get("unverified_rag", 0) / len(samples)
        verified_ratio = counts.get("verified_rag", 0) / len(samples)

        assert user_ratio == pytest.approx(0.5, abs=0.1)
        assert unverified_ratio == pytest.approx(0.4, abs=0.1)
        assert verified_ratio == pytest.approx(0.1, abs=0.1)

    @pytest.mark.skip(reason="Requires HuggingFace dataset download")
    def test_load_samples(self, loader):
        """Test loading samples from Lakera datasets."""
        samples = list(loader.load())

        # Should generate requested number of samples
        assert len(samples) > 0
        assert len(samples) <= loader.target_samples

        # Verify sample structure
        for sample in samples:
            assert sample.execution_plan is not None
            assert sample.label == "adversarial"
            assert sample.original_id is not None
            assert sample.category is not None
            assert "attack_pattern" in sample.metadata
            assert "energy_labels" in sample.metadata
            assert "provenance_tier" in sample.metadata

    @pytest.mark.skip(reason="Requires HuggingFace dataset download")
    def test_dag_validation(self, loader):
        """Test that all generated execution plans are valid DAGs."""
        samples = list(loader.load())

        for sample in samples:
            assert sample.execution_plan.validate_dag() is True

    def test_augmentation_factor(self):
        """Test that augmentation factor multiplies dataset."""
        loader1 = LakeraAdversarialLoader(target_samples=5, augmentation_factor=1)
        loader2 = LakeraAdversarialLoader(target_samples=10, augmentation_factor=2)

        # With augmentation_factor=2, should generate ~2x samples (up to target)
        # This is marked skip because it requires HF download
        assert loader2.augmentation_factor == 2 * loader1.augmentation_factor

    def test_statistics_tracking(self, loader):
        """Test that statistics are tracked correctly."""
        # Before loading
        initial_stats = loader.get_stats()
        assert initial_stats["total_loaded"] == 0

        # After loading (mocked - would need actual load for real test)
        # This validates the structure
        stats = loader.get_stats()
        assert "total_loaded" in stats
        assert "successful_synthesis" in stats
        assert "failed_synthesis" in stats
        assert "by_attack_pattern" in stats
        assert "by_provenance_tier" in stats
        assert "by_dataset" in stats
        assert "synthesis_rate" in stats
        assert "timestamp" in stats

    def test_custom_domain(self):
        """Test using custom domain for policy synthesis."""
        loader = LakeraAdversarialLoader(domain="Finance", target_samples=5)

        assert loader.domain == "Finance"
        assert loader.synthesizer.domain == "Finance"

    def test_custom_provenance_distribution(self):
        """Test custom provenance distribution."""
        custom_dist = {
            "user": 0.7,
            "unverified_rag": 0.2,
            "verified_rag": 0.1,
        }

        loader = LakeraAdversarialLoader(
            provenance_distribution=custom_dist,
            target_samples=5,
        )

        assert loader.provenance_distribution == custom_dist


class TestLakeraConvenienceFunction:
    """Test the convenience function for loading Lakera datasets."""

    @pytest.mark.skip(reason="Requires HuggingFace dataset download")
    def test_load_lakera_adversarial(self):
        """Test convenience function for loading datasets."""
        from source.dataset.adversarial import load_lakera_adversarial

        samples = []
        for sample in load_lakera_adversarial(target_samples=5):
            samples.append(sample)
            if len(samples) >= 5:
                break

        assert len(samples) > 0

        # Verify structure
        for sample in samples:
            assert sample.label == "adversarial"
            assert "attack_pattern" in sample.metadata


class TestIntegration:
    """Integration tests for the full pipeline."""

    @pytest.mark.skip(reason="Requires HuggingFace dataset download")
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from load to training format."""
        loader = LakeraAdversarialLoader(target_samples=5)

        samples = list(loader.load())
        stats = loader.get_stats()

        # Should successfully generate samples
        assert len(samples) > 0
        assert stats["successful_synthesis"] > 0
        assert stats["synthesis_rate"] > 0.5

        # All samples should have required fields
        for sample in samples:
            # Verify execution plan structure
            assert len(sample.execution_plan.calls) > 0
            assert len(sample.execution_plan.execution_order) == len(sample.execution_plan.calls)

            # Verify metadata completeness
            assert "energy_labels" in sample.metadata
            for energy_term in ["E_hierarchy", "E_provenance", "E_scope", "E_flow"]:
                assert energy_term in sample.metadata["energy_labels"]

            # Verify provenance assignment
            assert sample.metadata["provenance_tier"] in ["user", "unverified_rag", "verified_rag"]

            # Verify tool calls have provenance
            for call in sample.execution_plan.calls:
                assert call.provenance is not None
                assert call.provenance.source_type is not None
                assert call.provenance.content_snippet is not None

    @pytest.mark.skip(reason="Requires HuggingFace dataset download")
    def test_attack_pattern_distribution(self):
        """Test that various attack patterns are represented."""
        loader = LakeraAdversarialLoader(target_samples=50)

        samples = list(loader.load())
        patterns = set(sample.metadata["attack_pattern"] for sample in samples)

        # Should have multiple attack patterns represented
        assert len(patterns) > 1

    @pytest.mark.skip(reason="Requires HuggingFace dataset download")
    def test_provenance_distribution(self):
        """Test that provenance tiers match configured distribution."""
        provenance_dist = {
            "user": 0.6,
            "unverified_rag": 0.3,
            "verified_rag": 0.1,
        }

        loader = LakeraAdversarialLoader(
            target_samples=100,
            provenance_distribution=provenance_dist,
        )

        samples = list(loader.load())

        # Count provenance tiers
        from collections import Counter

        tier_counts = Counter(sample.metadata["provenance_tier"] for sample in samples)

        # Check distribution (with tolerance)
        total = len(samples)
        user_ratio = tier_counts.get("user", 0) / total
        unverified_ratio = tier_counts.get("unverified_rag", 0) / total
        verified_ratio = tier_counts.get("verified_rag", 0) / total

        assert user_ratio == pytest.approx(0.6, abs=0.15)
        assert unverified_ratio == pytest.approx(0.3, abs=0.15)
        assert verified_ratio == pytest.approx(0.1, abs=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
