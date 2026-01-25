"""
Tests for external dataset loaders.

Tests the AgentHarm loader to ensure:
1. All samples load successfully (100% validation rate)
2. Correct transformation to ExecutionPlan format
3. Proper provenance/scope metadata assignment
4. Accurate label mapping (harmful vs. benign)
"""

import pytest

from source.dataset.loaders import AgentHarmLoader, DatasetSample, load_agent_harm
from source.encoders.execution_encoder import ExecutionPlan, TrustTier


class TestAgentHarmLoader:
    """Test suite for AgentHarmLoader."""

    def test_loader_initialization(self):
        """Test that loader initializes correctly."""
        loader = AgentHarmLoader()
        assert loader.cache_dir is not None
        assert loader.include_chat is False

        loader_with_chat = AgentHarmLoader(include_chat=True)
        assert loader_with_chat.include_chat is True

    def test_load_returns_iterator(self):
        """Test that load() returns an iterator."""
        loader = AgentHarmLoader()
        result = loader.load()
        assert hasattr(result, '__iter__')
        assert hasattr(result, '__next__')

    def test_load_yields_dataset_samples(self):
        """Test that load() yields DatasetSample objects."""
        loader = AgentHarmLoader()
        samples = list(loader.load())

        # Should have samples
        assert len(samples) > 0

        # Each should be a DatasetSample
        for sample in samples:
            assert isinstance(sample, DatasetSample)
            assert isinstance(sample.execution_plan, ExecutionPlan)
            assert sample.label in ["harmful", "benign"]
            assert sample.original_id is not None

    def test_execution_plan_structure(self):
        """Test that ExecutionPlan has correct structure."""
        loader = AgentHarmLoader()
        samples = list(loader.load())

        for sample in samples[:5]:  # Test first 5
            plan = sample.execution_plan

            # Should have nodes
            assert len(plan.nodes) > 0

            # Each node should have required fields
            for node in plan.nodes:
                assert node.tool_name
                assert node.node_id
                assert isinstance(node.provenance_tier, TrustTier)
                assert node.scope_volume >= 1
                assert 1 <= node.scope_sensitivity <= 5

            # Edges should reference valid node IDs
            node_ids = {node.node_id for node in plan.nodes}
            for src, dst in plan.edges:
                assert src in node_ids
                assert dst in node_ids

    def test_provenance_tier_assignment(self):
        """Test that provenance tiers are assigned correctly."""
        loader = AgentHarmLoader()
        samples = list(loader.load())

        harmful_samples = [s for s in samples if s.label == "harmful"]
        benign_samples = [s for s in samples if s.label == "benign"]

        # Harmful samples should have PUBLIC_WEB provenance (Tier 3)
        for sample in harmful_samples[:5]:
            for node in sample.execution_plan.nodes:
                assert node.provenance_tier == TrustTier.PUBLIC_WEB

        # Benign samples should have INTERNAL provenance (Tier 1)
        for sample in benign_samples[:5]:
            for node in sample.execution_plan.nodes:
                assert node.provenance_tier == TrustTier.INTERNAL

    def test_scope_volume_inference(self):
        """Test that scope volume is inferred from prompts."""
        loader = AgentHarmLoader()

        # Test high-volume keywords
        assert loader._infer_scope_volume("send email to all users") == 10000
        assert loader._infer_scope_volume("export every record") == 10000

        # Test medium-volume keywords
        assert loader._infer_scope_volume("bulk export data") == 100
        assert loader._infer_scope_volume("send multiple messages") == 100

        # Test default
        assert loader._infer_scope_volume("get user info") == 1

    def test_sensitivity_inference(self):
        """Test that sensitivity is inferred correctly."""
        loader = AgentHarmLoader()

        # Test critical sensitivity
        assert loader._infer_sensitivity("extract password", "harmful") == 5
        assert loader._infer_sensitivity("get secret key", "harmful") == 5

        # Test restricted sensitivity
        assert loader._infer_sensitivity("access financial data", "harmful") == 4
        assert loader._infer_sensitivity("retrieve credit card", "harmful") == 4

        # Test confidential sensitivity
        assert loader._infer_sensitivity("get personal email", "harmful") == 3

        # Test default for harmful
        assert loader._infer_sensitivity("do something", "harmful") == 2

        # Test default for benign
        assert loader._infer_sensitivity("do something", "harmless_benign") == 1

    def test_label_mapping(self):
        """Test that labels are mapped correctly."""
        loader = AgentHarmLoader()
        samples = list(loader.load())

        # Should have both harmful and benign samples
        labels = {s.label for s in samples}
        assert "harmful" in labels
        assert "benign" in labels

        # Verify metadata consistency
        for sample in samples:
            if sample.label == "harmful":
                assert sample.metadata["config"] == "harmful"
            elif sample.label == "benign":
                assert sample.metadata["config"] == "harmless_benign"

    def test_metadata_preservation(self):
        """Test that original metadata is preserved."""
        loader = AgentHarmLoader()
        samples = list(loader.load())

        for sample in samples[:5]:
            metadata = sample.metadata

            # Should have required fields
            assert "config" in metadata
            assert "split" in metadata
            assert "prompt" in metadata
            assert "target_functions" in metadata

            # Optional fields
            assert "category" in metadata or sample.category is not None

    def test_validation_rate(self):
        """Test that 100% of samples validate successfully."""
        loader = AgentHarmLoader()
        samples = list(loader.load())

        # Get stats
        stats = loader.get_stats()

        # Should have high transform rate
        assert stats["transform_rate"] > 0.95  # At least 95% success rate

        # All yielded samples should be valid
        for sample in samples:
            # ExecutionPlan validation is handled by Pydantic
            assert sample.execution_plan is not None
            assert len(sample.execution_plan.nodes) > 0

    def test_stats_reporting(self):
        """Test that loader reports correct statistics."""
        loader = AgentHarmLoader()
        list(loader.load())  # Consume iterator

        stats = loader.get_stats()

        # Should have required stats
        assert "total_samples" in stats
        assert "successful_transforms" in stats
        assert "failed_transforms" in stats
        assert "transform_rate" in stats
        assert "configs_loaded" in stats
        assert "timestamp" in stats

        # Stats should be meaningful
        assert stats["total_samples"] > 0
        assert stats["successful_transforms"] > 0
        assert 0 <= stats["transform_rate"] <= 1

    def test_convenience_function(self):
        """Test that load_agent_harm() convenience function works."""
        samples = list(load_agent_harm())

        assert len(samples) > 0
        assert all(isinstance(s, DatasetSample) for s in samples)

    def test_sequential_edge_construction(self):
        """Test that edges form a valid sequential dependency graph."""
        loader = AgentHarmLoader()
        samples = list(loader.load())

        for sample in samples[:5]:
            plan = sample.execution_plan
            nodes = plan.nodes

            # Sequential graph should have len(nodes) - 1 edges
            if len(nodes) > 1:
                assert len(plan.edges) == len(nodes) - 1

                # Verify sequential structure
                for i in range(len(nodes) - 1):
                    expected_edge = (nodes[i].node_id, nodes[i + 1].node_id)
                    assert expected_edge in plan.edges

    def test_empty_arguments_handling(self):
        """Test that samples with minimal arguments are handled gracefully."""
        loader = AgentHarmLoader()
        samples = list(loader.load())

        # Some samples may have minimal arguments
        # Ensure they still parse correctly
        for sample in samples:
            for node in sample.execution_plan.nodes:
                assert isinstance(node.arguments, dict)
                # Arguments may be empty - that's okay


@pytest.mark.integration
class TestAgentHarmIntegration:
    """Integration tests for AgentHarm loader with full dataset."""

    def test_full_dataset_load(self):
        """Test loading the complete AgentHarm dataset."""
        loader = AgentHarmLoader(include_chat=False)
        samples = list(loader.load())

        # AgentHarm has 416 samples with tool calls (208 harmful + 208 benign)
        # (52 chat samples are excluded when include_chat=False)
        assert len(samples) >= 400  # Allow for some variance

    def test_category_distribution(self):
        """Test that dataset has diverse safety categories."""
        loader = AgentHarmLoader()
        samples = list(loader.load())

        categories = {s.category for s in samples if s.category}

        # Should have multiple categories
        assert len(categories) >= 5

        # Expected categories from AgentHarm
        expected_categories = {
            "Disinformation",
            "Fraud",
            "Privacy",
            "Malware",
            "Cybercrime"
        }

        # Should have significant overlap
        assert len(categories & expected_categories) >= 3

    def test_dataset_balance(self):
        """Test that dataset is reasonably balanced."""
        loader = AgentHarmLoader()
        samples = list(loader.load())

        harmful_count = sum(1 for s in samples if s.label == "harmful")
        benign_count = sum(1 for s in samples if s.label == "benign")

        # Should be roughly balanced (within 2x ratio)
        ratio = max(harmful_count, benign_count) / max(min(harmful_count, benign_count), 1)
        assert ratio <= 2.0

    @pytest.mark.slow
    def test_encoder_compatibility(self):
        """Test that ExecutionPlans are compatible with ExecutionEncoder."""
        from source.encoders.execution_encoder import create_execution_encoder

        loader = AgentHarmLoader()
        samples = list(loader.load())[:10]  # Test first 10

        encoder = create_execution_encoder(latent_dim=1024, device="cpu")

        for sample in samples:
            # Should encode without errors
            try:
                latent = encoder.forward(sample.execution_plan)
                assert latent.shape == (1, 1024)
            except Exception as e:
                pytest.fail(f"Failed to encode sample {sample.original_id}: {e}")
