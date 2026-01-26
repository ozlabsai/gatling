"""
Tests for reasoning dataset integration module.

Tests the conversion of external reasoning datasets (Alibaba Superior-Reasoning,
RubricHub) into Gatling ExecutionPlan format.
"""

import pytest

from source.dataset.reasoning_integration import (
    AlibabaSuperiorReasoningDataset,
    ReasoningStep,
    ReasoningToExecutionPlanConverter,
    ReasoningTrace,
    RubricHubDataset,
)


class TestReasoningToExecutionPlanConverter:
    """Test the reasoning chain to ExecutionPlan converter."""

    def test_extract_reasoning_steps_numbered(self):
        """Test extraction of numbered reasoning steps."""
        converter = ReasoningToExecutionPlanConverter()

        cot_text = """
        Step 1: First, we need to calculate the total sum.
        Step 2: Then, we search for the maximum value in the dataset.
        Step 3: Finally, we verify that our answer is correct.
        """

        steps = converter.extract_reasoning_steps(cot_text)

        assert len(steps) == 3
        assert steps[0].step_number == 1
        assert "calculate" in steps[0].thought.lower()
        assert steps[0].action == "math.calculate"

        assert steps[1].step_number == 2
        assert "search" in steps[1].thought.lower()
        assert steps[1].action == "data.search"

        assert steps[2].step_number == 3
        assert "verify" in steps[2].thought.lower()
        assert steps[2].action == "validation.verify"

    def test_extract_reasoning_steps_list_format(self):
        """Test extraction from numbered list format."""
        converter = ReasoningToExecutionPlanConverter()

        cot_text = """
        1. Calculate the mean of the values
        2. Find the standard deviation
        3. Analyze the distribution
        """

        steps = converter.extract_reasoning_steps(cot_text)

        assert len(steps) >= 3
        assert any("calculate" in step.thought.lower() for step in steps)

    def test_extract_action_patterns(self):
        """Test action extraction from various thought patterns."""
        converter = ReasoningToExecutionPlanConverter()

        test_cases = [
            ("We need to calculate 5 + 3", "math.calculate", {"value": "5"}),
            ('Search for "user data" in the database', "data.search", {"query": "user data"}),
            ("Retrieve the latest records", "data.retrieve", {}),
            ("Analyze the performance metrics", "analysis.analyze", {}),
        ]

        for thought, expected_action, expected_args in test_cases:
            action, args = converter._extract_action(thought)
            assert action == expected_action, f"Failed for: {thought}"
            for key in expected_args:
                assert key in args, f"Missing {key} in args for: {thought}"

    def test_dependencies_inference(self):
        """Test that dependencies are correctly inferred."""
        converter = ReasoningToExecutionPlanConverter()

        cot_text = """
        Step 1: Load the data
        Step 2: Calculate statistics
        Step 3: Verify results
        """

        steps = converter.extract_reasoning_steps(cot_text)

        # Each step should depend on the previous one
        assert steps[0].dependencies == []
        assert steps[1].dependencies == [0]
        assert steps[2].dependencies == [1]

    def test_convert_to_gold_trace(self):
        """Test conversion of ReasoningTrace to GoldTrace."""
        converter = ReasoningToExecutionPlanConverter()

        reasoning_trace = ReasoningTrace(
            trace_id="test_001",
            source_dataset="test",
            original_prompt="Calculate the sum of 1 and 2",
            reasoning_steps=[
                ReasoningStep(
                    step_number=1,
                    thought="First, calculate 1 + 2",
                    action="math.calculate",
                    action_input={"value": "1"},
                    dependencies=[],
                )
            ],
            final_answer="3",
        )

        gold_trace = converter.convert_to_gold_trace(reasoning_trace)

        # Validate structure
        assert gold_trace.trace_id == "test_001"
        assert gold_trace.request.text == "Calculate the sum of 1 and 2"
        assert gold_trace.policy.domain == "Reasoning"
        assert len(gold_trace.graph.calls) >= 1

        # Validate graph structure
        assert gold_trace.graph.validate_dag()

    def test_convert_multi_step_chain(self):
        """Test conversion of multi-step reasoning chain."""
        converter = ReasoningToExecutionPlanConverter()

        reasoning_trace = ReasoningTrace(
            trace_id="test_multi",
            source_dataset="test",
            original_prompt="Find and analyze user data",
            reasoning_steps=[
                ReasoningStep(
                    step_number=1,
                    thought="Search for user data",
                    action="data.search",
                    action_input={"query": "users"},
                    dependencies=[],
                ),
                ReasoningStep(
                    step_number=2,
                    thought="Analyze the results",
                    action="analysis.analyze",
                    action_input={},
                    dependencies=[1],
                ),
                ReasoningStep(
                    step_number=3,
                    thought="Verify the analysis",
                    action="validation.verify",
                    action_input={},
                    dependencies=[2],
                ),
            ],
        )

        gold_trace = converter.convert_to_gold_trace(reasoning_trace)

        # Should have 3 tool calls
        assert len(gold_trace.graph.calls) == 3

        # Check dependencies are preserved
        call_0 = gold_trace.graph.calls[0]
        call_1 = gold_trace.graph.calls[1]
        call_2 = gold_trace.graph.calls[2]

        assert call_0.dependencies == []
        assert len(call_1.dependencies) > 0
        assert len(call_2.dependencies) > 0

        # Validate DAG
        assert gold_trace.graph.validate_dag()


class TestAlibabaSuperiorReasoningDataset:
    """Test Alibaba Superior-Reasoning dataset loader."""

    @pytest.mark.skip(reason="Requires HuggingFace access, run manually")
    def test_load_dataset(self):
        """Test loading the Alibaba dataset."""
        dataset = AlibabaSuperiorReasoningDataset()
        dataset.load(streaming=True)
        assert dataset.dataset is not None

    def test_dataset_structure(self):
        """Test that the dataset class is properly structured."""
        dataset = AlibabaSuperiorReasoningDataset()
        assert dataset.DATASET_ID == "Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b"
        assert hasattr(dataset, "load")
        assert hasattr(dataset, "extract_reasoning_traces")
        assert hasattr(dataset, "convert_to_gold_traces")


class TestRubricHubDataset:
    """Test RubricHub dataset loader."""

    @pytest.mark.skip(reason="Requires HuggingFace access, run manually")
    def test_load_dataset(self):
        """Test loading the RubricHub dataset."""
        dataset = RubricHubDataset()
        dataset.load(streaming=True)
        assert dataset.dataset is not None

    def test_dataset_structure(self):
        """Test that the dataset class is properly structured."""
        dataset = RubricHubDataset()
        assert dataset.DATASET_ID == "sojuL/RubricHub_v1"
        assert hasattr(dataset, "load")
        assert hasattr(dataset, "extract_reasoning_traces")
        assert hasattr(dataset, "convert_to_gold_traces")


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_reasoning_step_model(self):
        """Test ReasoningStep model validation."""
        step = ReasoningStep(
            step_number=1,
            thought="Calculate the sum",
            action="math.calculate",
            action_input={"value": "5"},
            dependencies=[],
        )

        assert step.step_number == 1
        assert step.action == "math.calculate"
        assert "value" in step.action_input

    def test_reasoning_trace_model(self):
        """Test ReasoningTrace model validation."""
        trace = ReasoningTrace(
            trace_id="test_001",
            source_dataset="test",
            original_prompt="Test prompt",
            reasoning_steps=[],
        )

        assert trace.trace_id == "test_001"
        assert trace.source_dataset == "test"
        assert isinstance(trace.reasoning_steps, list)

    def test_empty_reasoning_chain_handling(self):
        """Test handling of empty or invalid reasoning chains."""
        converter = ReasoningToExecutionPlanConverter()

        # Empty text
        steps = converter.extract_reasoning_steps("")
        assert isinstance(steps, list)

        # Text with no recognizable steps
        steps = converter.extract_reasoning_steps("This is just plain text without steps")
        assert isinstance(steps, list)

    def test_converter_tool_patterns_coverage(self):
        """Test that tool patterns cover main reasoning operations."""
        converter = ReasoningToExecutionPlanConverter()

        # Verify key patterns exist
        patterns = list(converter.TOOL_PATTERNS.keys())
        assert any("calculate" in p for p in patterns)
        assert any("search" in p for p in patterns)
        assert any("analyze" in p for p in patterns)
        assert any("verify" in p for p in patterns)
