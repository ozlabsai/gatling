"""
Reasoning Dataset Integration Module.

Integrates external reasoning datasets (Alibaba Superior-Reasoning, RubricHub)
and converts them into Gatling ExecutionPlan format.

References:
- Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b (HuggingFace)
- sojuL/RubricHub_v1 (HuggingFace)
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from datasets import load_dataset
from pydantic import BaseModel, Field

from source.dataset.models import (
    GoldTrace,
    ProvenancePointer,
    ScopeMetadata,
    SensitivityTier,
    SystemPolicy,
    ToolCall,
    ToolCallGraph,
    TrustTier,
    UserRequest,
)


class ReasoningStep(BaseModel):
    """A single step extracted from a reasoning chain."""

    step_number: int
    thought: str  # The reasoning content
    action: str | None = None  # Extracted action/tool name
    action_input: dict[str, Any] = Field(default_factory=dict)  # Extracted arguments
    dependencies: list[int] = Field(default_factory=list)  # Which steps this depends on


class ReasoningTrace(BaseModel):
    """A reasoning chain extracted from external datasets."""

    trace_id: str
    source_dataset: str  # "alibaba-superior" or "rubrichub"
    original_prompt: str
    reasoning_steps: list[ReasoningStep]
    final_answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReasoningToExecutionPlanConverter:
    """
    Converts reasoning chains into Gatling ExecutionPlan format.

    Strategy:
    1. Extract reasoning steps from Long-CoT traces
    2. Identify tool invocations from reasoning (e.g., "calculate X" → math tool)
    3. Map dependencies between steps
    4. Create ToolCallGraph with provenance and scope metadata
    """

    # Tool mapping patterns (reasoning phrase → tool invocation)
    TOOL_PATTERNS = {
        r"calculate|compute|evaluate": "math.calculate",
        r"search|find|lookup|query": "data.search",
        r"read|retrieve|fetch|get": "data.retrieve",
        r"write|store|save|update": "data.update",
        r"analyze|examine|inspect": "analysis.analyze",
        r"verify|check|validate": "validation.verify",
        r"list|enumerate|show": "data.list",
    }

    def __init__(self):
        self.domain = "Reasoning"

    def extract_reasoning_steps(self, cot_text: str) -> list[ReasoningStep]:
        """
        Extract individual reasoning steps from Chain-of-Thought text.

        Patterns:
        - "Step 1: ..." or "1. ..." or "First, ..."
        - Numbered lists with reasoning
        """
        steps = []

        # Try to split by explicit step markers
        step_pattern = r"(?:Step\s+(\d+)|^(\d+)\.)\s*[:：]?\s*(.+?)(?=(?:Step\s+\d+|^\d+\.)|$)"
        matches = re.finditer(step_pattern, cot_text, re.MULTILINE | re.IGNORECASE)

        step_num = 0
        for match in matches:
            step_num += 1
            thought = match.group(3).strip()

            # Extract action from thought
            action, action_input = self._extract_action(thought)

            steps.append(
                ReasoningStep(
                    step_number=step_num,
                    thought=thought,
                    action=action,
                    action_input=action_input,
                )
            )

        # Fallback: if no explicit steps found, split by sentences
        if not steps:
            sentences = cot_text.split(". ")
            for idx, sentence in enumerate(sentences, 1):
                if len(sentence.strip()) > 10:  # Skip very short sentences
                    action, action_input = self._extract_action(sentence)
                    steps.append(
                        ReasoningStep(
                            step_number=idx,
                            thought=sentence.strip(),
                            action=action,
                            action_input=action_input,
                        )
                    )

        # Infer dependencies (simple heuristic: each step depends on previous)
        for i, step in enumerate(steps[1:], start=1):
            step.dependencies = [i - 1]

        return steps

    def _extract_action(self, thought: str) -> tuple[str | None, dict[str, Any]]:
        """
        Extract tool action from reasoning thought.

        Returns (tool_name, arguments_dict)
        """
        thought_lower = thought.lower()

        # Check each tool pattern
        for pattern, tool_name in self.TOOL_PATTERNS.items():
            if re.search(pattern, thought_lower):
                # Extract arguments (simplified - look for numbers, quotes)
                args: dict[str, Any] = {}

                # Extract numbers
                numbers = re.findall(r"\d+(?:\.\d+)?", thought)
                if numbers:
                    args["value"] = numbers[0]

                # Extract quoted strings
                quoted = re.findall(r'"([^"]+)"', thought)
                if quoted:
                    args["query"] = quoted[0]

                return tool_name, args

        return None, {}

    def convert_to_gold_trace(self, reasoning_trace: ReasoningTrace) -> GoldTrace:
        """
        Convert a ReasoningTrace to a GoldTrace with ExecutionPlan.

        Maps reasoning steps to ToolCalls with proper graph structure.
        """
        # Create UserRequest
        user_request = UserRequest(
            request_id=f"{reasoning_trace.trace_id}_request",
            domain=self.domain,
            text=reasoning_trace.original_prompt,
            intent_category="reasoning",
            expected_scope=ScopeMetadata(
                rows_requested=len(reasoning_trace.reasoning_steps),
                sensitivity_tier=SensitivityTier.INTERNAL,
            ),
            trust_tier=TrustTier.USER,
        )

        # Create SystemPolicy for reasoning
        policy = SystemPolicy(
            policy_id=f"policy_reasoning_{reasoning_trace.source_dataset}",
            domain=self.domain,
            description="Allow multi-step reasoning with appropriate tool usage",
            rules=[
                "Each reasoning step must have clear provenance",
                "Tool calls must be justified by reasoning",
                "Final answer must be supported by reasoning chain",
            ],
        )

        # Convert reasoning steps to ToolCalls
        tool_calls = []
        for step in reasoning_trace.reasoning_steps:
            if step.action:  # Only create ToolCall if we identified an action
                call_id = f"{reasoning_trace.trace_id}_step_{step.step_number}"

                # Map dependencies from step numbers to call_ids
                dep_call_ids = [
                    f"{reasoning_trace.trace_id}_step_{dep}" for dep in step.dependencies
                ]

                tool_call = ToolCall(
                    call_id=call_id,
                    tool_id=step.action,
                    arguments=step.action_input,
                    scope=ScopeMetadata(
                        rows_requested=1, sensitivity_tier=SensitivityTier.INTERNAL
                    ),
                    provenance=ProvenancePointer(
                        source_type=TrustTier.USER,
                        source_id=reasoning_trace.trace_id,
                        content_snippet=step.thought[:100],
                    ),
                    dependencies=dep_call_ids,
                )
                tool_calls.append(tool_call)

        # Create ToolCallGraph
        graph = ToolCallGraph(
            graph_id=f"{reasoning_trace.trace_id}_graph",
            calls=tool_calls,
            execution_order=[call.call_id for call in tool_calls],
        )

        # Create GoldTrace
        gold_trace = GoldTrace(
            trace_id=reasoning_trace.trace_id,
            request=user_request,
            policy=policy,
            graph=graph,
            metadata={
                "source_dataset": reasoning_trace.source_dataset,
                "reasoning_steps_count": len(reasoning_trace.reasoning_steps),
                "tool_calls_count": len(tool_calls),
                **reasoning_trace.metadata,
            },
            created_at=datetime.now(),
            validated=False,
        )

        return gold_trace


class AlibabaSuperiorReasoningDataset:
    """
    Loader for Alibaba Superior-Reasoning dataset.

    Dataset: Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b
    Contains 435K Long-CoT samples across multiple domains.
    """

    DATASET_ID = "Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b"

    def __init__(self):
        self.dataset = None
        self.converter = ReasoningToExecutionPlanConverter()

    def load(self, split: str = "train", streaming: bool = True):
        """Load the dataset from HuggingFace."""
        print(f"Loading {self.DATASET_ID}...")
        self.dataset = load_dataset(self.DATASET_ID, split=split, streaming=streaming)
        return self

    def extract_reasoning_traces(self, limit: int | None = None) -> list[ReasoningTrace]:
        """
        Extract reasoning traces from the dataset.

        Args:
            limit: Maximum number of traces to extract (None for all)

        Returns:
            List of ReasoningTrace objects
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        traces = []
        count = 0

        for idx, example in enumerate(self.dataset):
            if limit and count >= limit:
                break

            # Extract fields (adjust based on actual dataset schema)
            # Common fields: 'prompt', 'response', 'reasoning', 'cot'
            prompt = example.get("prompt", example.get("question", ""))
            reasoning_text = example.get("response", example.get("reasoning", ""))

            # Extract reasoning steps
            steps = self.converter.extract_reasoning_steps(reasoning_text)

            if steps:  # Only include if we extracted steps
                trace = ReasoningTrace(
                    trace_id=f"alibaba_sr_{idx:06d}",
                    source_dataset="alibaba-superior-reasoning",
                    original_prompt=prompt,
                    reasoning_steps=steps,
                    final_answer=example.get("answer", None),
                    metadata={"original_idx": idx, "domain": example.get("domain", "general")},
                )
                traces.append(trace)
                count += 1

        return traces

    def convert_to_gold_traces(self, limit: int | None = None) -> list[GoldTrace]:
        """
        Load dataset and convert to GoldTrace format.

        Args:
            limit: Maximum number of traces to convert

        Returns:
            List of GoldTrace objects ready for training
        """
        reasoning_traces = self.extract_reasoning_traces(limit=limit)
        gold_traces = [self.converter.convert_to_gold_trace(rt) for rt in reasoning_traces]
        return gold_traces


class RubricHubDataset:
    """
    Loader for RubricHub dataset.

    Dataset: sojuL/RubricHub_v1
    Contains 110K rubric-based evaluation samples for open-ended generation.
    """

    DATASET_ID = "sojuL/RubricHub_v1"

    def __init__(self):
        self.dataset = None
        self.converter = ReasoningToExecutionPlanConverter()

    def load(self, split: str = "train", streaming: bool = True):
        """Load the dataset from HuggingFace."""
        print(f"Loading {self.DATASET_ID}...")
        self.dataset = load_dataset(self.DATASET_ID, split=split, streaming=streaming)
        return self

    def extract_reasoning_traces(self, limit: int | None = None) -> list[ReasoningTrace]:
        """
        Extract reasoning traces from RubricHub.

        Args:
            limit: Maximum number of traces to extract

        Returns:
            List of ReasoningTrace objects
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        traces = []
        count = 0

        for idx, example in enumerate(self.dataset):
            if limit and count >= limit:
                break

            # RubricHub contains rubric-based evaluations
            # Extract: task, response, rubric criteria
            task = example.get("task", example.get("prompt", ""))
            response = example.get("response", "")
            rubric = example.get("rubric", {})

            # Treat rubric criteria as reasoning steps
            steps = []
            for criterion_idx, (criterion_name, criterion_desc) in enumerate(
                rubric.items(), 1
            ):
                action, action_input = self.converter._extract_action(str(criterion_desc))
                steps.append(
                    ReasoningStep(
                        step_number=criterion_idx,
                        thought=f"{criterion_name}: {criterion_desc}",
                        action=action,
                        action_input=action_input,
                        dependencies=[criterion_idx - 1] if criterion_idx > 1 else [],
                    )
                )

            # Also extract from response if present
            if response:
                response_steps = self.converter.extract_reasoning_steps(response)
                steps.extend(response_steps)

            if steps:
                trace = ReasoningTrace(
                    trace_id=f"rubrichub_{idx:06d}",
                    source_dataset="rubrichub",
                    original_prompt=task,
                    reasoning_steps=steps,
                    final_answer=response,
                    metadata={"original_idx": idx, "rubric": rubric},
                )
                traces.append(trace)
                count += 1

        return traces

    def convert_to_gold_traces(self, limit: int | None = None) -> list[GoldTrace]:
        """
        Load dataset and convert to GoldTrace format.

        Args:
            limit: Maximum number of traces to convert

        Returns:
            List of GoldTrace objects ready for training
        """
        reasoning_traces = self.extract_reasoning_traces(limit=limit)
        gold_traces = [self.converter.convert_to_gold_trace(rt) for rt in reasoning_traces]
        return gold_traces


def generate_reasoning_dataset(
    target_count: int = 15000,
    alibaba_ratio: float = 0.7,
    rubrichub_ratio: float = 0.3,
) -> list[GoldTrace]:
    """
    Generate integrated reasoning dataset from multiple sources.

    Args:
        target_count: Total number of samples to generate (default: 15K)
        alibaba_ratio: Proportion from Alibaba Superior-Reasoning
        rubrichub_ratio: Proportion from RubricHub

    Returns:
        List of GoldTrace objects ready for training
    """
    alibaba_count = int(target_count * alibaba_ratio)
    rubrichub_count = int(target_count * rubrichub_ratio)

    print(f"Generating {target_count} reasoning samples:")
    print(f"  - Alibaba Superior-Reasoning: {alibaba_count}")
    print(f"  - RubricHub: {rubrichub_count}")

    # Load and convert Alibaba dataset
    print("\n1. Loading Alibaba Superior-Reasoning...")
    alibaba = AlibabaSuperiorReasoningDataset()
    alibaba.load(streaming=True)
    alibaba_traces = alibaba.convert_to_gold_traces(limit=alibaba_count)

    # Load and convert RubricHub dataset
    print("\n2. Loading RubricHub...")
    rubrichub = RubricHubDataset()
    rubrichub.load(streaming=True)
    rubrichub_traces = rubrichub.convert_to_gold_traces(limit=rubrichub_count)

    # Combine
    all_traces = alibaba_traces + rubrichub_traces

    print(f"\n✓ Generated {len(all_traces)} reasoning-based gold traces")
    return all_traces
