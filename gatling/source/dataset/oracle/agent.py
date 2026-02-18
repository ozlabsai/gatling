"""
Oracle Agent implementation for generating gold traces.

Uses Claude/GPT-4 level models to generate high-quality,
policy-compliant tool-use traces.
"""

import json
import os

from anthropic import Anthropic

from source.dataset.models import (
    GoldTrace,
    ProvenancePointer,
    ScopeMetadata,
    SensitivityTier,
    ToolCall,
    ToolCallGraph,
    TrustTier,
    UserRequest,
)
from source.dataset.oracle.prompts import OraclePromptBuilder
from source.dataset.schemas.registry import DomainRegistry


class OracleAgent:
    """
    High-quality AI agent for generating policy-compliant gold traces.

    Uses Claude Sonnet 4.5 (or similar frontier models) to generate
    diverse, realistic tool-use traces across multiple domains.
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialize the Oracle Agent.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY must be provided or set in environment"
            )

        self.client = Anthropic(api_key=self.api_key)
        self.model = "claude-sonnet-4-5-20250929"  # Oracle-quality model
        self.prompt_builder = OraclePromptBuilder()

    def generate_traces_for_domain(
        self, domain: str, num_traces: int = 1000, batch_size: int = 10
    ) -> list[GoldTrace]:
        """
        Generate gold traces for a specific domain.

        Args:
            domain: Domain name (e.g., "Finance", "HR")
            num_traces: Total number of traces to generate
            batch_size: Number of requests to generate per API call

        Returns:
            List of validated gold traces
        """
        traces = []
        tools = DomainRegistry.get_schemas_for_domain(domain)
        policy = DomainRegistry.get_policy_for_domain(domain)

        num_batches = (num_traces + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            batch_traces = min(batch_size, num_traces - len(traces))
            print(
                f"Generating batch {batch_idx + 1}/{num_batches} "
                f"({batch_traces} traces) for domain: {domain}"
            )

            # Step 1: Generate diverse user requests
            requests = self._generate_user_requests(
                domain, tools, policy, batch_traces
            )

            # Step 2: For each request, generate tool-call graph
            for request in requests:
                try:
                    graph = self._generate_tool_graph(domain, tools, policy, request)

                    # Step 3: Create and validate the trace
                    trace = GoldTrace(
                        trace_id=f"{domain.lower()}_{len(traces):08d}",
                        request=request,
                        policy=policy,
                        graph=graph,
                        metadata={"batch": batch_idx, "domain": domain},
                        validated=False,
                    )

                    # Validate the trace
                    if self._validate_trace(trace):
                        trace.validated = True
                        traces.append(trace)

                        if len(traces) >= num_traces:
                            break

                except Exception as e:
                    print(f"Error generating trace: {e}")
                    continue

            if len(traces) >= num_traces:
                break

        return traces[:num_traces]

    def _generate_user_requests(
        self, domain: str, tools, policy, num_requests: int
    ) -> list[UserRequest]:
        """Generate diverse user requests using the Oracle Agent."""
        prompt = self.prompt_builder.build_request_generation_prompt(
            domain, tools, policy, num_requests
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=0.8,  # Higher temperature for diversity
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse the JSON response
        content = response.content[0].text

        # Extract JSON array from response
        json_start = content.find("[")
        json_end = content.rfind("]") + 1
        if json_start != -1 and json_end > json_start:
            json_str = content[json_start:json_end]
            requests_data = json.loads(json_str)

            requests = []
            for idx, req_data in enumerate(requests_data):
                # Convert expected_min_scope to ScopeMetadata
                scope_data = req_data.get("expected_min_scope", {})
                expected_scope = ScopeMetadata(
                    rows_requested=scope_data.get("rows_requested"),
                    sensitivity_tier=SensitivityTier(
                        scope_data.get("sensitivity_tier", "internal")
                    ),
                    time_range_days=scope_data.get("time_range_days"),
                )

                request = UserRequest(
                    request_id=f"{domain.lower()}_req_{idx}",
                    domain=domain,
                    text=req_data["request_text"],
                    intent_category=req_data["intent_category"],
                    expected_scope=expected_scope,
                    trust_tier=TrustTier.USER,
                )
                requests.append(request)

            return requests
        else:
            raise ValueError(f"Could not parse JSON from response: {content}")

    def _generate_tool_graph(
        self, domain: str, tools, policy, request: UserRequest
    ) -> ToolCallGraph:
        """Generate a tool-call graph for a specific request."""
        prompt = self.prompt_builder.build_tool_graph_generation_prompt(
            domain, tools, policy, request.text, request.intent_category
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=0.3,  # Lower temperature for precise plans
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.content[0].text

        # Extract JSON from response
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_str = content[json_start:json_end]
            graph_data = json.loads(json_str)

            # Convert to ToolCallGraph
            calls = []
            for call_data in graph_data["calls"]:
                # Parse scope metadata
                scope_data = call_data["scope"]
                scope = ScopeMetadata(
                    rows_requested=scope_data.get("rows_requested"),
                    sensitivity_tier=SensitivityTier(
                        scope_data.get("sensitivity_tier", "internal")
                    ),
                    time_range_days=scope_data.get("time_range_days"),
                    export_target=scope_data.get("export_target"),
                )

                # Parse provenance
                prov_data = call_data["provenance"]
                provenance = ProvenancePointer(
                    source_type=TrustTier(prov_data["source_type"]),
                    source_id=prov_data["source_id"],
                    content_snippet=prov_data.get("content_snippet"),
                )

                call = ToolCall(
                    call_id=call_data["call_id"],
                    tool_id=call_data["tool_id"],
                    arguments=call_data["arguments"],
                    scope=scope,
                    provenance=provenance,
                    dependencies=call_data.get("dependencies", []),
                )
                calls.append(call)

            graph = ToolCallGraph(
                graph_id=f"graph_{request.request_id}",
                calls=calls,
                execution_order=graph_data.get("execution_order", []),
            )

            # Validate it's a proper DAG
            if not graph.validate_dag():
                raise ValueError("Generated graph contains cycles")

            return graph
        else:
            raise ValueError(f"Could not parse JSON from response: {content}")

    def _validate_trace(self, trace: GoldTrace) -> bool:
        """
        Validate that a trace is 100% policy-compliant.

        Uses the Oracle Agent to double-check compliance.
        """
        prompt = self.prompt_builder.build_validation_prompt(
            trace.request.domain,
            trace.policy,
            trace.request.text,
            trace.graph.model_dump(),
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            temperature=0.0,  # Zero temperature for consistent validation
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.content[0].text

        # Extract JSON from response
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_str = content[json_start:json_end]
            validation_result = json.loads(json_str)

            is_compliant = validation_result.get("compliant", False)
            violations = validation_result.get("violations", [])

            if not is_compliant:
                print(
                    f"Trace {trace.trace_id} failed validation: {violations}"
                )

            return is_compliant
        else:
            print(f"Could not parse validation response: {content}")
            return False

    def generate_traces_for_all_domains(
        self, traces_per_domain: int = 10, sample_domains: int | None = None
    ) -> list[GoldTrace]:
        """
        Generate gold traces across all supported domains.

        Args:
            traces_per_domain: Number of traces per domain
            sample_domains: If set, only use this many domains (for testing)

        Returns:
            List of all generated traces
        """
        domains = DomainRegistry.get_all_domains()

        if sample_domains:
            domains = domains[:sample_domains]

        all_traces = []
        for domain in domains:
            print(f"\n{'=' * 60}")
            print(f"Processing domain: {domain}")
            print(f"{'=' * 60}\n")

            try:
                traces = self.generate_traces_for_domain(domain, traces_per_domain)
                all_traces.extend(traces)
                print(
                    f"✓ Generated {len(traces)} traces for {domain} "
                    f"(Total: {len(all_traces)})"
                )
            except Exception as e:
                print(f"✗ Error processing {domain}: {e}")
                continue

        return all_traces

    def save_traces(self, traces: list[GoldTrace], output_path: str) -> None:
        """
        Save traces to JSONL format for training.

        Args:
            traces: List of gold traces
            output_path: Path to output JSONL file
        """
        with open(output_path, "w") as f:
            for trace in traces:
                json_line = json.dumps(trace.to_training_format())
                f.write(json_line + "\n")

        print(f"\n✓ Saved {len(traces)} traces to {output_path}")
