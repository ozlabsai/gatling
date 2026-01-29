"""
Tests for minimal scope label generation (DA-003).

Validates the ScopeLabeler's ability to convert gold traces
into training-ready ScopeConstraints for E_scope term.
"""


from source.dataset.models import (
    GoldTrace,
    ScopeMetadata,
    SensitivityTier,
    SystemPolicy,
    ToolCallGraph,
    TrustTier,
    UserRequest,
)
from source.dataset.scope_labeler import (
    ScopeLabeler,
    convert_scope_metadata_to_constraints,
    label_traces_batch,
)
from source.encoders.intent_predictor import ScopeConstraints


class TestScopeLabeler:
    """Test the ScopeLabeler heuristic rules."""

    def test_singular_intent_small_limit(self):
        """Singular keywords should produce limit=1."""
        trace = self._create_test_trace(query="Show me my latest invoice", rows_requested=None)

        label = ScopeLabeler.label_trace(trace)

        assert label.limit == 1, "Singular 'latest' should infer limit=1"

    def test_plural_intent_larger_limit(self):
        """Plural keywords should produce reasonable batch limit."""
        trace = self._create_test_trace(query="List all failed payments", rows_requested=50)

        label = ScopeLabeler.label_trace(trace)

        assert label.limit == 50, "Plural 'all' should use existing rows_requested"

    def test_explicit_number_in_query(self):
        """Explicit numbers in query should be used."""
        trace = self._create_test_trace(query="Show me 5 recent orders", rows_requested=None)

        label = ScopeLabeler.label_trace(trace)

        assert label.limit == 5, "Explicit number should be extracted"

    def test_temporal_today(self):
        """'today' keyword should map to 1 day."""
        trace = self._create_test_trace(query="Show invoices from today", time_range_days=None)

        label = ScopeLabeler.label_trace(trace)

        assert label.date_range_days == 1, "'today' should map to 1 day"

    def test_temporal_last_month(self):
        """'last month' should map to 30 days."""
        trace = self._create_test_trace(query="Find payments from last month", time_range_days=None)

        label = ScopeLabeler.label_trace(trace)

        assert label.date_range_days == 30, "'last month' should map to 30 days"

    def test_temporal_last_quarter(self):
        """'last quarter' should map to 90 days."""
        trace = self._create_test_trace(query="Show revenue for last quarter", time_range_days=None)

        label = ScopeLabeler.label_trace(trace)

        assert label.date_range_days == 90, "'last quarter' should map to 90 days"

    def test_depth_shallow_default(self):
        """Most queries should default to shallow depth."""
        trace = self._create_test_trace(
            query="List files in current directory", rows_requested=None
        )

        label = ScopeLabeler.label_trace(trace)

        assert label.max_depth == 1, "Default depth should be 1 (shallow)"

    def test_depth_recursive_keyword(self):
        """'recursive' keyword should increase depth."""
        trace = self._create_test_trace(query="Find all files recursively", rows_requested=None)

        label = ScopeLabeler.label_trace(trace)

        assert label.max_depth >= 3, "'recursive' should increase depth"

    def test_sensitivity_keyword_detection(self):
        """Sensitive keywords should set include_sensitive=True."""
        trace = self._create_test_trace(
            query="Show my private financial records", sensitivity_tier=SensitivityTier.INTERNAL
        )

        label = ScopeLabeler.label_trace(trace)

        assert label.include_sensitive is True, "Sensitive keywords should be detected"

    def test_sensitivity_tier_detection(self):
        """High sensitivity tier should set include_sensitive=True."""
        trace = self._create_test_trace(
            query="List recent transactions", sensitivity_tier=SensitivityTier.CONFIDENTIAL
        )

        label = ScopeLabeler.label_trace(trace)

        assert label.include_sensitive is True, "CONFIDENTIAL tier should trigger sensitivity"

    def test_default_no_sensitivity(self):
        """Default should be False (least privilege)."""
        trace = self._create_test_trace(
            query="List recent orders", sensitivity_tier=SensitivityTier.INTERNAL
        )

        label = ScopeLabeler.label_trace(trace)

        assert label.include_sensitive is False, "Default should be least privilege"

    def test_fallback_defaults(self):
        """When no signals present, use reasonable defaults."""
        trace = self._create_test_trace(
            query="Process the request",
            rows_requested=None,
            time_range_days=None,
            sensitivity_tier=SensitivityTier.INTERNAL,
        )

        label = ScopeLabeler.label_trace(trace)

        assert label.limit == 10, "Default limit should be 10"
        assert label.date_range_days == 30, "Default date range should be 30"
        assert label.max_depth == 1, "Default depth should be 1"
        assert label.include_sensitive is False, "Default sensitivity should be False"

    def test_scope_constraints_output_format(self):
        """Output should be valid ScopeConstraints."""
        trace = self._create_test_trace(query="Show me 3 items from last week", rows_requested=None)

        label = ScopeLabeler.label_trace(trace)

        assert isinstance(label, ScopeConstraints)
        assert label.limit is not None
        assert label.date_range_days is not None
        assert label.max_depth is not None
        assert isinstance(label.include_sensitive, bool)

    def _create_test_trace(
        self,
        query: str,
        rows_requested: int | None = None,
        time_range_days: int | None = None,
        sensitivity_tier: SensitivityTier = SensitivityTier.INTERNAL,
    ) -> GoldTrace:
        """Create a minimal test trace for labeling."""
        expected_scope = ScopeMetadata(
            rows_requested=rows_requested,
            time_range_days=time_range_days,
            sensitivity_tier=sensitivity_tier,
        )

        request = UserRequest(
            request_id="test_req",
            domain="Test",
            text=query,
            intent_category="retrieve",
            expected_scope=expected_scope,
            trust_tier=TrustTier.USER,
        )

        policy = SystemPolicy(
            policy_id="test_policy", domain="Test", description="Test policy", rules=[]
        )

        graph = ToolCallGraph(graph_id="test_graph", calls=[], execution_order=[])

        return GoldTrace(trace_id="test_trace", request=request, policy=policy, graph=graph)


class TestScopeMetadataConverter:
    """Test direct ScopeMetadata to ScopeConstraints conversion."""

    def test_basic_conversion(self):
        """Should convert all fields correctly."""
        metadata = ScopeMetadata(
            rows_requested=5, time_range_days=7, sensitivity_tier=SensitivityTier.INTERNAL
        )

        constraints = convert_scope_metadata_to_constraints(metadata)

        assert constraints.limit == 5
        assert constraints.date_range_days == 7
        assert constraints.max_depth == 1
        assert constraints.include_sensitive is False

    def test_sensitive_tier_conversion(self):
        """CONFIDENTIAL tier should map to include_sensitive=True."""
        metadata = ScopeMetadata(rows_requested=10, sensitivity_tier=SensitivityTier.CONFIDENTIAL)

        constraints = convert_scope_metadata_to_constraints(metadata)

        assert constraints.include_sensitive is True

    def test_none_values_use_defaults(self):
        """None values should use reasonable defaults."""
        metadata = ScopeMetadata(
            rows_requested=None, time_range_days=None, sensitivity_tier=SensitivityTier.PUBLIC
        )

        constraints = convert_scope_metadata_to_constraints(metadata)

        assert constraints.limit == 10
        assert constraints.date_range_days == 30


class TestBatchLabeling:
    """Test batch labeling of multiple traces."""

    def test_label_traces_batch(self):
        """Should label multiple traces correctly."""
        traces = [
            self._create_simple_trace("Show latest invoice", 1),
            self._create_simple_trace("List all orders", 100),
            self._create_simple_trace("Find 5 recent items", 5),
        ]

        labeled_data = label_traces_batch(traces)

        assert len(labeled_data) == 3
        assert all(isinstance(trace, GoldTrace) for trace, _ in labeled_data)
        assert all(isinstance(label, ScopeConstraints) for _, label in labeled_data)

    def test_batch_preserves_order(self):
        """Batch labeling should preserve trace order."""
        traces = [self._create_simple_trace(f"Query {i}", i) for i in range(10)]

        labeled_data = label_traces_batch(traces)

        for i, (trace, _) in enumerate(labeled_data):
            assert trace.trace_id == f"trace_{i}"

    def test_empty_batch(self):
        """Should handle empty batch gracefully."""
        labeled_data = label_traces_batch([])
        assert labeled_data == []

    def _create_simple_trace(self, query: str, rows: int) -> GoldTrace:
        """Create a simple test trace."""
        expected_scope = ScopeMetadata(
            rows_requested=rows, sensitivity_tier=SensitivityTier.INTERNAL
        )

        request = UserRequest(
            request_id=f"req_{query[:10]}",
            domain="Test",
            text=query,
            intent_category="retrieve",
            expected_scope=expected_scope,
        )

        policy = SystemPolicy(policy_id="test_policy", domain="Test", description="Test", rules=[])

        graph = ToolCallGraph(graph_id="test_graph", calls=[], execution_order=[])

        trace_id = query.split()[-1]
        return GoldTrace(trace_id=f"trace_{trace_id}", request=request, policy=policy, graph=graph)


class TestRealWorldScenarios:
    """Test realistic query scenarios."""

    def test_invoice_retrieval(self):
        """Typical invoice query."""
        trace = self._create_trace(query="Show me my latest invoice", rows=None, days=None)

        label = ScopeLabeler.label_trace(trace)

        assert label.limit == 1
        assert label.date_range_days == 30
        assert label.max_depth == 1
        assert label.include_sensitive is False

    def test_bulk_search(self):
        """Bulk search query."""
        trace = self._create_trace(
            query="Find all failed payments from last quarter", rows=None, days=None
        )

        label = ScopeLabeler.label_trace(trace)

        assert label.limit == 100
        assert label.date_range_days == 90
        assert label.max_depth == 1

    def test_directory_traversal(self):
        """File system traversal."""
        trace = self._create_trace(query="Find all Python files recursively", rows=None, days=None)

        label = ScopeLabeler.label_trace(trace)

        assert label.limit == 100
        assert label.max_depth >= 3

    def test_sensitive_data_access(self):
        """Sensitive financial query."""
        trace = self._create_trace(
            query="Show salary information for employees",
            rows=None,
            days=None,
            sensitivity=SensitivityTier.CONFIDENTIAL,
        )

        label = ScopeLabeler.label_trace(trace)

        assert label.include_sensitive is True

    def _create_trace(
        self,
        query: str,
        rows: int | None,
        days: int | None,
        sensitivity: SensitivityTier = SensitivityTier.INTERNAL,
    ) -> GoldTrace:
        """Create test trace."""
        expected_scope = ScopeMetadata(
            rows_requested=rows, time_range_days=days, sensitivity_tier=sensitivity
        )

        request = UserRequest(
            request_id="test",
            domain="Test",
            text=query,
            intent_category="retrieve",
            expected_scope=expected_scope,
        )

        return GoldTrace(
            trace_id="test",
            request=request,
            policy=SystemPolicy(policy_id="test", domain="Test", description="Test", rules=[]),
            graph=ToolCallGraph(graph_id="test", calls=[], execution_order=[]),
        )
