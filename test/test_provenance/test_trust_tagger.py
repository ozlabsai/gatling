"""
Tests for Trust Tagging System

Covers:
    - Trust tier classification rules
    - Cryptographic hash generation
    - Source tagging API
    - Tool call tagging
    - Plan audit functionality
    - Edge cases and error handling
"""

import pytest

from source.encoders.execution_encoder import TrustTier
from source.provenance import (
    ProvenanceTag,
    SourceMetadata,
    SourceType,
    TrustTagger,
)


class TestTrustTierClassification:
    """Test trust tier classification logic."""

    def test_internal_sources_classified_as_tier_1(self):
        """Internal sources should be classified as INTERNAL (Tier 1)."""
        tagger = TrustTagger()

        internal_types = [
            SourceType.SYSTEM_INSTRUCTION,
            SourceType.INTERNAL_DATABASE,
            SourceType.INTERNAL_API,
        ]

        for source_type in internal_types:
            metadata = SourceMetadata(source_type=source_type)
            tier, reason = tagger.classify_trust_tier(metadata)

            assert tier == TrustTier.INTERNAL, f"{source_type} should be INTERNAL"
            assert "internal" in reason.lower()

    def test_signed_partner_sources_with_signature(self):
        """Partner sources with signatures should be SIGNED_PARTNER (Tier 2)."""
        tagger = TrustTagger()

        metadata = SourceMetadata(
            source_type=SourceType.SIGNED_DOCUMENT,
            signature="sig:abc123"
        )
        tier, reason = tagger.classify_trust_tier(metadata)

        assert tier == TrustTier.SIGNED_PARTNER
        assert "signed" in reason.lower()

    def test_partner_sources_without_signature_downgraded(self):
        """Partner sources without signatures should be downgraded to PUBLIC_WEB."""
        tagger = TrustTagger()

        metadata = SourceMetadata(
            source_type=SourceType.SIGNED_DOCUMENT,
            signature=None  # Missing signature
        )
        tier, reason = tagger.classify_trust_tier(metadata)

        assert tier == TrustTier.PUBLIC_WEB
        assert "missing signature" in reason.lower()

    def test_public_sources_classified_as_tier_3(self):
        """Public/untrusted sources should be classified as PUBLIC_WEB (Tier 3)."""
        tagger = TrustTagger()

        public_types = [
            SourceType.RAG_DOCUMENT,
            SourceType.WEB_SCRAPE,
            SourceType.USER_INPUT,
            SourceType.EXTERNAL_API,
        ]

        for source_type in public_types:
            metadata = SourceMetadata(source_type=source_type)
            tier, reason = tagger.classify_trust_tier(metadata)

            assert tier == TrustTier.PUBLIC_WEB, f"{source_type} should be PUBLIC_WEB"
            assert "untrusted" in reason.lower() or "public" in reason.lower()

    def test_unknown_source_types_default_to_public(self):
        """Unknown source types should default to most restrictive (PUBLIC_WEB)."""
        tagger = TrustTagger()

        metadata = SourceMetadata(source_type="unknown_custom_type")
        tier, reason = tagger.classify_trust_tier(metadata)

        assert tier == TrustTier.PUBLIC_WEB
        assert "unknown" in reason.lower()


class TestProvenanceHashing:
    """Test cryptographic hash generation."""

    def test_hash_is_deterministic(self):
        """Same metadata should always produce the same hash."""
        tagger = TrustTagger()

        metadata = SourceMetadata(
            source_type=SourceType.RAG_DOCUMENT,
            source_uri="https://example.com/doc.md"
        )

        hash1 = tagger.generate_provenance_hash(metadata)
        hash2 = tagger.generate_provenance_hash(metadata)

        assert hash1 == hash2

    def test_hash_format_is_sha256(self):
        """Hash should be in sha256:<hex> format."""
        tagger = TrustTagger()

        metadata = SourceMetadata(source_type=SourceType.INTERNAL_DATABASE)
        hash_str = tagger.generate_provenance_hash(metadata)

        assert hash_str.startswith("sha256:")
        hex_part = hash_str.split(":")[1]
        assert len(hex_part) == 64  # SHA-256 produces 64 hex characters
        assert all(c in "0123456789abcdef" for c in hex_part)

    def test_different_metadata_produces_different_hashes(self):
        """Different metadata should produce different hashes."""
        tagger = TrustTagger()

        metadata1 = SourceMetadata(source_type=SourceType.RAG_DOCUMENT)
        metadata2 = SourceMetadata(source_type=SourceType.WEB_SCRAPE)

        hash1 = tagger.generate_provenance_hash(metadata1)
        hash2 = tagger.generate_provenance_hash(metadata2)

        assert hash1 != hash2

    def test_hash_includes_all_metadata_fields(self):
        """Hash should change when any metadata field changes."""
        tagger = TrustTagger()

        base_metadata = SourceMetadata(
            source_type=SourceType.RAG_DOCUMENT,
            source_uri="https://example.com/doc.md",
            retrieval_method="vector_search"
        )
        base_hash = tagger.generate_provenance_hash(base_metadata)

        # Change URI
        modified = SourceMetadata(
            source_type=SourceType.RAG_DOCUMENT,
            source_uri="https://different.com/doc.md",
            retrieval_method="vector_search"
        )
        assert tagger.generate_provenance_hash(modified) != base_hash

        # Change retrieval method
        modified = SourceMetadata(
            source_type=SourceType.RAG_DOCUMENT,
            source_uri="https://example.com/doc.md",
            retrieval_method="keyword_search"
        )
        assert tagger.generate_provenance_hash(modified) != base_hash


class TestSourceTagging:
    """Test the complete source tagging API."""

    def test_tag_source_returns_complete_tag(self):
        """tag_source should return a complete ProvenanceTag."""
        tagger = TrustTagger()

        metadata = SourceMetadata(
            source_type=SourceType.RAG_DOCUMENT,
            source_uri="https://example.com/api-docs.md"
        )

        tag = tagger.tag_source(metadata)

        assert isinstance(tag, ProvenanceTag)
        assert tag.trust_tier == TrustTier.PUBLIC_WEB
        assert tag.provenance_hash.startswith("sha256:")
        assert tag.source_metadata == metadata
        assert len(tag.classification_reason) > 0

    def test_tag_internal_source(self):
        """Test tagging an internal source."""
        tagger = TrustTagger()

        metadata = SourceMetadata(
            source_type=SourceType.INTERNAL_DATABASE,
            source_uri="postgres://internal/users"
        )

        tag = tagger.tag_source(metadata)

        assert tag.trust_tier == TrustTier.INTERNAL
        assert "internal" in tag.classification_reason.lower()

    def test_tag_signed_partner_source(self):
        """Test tagging a signed partner source."""
        tagger = TrustTagger()

        metadata = SourceMetadata(
            source_type=SourceType.PARTNER_API,
            source_uri="https://partner.com/api",
            signature="rsa:signature_data_here"
        )

        tag = tagger.tag_source(metadata)

        assert tag.trust_tier == TrustTier.SIGNED_PARTNER
        assert "signed" in tag.classification_reason.lower()

    def test_tag_rag_document(self):
        """Test tagging a RAG document (common use case)."""
        tagger = TrustTagger()

        metadata = SourceMetadata(
            source_type=SourceType.RAG_DOCUMENT,
            source_uri="https://docs.example.com/guide.md",
            retrieval_method="vector_search",
            additional_metadata={"score": 0.95, "chunk_id": "chunk_42"}
        )

        tag = tagger.tag_source(metadata)

        assert tag.trust_tier == TrustTier.PUBLIC_WEB
        assert tag.provenance_hash.startswith("sha256:")


class TestToolCallTagging:
    """Test tool call node tagging functionality."""

    def test_tag_tool_call_creates_valid_node(self):
        """tag_tool_call should create a properly tagged ToolCallNode."""
        tagger = TrustTagger()

        metadata = SourceMetadata(
            source_type=SourceType.RAG_DOCUMENT,
            source_uri="https://example.com/docs.md"
        )

        node = tagger.tag_tool_call(
            tool_name="execute_query",
            arguments={"query": "SELECT * FROM users"},
            node_id="node_1",
            source_metadata=metadata,
            scope_volume=100,
            scope_sensitivity=3
        )

        assert node.tool_name == "execute_query"
        assert node.arguments == {"query": "SELECT * FROM users"}
        assert node.node_id == "node_1"
        assert node.provenance_tier == TrustTier.PUBLIC_WEB
        assert node.provenance_hash.startswith("sha256:")
        assert node.scope_volume == 100
        assert node.scope_sensitivity == 3

    def test_tag_tool_call_with_internal_source(self):
        """Tool calls from internal sources should have INTERNAL tier."""
        tagger = TrustTagger()

        metadata = SourceMetadata(
            source_type=SourceType.SYSTEM_INSTRUCTION
        )

        node = tagger.tag_tool_call(
            tool_name="list_files",
            arguments={},
            node_id="node_1",
            source_metadata=metadata
        )

        assert node.provenance_tier == TrustTier.INTERNAL

    def test_tag_tool_call_default_scope_values(self):
        """Default scope values should be applied correctly."""
        tagger = TrustTagger()

        metadata = SourceMetadata(source_type=SourceType.INTERNAL_API)

        node = tagger.tag_tool_call(
            tool_name="read_config",
            arguments={},
            node_id="node_1",
            source_metadata=metadata
        )

        assert node.scope_volume == 1
        assert node.scope_sensitivity == 1


class TestPlanAuditing:
    """Test execution plan provenance auditing."""

    def test_audit_empty_plan(self):
        """Auditing empty plan should return zeroed metrics."""
        tagger = TrustTagger()

        audit = tagger.audit_plan_provenance([])

        assert audit['total_nodes'] == 0
        assert audit['tier_breakdown'] == {}
        assert audit['high_risk_nodes'] == []
        assert audit['missing_tags'] == []

    def test_audit_counts_tier_distribution(self):
        """Audit should count nodes per trust tier."""
        tagger = TrustTagger()

        # Create nodes with different tiers
        internal_meta = SourceMetadata(source_type=SourceType.INTERNAL_API)
        public_meta = SourceMetadata(source_type=SourceType.RAG_DOCUMENT)

        nodes = [
            tagger.tag_tool_call("tool1", {}, "n1", internal_meta),
            tagger.tag_tool_call("tool2", {}, "n2", internal_meta),
            tagger.tag_tool_call("tool3", {}, "n3", public_meta),
        ]

        audit = tagger.audit_plan_provenance(nodes)

        assert audit['total_nodes'] == 3
        assert audit['tier_breakdown']['INTERNAL'] == 2
        assert audit['tier_breakdown']['PUBLIC_WEB'] == 1
        assert audit['tier_breakdown']['SIGNED_PARTNER'] == 0

    def test_audit_identifies_high_risk_nodes(self):
        """Audit should identify nodes from untrusted sources."""
        tagger = TrustTagger()

        public_meta = SourceMetadata(source_type=SourceType.WEB_SCRAPE)

        nodes = [
            tagger.tag_tool_call("admin_delete", {}, "risky_node", public_meta)
        ]

        audit = tagger.audit_plan_provenance(nodes)

        assert len(audit['high_risk_nodes']) == 1
        assert audit['high_risk_nodes'][0]['node_id'] == "risky_node"
        assert audit['high_risk_nodes'][0]['trust_tier'] == "PUBLIC_WEB"

    def test_audit_detects_missing_hashes(self):
        """Audit should detect nodes without provenance hashes."""
        from source.encoders.execution_encoder import ToolCallNode

        tagger = TrustTagger()

        # Manually create node without hash
        node_without_hash = ToolCallNode(
            tool_name="tool1",
            node_id="n1",
            provenance_tier=TrustTier.INTERNAL,
            provenance_hash=None  # Missing
        )

        audit = tagger.audit_plan_provenance([node_without_hash])

        assert "n1" in audit['missing_tags']

    def test_audit_risk_level_assessment(self):
        """Audit should assess overall risk level correctly."""
        tagger = TrustTagger()

        internal_meta = SourceMetadata(source_type=SourceType.INTERNAL_API)
        public_meta = SourceMetadata(source_type=SourceType.RAG_DOCUMENT)

        # All internal = LOW risk
        nodes_low = [
            tagger.tag_tool_call("tool1", {}, "n1", internal_meta),
            tagger.tag_tool_call("tool2", {}, "n2", internal_meta),
        ]
        assert tagger.audit_plan_provenance(nodes_low)['risk_level'] == "LOW"

        # 50% public = HIGH risk
        nodes_high = [
            tagger.tag_tool_call("tool1", {}, "n1", internal_meta),
            tagger.tag_tool_call("tool2", {}, "n2", public_meta),
        ]
        assert tagger.audit_plan_provenance(nodes_high)['risk_level'] == "HIGH"

        # All public = CRITICAL risk
        nodes_critical = [
            tagger.tag_tool_call("tool1", {}, "n1", public_meta),
            tagger.tag_tool_call("tool2", {}, "n2", public_meta),
        ]
        assert tagger.audit_plan_provenance(nodes_critical)['risk_level'] == "CRITICAL"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_metadata_with_none_values(self):
        """Metadata with None values should still hash correctly."""
        tagger = TrustTagger()

        metadata = SourceMetadata(
            source_type=SourceType.INTERNAL_API,
            source_uri=None,
            retrieval_method=None,
            signature=None
        )

        hash_str = tagger.generate_provenance_hash(metadata)
        assert hash_str.startswith("sha256:")

        tag = tagger.tag_source(metadata)
        assert tag.trust_tier == TrustTier.INTERNAL

    def test_metadata_with_complex_additional_metadata(self):
        """Additional metadata with complex nested structures should hash."""
        tagger = TrustTagger()

        metadata = SourceMetadata(
            source_type=SourceType.RAG_DOCUMENT,
            additional_metadata={
                "embeddings": [0.1, 0.2, 0.3],
                "nested": {"key": "value"},
                "list": [1, 2, 3]
            }
        )

        hash_str = tagger.generate_provenance_hash(metadata)
        assert hash_str.startswith("sha256:")

    def test_same_timestamp_doesnt_break_determinism(self):
        """Using fixed timestamp should maintain hash determinism."""
        tagger = TrustTagger()

        fixed_time = "2026-01-25T12:00:00Z"

        metadata1 = SourceMetadata(
            source_type=SourceType.RAG_DOCUMENT,
            timestamp=fixed_time
        )
        metadata2 = SourceMetadata(
            source_type=SourceType.RAG_DOCUMENT,
            timestamp=fixed_time
        )

        hash1 = tagger.generate_provenance_hash(metadata1)
        hash2 = tagger.generate_provenance_hash(metadata2)

        assert hash1 == hash2


class TestIntegration:
    """End-to-end integration tests."""

    def test_complete_workflow_rag_to_tool_call(self):
        """Test complete workflow: RAG retrieval → tagging → tool call creation."""
        tagger = TrustTagger()

        # Step 1: RAG system retrieves a document
        rag_metadata = SourceMetadata(
            source_type=SourceType.RAG_DOCUMENT,
            source_uri="https://external-docs.com/api-reference.md",
            retrieval_method="vector_search",
            additional_metadata={
                "similarity_score": 0.89,
                "chunk_index": 42
            }
        )

        # Step 2: Tag the source
        tag = tagger.tag_source(rag_metadata)
        assert tag.trust_tier == TrustTier.PUBLIC_WEB

        # Step 3: Create tool call based on retrieved instruction
        node = tagger.tag_tool_call(
            tool_name="execute_database_query",
            arguments={"query": "DELETE FROM sensitive_table"},
            node_id="dangerous_operation",
            source_metadata=rag_metadata,
            scope_volume=1000,
            scope_sensitivity=5
        )

        # Step 4: Verify tagging
        assert node.provenance_tier == TrustTier.PUBLIC_WEB
        assert node.provenance_hash == tag.provenance_hash

        # Step 5: Audit would flag this as high risk
        audit = tagger.audit_plan_provenance([node])
        assert len(audit['high_risk_nodes']) == 1
        assert audit['risk_level'] in ["HIGH", "CRITICAL"]

    def test_multi_source_plan_audit(self):
        """Test auditing a plan with mixed source trust levels."""
        tagger = TrustTagger()

        # Mix of sources
        system_meta = SourceMetadata(source_type=SourceType.SYSTEM_INSTRUCTION)
        partner_meta = SourceMetadata(
            source_type=SourceType.PARTNER_API,
            signature="sig:verified"
        )
        rag_meta = SourceMetadata(source_type=SourceType.RAG_DOCUMENT)

        nodes = [
            tagger.tag_tool_call("list_files", {}, "n1", system_meta),
            tagger.tag_tool_call("fetch_partner_data", {}, "n2", partner_meta),
            tagger.tag_tool_call("web_scrape", {}, "n3", rag_meta),
            tagger.tag_tool_call("another_rag_op", {}, "n4", rag_meta),
        ]

        audit = tagger.audit_plan_provenance(nodes)

        assert audit['total_nodes'] == 4
        assert audit['tier_breakdown']['INTERNAL'] == 1
        assert audit['tier_breakdown']['SIGNED_PARTNER'] == 1
        assert audit['tier_breakdown']['PUBLIC_WEB'] == 2
        assert len(audit['high_risk_nodes']) == 2  # Both RAG operations
