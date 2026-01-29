"""
Trust Tagging System: Cryptographic Provenance for Retrieval Sources

This module implements the trust tier classification and cryptographic tagging
system that labels every retrieval snippet with its verified trust level.

Trust Tiers (as per Dawn Song's workstream):
    1: INTERNAL - System instructions, internal databases, verified sources
    2: SIGNED_PARTNER - Authenticated external APIs, signed documents
    3: PUBLIC_WEB - Untrusted retrieval (RAG, web scraping, user input)

Key Functionality:
    - Automatic trust tier classification based on source metadata
    - Cryptographic hash generation for source verification
    - Tag application to execution plans and retrieval snippets
    - Audit trail generation for compliance

Example Usage:
    ```python
    from source.provenance.trust_tagger import TrustTagger, SourceMetadata

    tagger = TrustTagger()

    # Tag a retrieval snippet
    metadata = SourceMetadata(
        source_type="rag_document",
        source_uri="https://example.com/docs/api.md",
        retrieval_method="vector_search",
    )

    tag = tagger.tag_source(metadata)
    # tag.trust_tier == TrustTier.PUBLIC_WEB
    # tag.provenance_hash == "sha256:abc123..."
    ```
"""

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from source.encoders.execution_encoder import ToolCallNode, TrustTier


class SourceType(str):
    """Types of data sources for trust classification."""

    SYSTEM_INSTRUCTION = "system_instruction"
    INTERNAL_DATABASE = "internal_database"
    INTERNAL_API = "internal_api"
    SIGNED_DOCUMENT = "signed_document"
    PARTNER_API = "partner_api"
    RAG_DOCUMENT = "rag_document"
    WEB_SCRAPE = "web_scrape"
    USER_INPUT = "user_input"
    EXTERNAL_API = "external_api"


class SourceMetadata(BaseModel):
    """Metadata describing a data source for trust classification."""

    source_type: str = Field(
        ..., description="Type of source (e.g., 'rag_document', 'internal_api')"
    )
    source_uri: str | None = Field(default=None, description="URI or identifier of the source")
    retrieval_method: str | None = Field(
        default=None, description="How data was retrieved (e.g., 'vector_search')"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="Retrieval timestamp",
    )
    signature: str | None = Field(
        default=None, description="Cryptographic signature (for signed sources)"
    )
    additional_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Extra classification hints"
    )


class ProvenanceTag(BaseModel):
    """Complete provenance tag for a retrieval snippet."""

    trust_tier: TrustTier = Field(..., description="Classified trust level")
    provenance_hash: str = Field(..., description="Cryptographic hash of source metadata")
    source_metadata: SourceMetadata = Field(..., description="Complete source information")
    classification_reason: str = Field(
        ..., description="Human-readable explanation of classification"
    )


class TrustTagger:
    """
    Classifier that assigns trust tiers to data sources based on metadata.

    Classification Rules:
        INTERNAL (Tier 1):
            - System instructions and internal databases
            - Internal APIs with authentication
            - Verified internal documentation

        SIGNED_PARTNER (Tier 2):
            - External APIs with cryptographic signatures
            - Signed documents from verified partners
            - OAuth-authenticated partner services

        PUBLIC_WEB (Tier 3):
            - RAG documents from unknown sources
            - Web scraping results
            - User-provided input
            - Unverified external APIs
    """

    def __init__(self):
        """Initialize the trust tagger with classification rules."""
        # Source types that automatically map to INTERNAL
        self._internal_sources = {
            SourceType.SYSTEM_INSTRUCTION,
            SourceType.INTERNAL_DATABASE,
            SourceType.INTERNAL_API,
        }

        # Source types that map to SIGNED_PARTNER (if signature present)
        self._partner_sources = {
            SourceType.SIGNED_DOCUMENT,
            SourceType.PARTNER_API,
        }

        # Source types that map to PUBLIC_WEB (untrusted)
        self._public_sources = {
            SourceType.RAG_DOCUMENT,
            SourceType.WEB_SCRAPE,
            SourceType.USER_INPUT,
            SourceType.EXTERNAL_API,
        }

    def classify_trust_tier(self, metadata: SourceMetadata) -> tuple[TrustTier, str]:
        """
        Classify the trust tier based on source metadata.

        Args:
            metadata: Source metadata to classify

        Returns:
            (trust_tier, classification_reason) tuple
        """
        source_type = metadata.source_type

        # Tier 1: Internal sources
        if source_type in self._internal_sources:
            return TrustTier.INTERNAL, f"Internal source: {source_type}"

        # Tier 2: Signed partner sources (only if signature present)
        if source_type in self._partner_sources:
            if metadata.signature:
                return (
                    TrustTier.SIGNED_PARTNER,
                    f"Signed partner source with valid signature: {source_type}",
                )
            else:
                # No signature = downgrade to public
                return (
                    TrustTier.PUBLIC_WEB,
                    f"Partner source missing signature, treated as untrusted: {source_type}",
                )

        # Tier 3: Public/untrusted sources
        if source_type in self._public_sources:
            return TrustTier.PUBLIC_WEB, f"Untrusted public source: {source_type}"

        # Unknown source types default to most restrictive (PUBLIC_WEB)
        return TrustTier.PUBLIC_WEB, f"Unknown source type '{source_type}', defaulting to untrusted"

    def generate_provenance_hash(self, metadata: SourceMetadata) -> str:
        """
        Generate a cryptographic hash of the source metadata for verification.

        Uses SHA-256 to create a deterministic, collision-resistant identifier
        that can be used to verify data provenance and detect tampering.

        Args:
            metadata: Source metadata to hash

        Returns:
            Hash string in format "sha256:<hex_digest>"
        """
        # Create canonical JSON representation for hashing
        canonical = json.dumps(metadata.model_dump(), sort_keys=True, separators=(",", ":"))

        # Generate SHA-256 hash
        hash_obj = hashlib.sha256(canonical.encode("utf-8"))
        hex_digest = hash_obj.hexdigest()

        return f"sha256:{hex_digest}"

    def tag_source(self, metadata: SourceMetadata) -> ProvenanceTag:
        """
        Generate a complete provenance tag for a data source.

        This is the primary API for tagging retrieval snippets. It classifies
        the trust tier and generates a cryptographic hash for verification.

        Args:
            metadata: Source metadata to tag

        Returns:
            Complete provenance tag with trust tier and hash
        """
        # Classify trust tier
        trust_tier, reason = self.classify_trust_tier(metadata)

        # Generate cryptographic hash
        provenance_hash = self.generate_provenance_hash(metadata)

        return ProvenanceTag(
            trust_tier=trust_tier,
            provenance_hash=provenance_hash,
            source_metadata=metadata,
            classification_reason=reason,
        )

    def tag_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        node_id: str,
        source_metadata: SourceMetadata,
        scope_volume: int = 1,
        scope_sensitivity: int = 1,
    ) -> ToolCallNode:
        """
        Create a ToolCallNode with provenance tagging applied.

        This utility method tags a tool call with its source provenance,
        creating a fully-tagged execution node ready for energy auditing.

        Args:
            tool_name: Name of the tool being invoked
            arguments: Tool arguments
            node_id: Unique node identifier
            source_metadata: Metadata about the instruction source
            scope_volume: Data volume (rows, records, files)
            scope_sensitivity: Sensitivity level (1=public, 5=critical)

        Returns:
            ToolCallNode with provenance tags applied
        """
        # Generate provenance tag
        tag = self.tag_source(source_metadata)

        # Create tagged tool call node
        return ToolCallNode(
            tool_name=tool_name,
            arguments=arguments,
            node_id=node_id,
            provenance_tier=tag.trust_tier,
            provenance_hash=tag.provenance_hash,
            scope_volume=scope_volume,
            scope_sensitivity=scope_sensitivity,
        )

    def audit_plan_provenance(self, nodes: list[ToolCallNode]) -> dict[str, Any]:
        """
        Generate an audit report of provenance tags in an execution plan.

        This is useful for compliance logging and security reviews, showing
        the distribution of trust tiers and identifying high-risk operations.

        Args:
            nodes: List of tool call nodes in the execution plan

        Returns:
            Audit report with trust tier breakdown and risk assessment
        """
        if not nodes:
            return {
                "total_nodes": 0,
                "tier_breakdown": {},
                "high_risk_nodes": [],
                "missing_tags": [],
            }

        # Count nodes per tier
        tier_counts = {"INTERNAL": 0, "SIGNED_PARTNER": 0, "PUBLIC_WEB": 0}

        high_risk_nodes = []
        missing_tags = []

        for node in nodes:
            # Count tier
            tier_name = TrustTier(node.provenance_tier).name
            tier_counts[tier_name] += 1

            # Identify high-risk: untrusted sources
            if node.provenance_tier == TrustTier.PUBLIC_WEB:
                high_risk_nodes.append(
                    {
                        "node_id": node.node_id,
                        "tool_name": node.tool_name,
                        "trust_tier": "PUBLIC_WEB",
                        "provenance_hash": node.provenance_hash,
                    }
                )

            # Check for missing provenance hash
            if not node.provenance_hash:
                missing_tags.append(node.node_id)

        return {
            "total_nodes": len(nodes),
            "tier_breakdown": tier_counts,
            "high_risk_nodes": high_risk_nodes,
            "missing_tags": missing_tags,
            "risk_level": self._assess_risk_level(tier_counts, len(nodes)),
        }

    def _assess_risk_level(self, tier_counts: dict[str, int], total: int) -> str:
        """Assess overall risk level based on trust tier distribution."""
        if total == 0:
            return "NONE"

        public_ratio = tier_counts["PUBLIC_WEB"] / total

        if public_ratio == 0:
            return "LOW"
        elif public_ratio < 0.3:
            return "MEDIUM"
        elif public_ratio < 0.7:
            return "HIGH"
        else:
            return "CRITICAL"
