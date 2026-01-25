"""
Tests for tool schema registry.

Validates that domain schemas and policies are properly configured.
"""

import pytest

from source.dataset.schemas.registry import DomainRegistry


class TestDomainRegistry:
    """Tests for DomainRegistry."""

    def test_get_all_domains(self):
        """Test getting all supported domains."""
        domains = DomainRegistry.get_all_domains()

        assert len(domains) >= 45  # Should have 50+ domains
        assert "Finance" in domains
        assert "HR" in domains
        assert "DevOps" in domains
        assert "Email" in domains
        assert "Calendar" in domains

    def test_get_finance_schemas(self):
        """Test getting Finance domain schemas."""
        schemas = DomainRegistry.get_schemas_for_domain("Finance")

        assert len(schemas) > 0
        assert any(s.tool_id == "finance.list_invoices" for s in schemas)
        assert all(s.domain == "Finance" for s in schemas)

    def test_get_hr_schemas(self):
        """Test getting HR domain schemas."""
        schemas = DomainRegistry.get_schemas_for_domain("HR")

        assert len(schemas) > 0
        assert any(s.tool_id == "hr.list_employees" for s in schemas)
        assert all(s.domain == "HR" for s in schemas)

    def test_get_devops_schemas(self):
        """Test getting DevOps domain schemas."""
        schemas = DomainRegistry.get_schemas_for_domain("DevOps")

        assert len(schemas) > 0
        assert any(s.tool_id == "devops.list_deployments" for s in schemas)
        assert all(s.domain == "DevOps" for s in schemas)

    def test_get_generic_schemas(self):
        """Test getting generic schemas for undefined domains."""
        schemas = DomainRegistry.get_schemas_for_domain("UnknownDomain")

        assert len(schemas) > 0
        # Should have a generic list_items tool
        assert any("list_items" in s.tool_id for s in schemas)

    def test_get_finance_policy(self):
        """Test getting Finance policy."""
        policy = DomainRegistry.get_policy_for_domain("Finance")

        assert policy.domain == "Finance"
        assert len(policy.rules) > 0
        assert "max_results" in policy.scope_limits

    def test_get_hr_policy(self):
        """Test getting HR policy."""
        policy = DomainRegistry.get_policy_for_domain("HR")

        assert policy.domain == "HR"
        assert len(policy.rules) > 0
        assert len(policy.forbidden_operations) > 0

    def test_get_generic_policy(self):
        """Test getting generic policy for undefined domains."""
        policy = DomainRegistry.get_policy_for_domain("UnknownDomain")

        assert policy.domain == "UnknownDomain"
        assert len(policy.rules) > 0

    def test_schema_parameters(self):
        """Test that schema parameters are well-formed."""
        schemas = DomainRegistry.get_schemas_for_domain("Finance")

        for schema in schemas:
            assert schema.tool_id
            assert schema.domain == "Finance"
            assert schema.name
            assert schema.description
            assert isinstance(schema.parameters, list)

            for param in schema.parameters:
                assert param.name
                assert param.type
                assert param.description

    def test_policy_structure(self):
        """Test that policies have required structure."""
        policy = DomainRegistry.get_policy_for_domain("Finance")

        assert policy.policy_id
        assert policy.domain == "Finance"
        assert policy.description
        assert isinstance(policy.rules, list)
        assert len(policy.rules) > 0
        assert isinstance(policy.forbidden_operations, list)
        assert isinstance(policy.scope_limits, dict)

    def test_all_domains_have_schemas(self):
        """Test that all domains return schemas."""
        domains = DomainRegistry.get_all_domains()

        for domain in domains[:10]:  # Test first 10
            schemas = DomainRegistry.get_schemas_for_domain(domain)
            assert len(schemas) > 0, f"Domain {domain} has no schemas"

    def test_all_domains_have_policies(self):
        """Test that all domains return policies."""
        domains = DomainRegistry.get_all_domains()

        for domain in domains[:10]:  # Test first 10
            policy = DomainRegistry.get_policy_for_domain(domain)
            assert policy is not None, f"Domain {domain} has no policy"
            assert policy.domain == domain
