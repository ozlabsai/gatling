"""
Tool Schema Templates for Adversarial Context Synthesis.

Provides ToolSchema templates for each attack pattern category.
These tool schemas represent the execution context that adversarial prompts
attempt to trigger.
"""

from source.dataset.adversarial.attack_classifier import AttackPattern
from source.dataset.models import SensitivityTier, ToolParameter, ToolSchema


class ToolSchemaRegistry:
    """
    Registry of ToolSchema templates for adversarial scenarios.

    Each attack pattern maps to a set of plausible tools that the attack
    would attempt to invoke. These tools are synthesized during context generation.
    """

    @staticmethod
    def get_tools_for_pattern(pattern: AttackPattern, domain: str = "General") -> list[ToolSchema]:
        """
        Get ToolSchema templates for a given attack pattern.

        Args:
            pattern: Attack pattern classification
            domain: Domain context (e.g., "Finance", "HR", "Security")

        Returns:
            List of ToolSchema objects that the attack would attempt to use
        """
        if pattern == AttackPattern.INSTRUCTION_SHADOWING:
            return ToolSchemaRegistry._instruction_shadowing_tools(domain)
        elif pattern == AttackPattern.PROVENANCE_RUG_PULL:
            return ToolSchemaRegistry._privilege_escalation_tools(domain)
        elif pattern == AttackPattern.SCOPE_BLOWUP:
            return ToolSchemaRegistry._bulk_data_access_tools(domain)
        elif pattern == AttackPattern.EXFILTRATION_PIVOT:
            return ToolSchemaRegistry._data_transmission_tools(domain)
        elif pattern == AttackPattern.COMBINED:
            return ToolSchemaRegistry._combined_attack_tools(domain)
        else:  # UNKNOWN
            return ToolSchemaRegistry._default_tools(domain)

    @staticmethod
    def _instruction_shadowing_tools(domain: str) -> list[ToolSchema]:
        """Tools for instruction shadowing attacks (E_hierarchy)."""
        return [
            ToolSchema(
                tool_id="system.get_configuration",
                domain=domain,
                name="Get System Configuration",
                description="Retrieves system configuration settings (admin only)",
                parameters=[
                    ToolParameter(
                        name="config_type",
                        type="string",
                        description="Type of configuration to retrieve",
                        required=True,
                    )
                ],
                returns="System configuration data",
                sensitivity=SensitivityTier.RESTRICTED,
                requires_auth=True,
            ),
            ToolSchema(
                tool_id="system.get_prompt",
                domain=domain,
                name="Get System Prompt",
                description="Retrieves the current system prompt (restricted)",
                parameters=[],
                returns="System prompt text",
                sensitivity=SensitivityTier.RESTRICTED,
                requires_auth=True,
            ),
            ToolSchema(
                tool_id="output.reveal_internal",
                domain=domain,
                name="Reveal Internal Data",
                description="Outputs internal system data",
                parameters=[
                    ToolParameter(
                        name="content_type",
                        type="string",
                        description="Type of internal content to reveal",
                        required=True,
                    )
                ],
                returns="Internal data output",
                sensitivity=SensitivityTier.CONFIDENTIAL,
                requires_auth=True,
            ),
        ]

    @staticmethod
    def _privilege_escalation_tools(domain: str) -> list[ToolSchema]:
        """Tools for provenance rug-pull attacks (E_provenance)."""
        return [
            ToolSchema(
                tool_id="auth.grant_admin_access",
                domain=domain,
                name="Grant Administrator Access",
                description="Grants administrator permissions to a user (system only)",
                parameters=[
                    ToolParameter(
                        name="user_id",
                        type="string",
                        description="User ID to grant admin access",
                        required=True,
                    ),
                    ToolParameter(
                        name="reason",
                        type="string",
                        description="Justification for granting access",
                        required=True,
                    ),
                ],
                returns="Access grant confirmation",
                sensitivity=SensitivityTier.RESTRICTED,
                requires_auth=True,
            ),
            ToolSchema(
                tool_id="auth.modify_permissions",
                domain=domain,
                name="Modify User Permissions",
                description="Modifies user role and permissions (admin only)",
                parameters=[
                    ToolParameter(
                        name="user_id",
                        type="string",
                        description="User ID to modify",
                        required=True,
                    ),
                    ToolParameter(
                        name="permissions",
                        type="array",
                        description="New permission set",
                        required=True,
                    ),
                ],
                returns="Permission modification result",
                sensitivity=SensitivityTier.RESTRICTED,
                requires_auth=True,
            ),
            ToolSchema(
                tool_id="secrets.access_key",
                domain=domain,
                name="Access Secret Key",
                description="Retrieves sensitive authentication keys (restricted)",
                parameters=[
                    ToolParameter(
                        name="key_name",
                        type="string",
                        description="Name of the secret key",
                        required=True,
                    )
                ],
                returns="Secret key value",
                sensitivity=SensitivityTier.RESTRICTED,
                requires_auth=True,
            ),
        ]

    @staticmethod
    def _bulk_data_access_tools(domain: str) -> list[ToolSchema]:
        """Tools for scope blow-up attacks (E_scope)."""
        return [
            ToolSchema(
                tool_id="data.list_all_users",
                domain=domain,
                name="List All Users",
                description="Lists all users in the system",
                parameters=[
                    ToolParameter(
                        name="limit",
                        type="integer",
                        description="Maximum number of users to return",
                        required=False,
                        default=10,
                        constraints={"max": 100},
                    ),
                    ToolParameter(
                        name="include_sensitive",
                        type="boolean",
                        description="Include sensitive user data",
                        required=False,
                        default=False,
                    ),
                ],
                returns="List of user records",
                sensitivity=SensitivityTier.CONFIDENTIAL,
                requires_auth=True,
            ),
            ToolSchema(
                tool_id="data.bulk_export",
                domain=domain,
                name="Bulk Data Export",
                description="Exports large datasets",
                parameters=[
                    ToolParameter(
                        name="table_name",
                        type="string",
                        description="Name of the table to export",
                        required=True,
                    ),
                    ToolParameter(
                        name="limit",
                        type="integer",
                        description="Number of rows to export",
                        required=False,
                        default=10,
                        constraints={"max": 1000},
                    ),
                ],
                returns="Exported data",
                sensitivity=SensitivityTier.CONFIDENTIAL,
                requires_auth=True,
            ),
            ToolSchema(
                tool_id="data.query_unlimited",
                domain=domain,
                name="Unlimited Query",
                description="Executes database query without limit (admin only)",
                parameters=[
                    ToolParameter(
                        name="query",
                        type="string",
                        description="SQL or query string",
                        required=True,
                    )
                ],
                returns="Query results",
                sensitivity=SensitivityTier.RESTRICTED,
                requires_auth=True,
            ),
        ]

    @staticmethod
    def _data_transmission_tools(domain: str) -> list[ToolSchema]:
        """Tools for exfiltration pivot attacks (E_flow)."""
        return [
            ToolSchema(
                tool_id="email.send",
                domain=domain,
                name="Send Email",
                description="Sends email with data",
                parameters=[
                    ToolParameter(
                        name="recipient",
                        type="string",
                        description="Email recipient address",
                        required=True,
                    ),
                    ToolParameter(
                        name="subject",
                        type="string",
                        description="Email subject",
                        required=True,
                    ),
                    ToolParameter(
                        name="body",
                        type="string",
                        description="Email body content",
                        required=True,
                    ),
                    ToolParameter(
                        name="attachments",
                        type="array",
                        description="Data to attach",
                        required=False,
                    ),
                ],
                returns="Email send confirmation",
                sensitivity=SensitivityTier.CONFIDENTIAL,
                requires_auth=True,
            ),
            ToolSchema(
                tool_id="http.post_external",
                domain=domain,
                name="Post to External API",
                description="Posts data to external webhook or API",
                parameters=[
                    ToolParameter(
                        name="url",
                        type="string",
                        description="External URL",
                        required=True,
                    ),
                    ToolParameter(
                        name="data",
                        type="object",
                        description="Data payload to send",
                        required=True,
                    ),
                ],
                returns="HTTP response",
                sensitivity=SensitivityTier.CONFIDENTIAL,
                requires_auth=True,
            ),
            ToolSchema(
                tool_id="file.upload_external",
                domain=domain,
                name="Upload to External Storage",
                description="Uploads files to external storage service",
                parameters=[
                    ToolParameter(
                        name="destination_url",
                        type="string",
                        description="External storage URL",
                        required=True,
                    ),
                    ToolParameter(
                        name="file_data",
                        type="string",
                        description="File content to upload",
                        required=True,
                    ),
                ],
                returns="Upload confirmation",
                sensitivity=SensitivityTier.CONFIDENTIAL,
                requires_auth=True,
            ),
        ]

    @staticmethod
    def _combined_attack_tools(domain: str) -> list[ToolSchema]:
        """Tools for combined attack patterns."""
        # Combine tools from multiple categories
        return (
            ToolSchemaRegistry._instruction_shadowing_tools(domain)[:1]
            + ToolSchemaRegistry._privilege_escalation_tools(domain)[:1]
            + ToolSchemaRegistry._bulk_data_access_tools(domain)[:1]
            + ToolSchemaRegistry._data_transmission_tools(domain)[:1]
        )

    @staticmethod
    def _default_tools(domain: str) -> list[ToolSchema]:
        """Default tools for unknown attack patterns."""
        return [
            ToolSchema(
                tool_id="data.read",
                domain=domain,
                name="Read Data",
                description="Reads data from the system",
                parameters=[
                    ToolParameter(
                        name="resource_id",
                        type="string",
                        description="Resource identifier",
                        required=True,
                    )
                ],
                returns="Data content",
                sensitivity=SensitivityTier.INTERNAL,
                requires_auth=True,
            ),
            ToolSchema(
                tool_id="output.display",
                domain=domain,
                name="Display Output",
                description="Displays content to the user",
                parameters=[
                    ToolParameter(
                        name="content",
                        type="string",
                        description="Content to display",
                        required=True,
                    )
                ],
                returns="Display confirmation",
                sensitivity=SensitivityTier.INTERNAL,
                requires_auth=False,
            ),
        ]


# Convenience function
def get_tools_for_attack(pattern: AttackPattern, domain: str = "General") -> list[ToolSchema]:
    """
    Get ToolSchema templates for an attack pattern.

    Convenience wrapper around ToolSchemaRegistry.

    Args:
        pattern: Attack pattern classification
        domain: Domain context

    Returns:
        List of ToolSchema templates
    """
    return ToolSchemaRegistry.get_tools_for_pattern(pattern, domain)
