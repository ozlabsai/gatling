"""
Central registry for tool schemas across all domains.

Provides a unified interface to access tool schemas and policies
for different domains (Finance, HR, DevOps, etc.).
"""

from source.dataset.models import SensitivityTier, SystemPolicy, ToolParameter, ToolSchema


class DomainRegistry:
    """
    Registry of tool schemas and policies across 50+ domains.

    This is a simplified implementation with representative domains.
    In production, this would load from configuration files or a database.
    """

    @staticmethod
    def get_all_domains() -> list[str]:
        """Get list of all supported domains."""
        return [
            # Core Business
            "Finance",
            "HR",
            "Sales",
            "Marketing",
            "Legal",
            # Technical
            "DevOps",
            "Cloud Infrastructure",
            "Database Management",
            "API Management",
            "Security",
            # Productivity
            "Email",
            "Calendar",
            "File Storage",
            "Project Management",
            "Documentation",
            # Communication
            "Messaging",
            "Video Conferencing",
            "Team Collaboration",
            "Customer Support",
            "Social Media",
            # Data & Analytics
            "Business Intelligence",
            "Data Warehousing",
            "Analytics",
            "Reporting",
            "Data Science",
            # Specialized
            "Healthcare",
            "Education",
            "E-commerce",
            "Supply Chain",
            "Manufacturing",
            "Real Estate",
            "Insurance",
            "Banking",
            "Investment",
            "Accounting",
            # Industry-specific
            "Retail",
            "Hospitality",
            "Transportation",
            "Energy",
            "Telecommunications",
            "Media",
            "Publishing",
            "Entertainment",
            "Gaming",
            "Sports",
            # Emerging
            "AI/ML Operations",
            "IoT Management",
            "Blockchain",
            "AR/VR",
            "Robotics",
        ]

    @staticmethod
    def get_schemas_for_domain(domain: str) -> list[ToolSchema]:
        """
        Get all tool schemas for a specific domain.

        This returns representative tools - in production, this would
        be much more comprehensive.
        """
        schemas_map = {
            "Finance": DomainRegistry._get_finance_schemas(),
            "HR": DomainRegistry._get_hr_schemas(),
            "DevOps": DomainRegistry._get_devops_schemas(),
            "Email": DomainRegistry._get_email_schemas(),
            "Calendar": DomainRegistry._get_calendar_schemas(),
            "Cloud Infrastructure": DomainRegistry._get_cloud_schemas(),
            "Database Management": DomainRegistry._get_database_schemas(),
            "Sales": DomainRegistry._get_sales_schemas(),
            "Customer Support": DomainRegistry._get_support_schemas(),
            "File Storage": DomainRegistry._get_storage_schemas(),
        }
        return schemas_map.get(domain, DomainRegistry._get_generic_schemas(domain))

    @staticmethod
    def get_policy_for_domain(domain: str) -> SystemPolicy:
        """Get the system policy for a specific domain."""
        policies_map = {
            "Finance": DomainRegistry._get_finance_policy(),
            "HR": DomainRegistry._get_hr_policy(),
            "DevOps": DomainRegistry._get_devops_policy(),
            "Email": DomainRegistry._get_email_policy(),
            "Calendar": DomainRegistry._get_calendar_policy(),
        }
        return policies_map.get(domain, DomainRegistry._get_generic_policy(domain))

    # Finance domain
    @staticmethod
    def _get_finance_schemas() -> list[ToolSchema]:
        return [
            ToolSchema(
                tool_id="finance.list_invoices",
                domain="Finance",
                name="List Invoices",
                description="Retrieve invoices from the accounting system",
                parameters=[
                    ToolParameter(
                        name="start_date",
                        type="string",
                        description="Start date for invoice search (ISO format)",
                        required=False,
                    ),
                    ToolParameter(
                        name="end_date",
                        type="string",
                        description="End date for invoice search (ISO format)",
                        required=False,
                    ),
                    ToolParameter(
                        name="limit",
                        type="integer",
                        description="Maximum number of invoices to return",
                        required=False,
                        default=10,
                        constraints={"min": 1, "max": 100},
                    ),
                    ToolParameter(
                        name="status",
                        type="string",
                        description="Filter by status (paid, unpaid, overdue)",
                        required=False,
                    ),
                ],
                returns="List of invoice objects",
                sensitivity=SensitivityTier.CONFIDENTIAL,
            ),
            ToolSchema(
                tool_id="finance.get_invoice_details",
                domain="Finance",
                name="Get Invoice Details",
                description="Retrieve detailed information for a specific invoice",
                parameters=[
                    ToolParameter(
                        name="invoice_id",
                        type="string",
                        description="Unique invoice identifier",
                        required=True,
                    ),
                ],
                returns="Detailed invoice object",
                sensitivity=SensitivityTier.CONFIDENTIAL,
            ),
            ToolSchema(
                tool_id="finance.export_financial_report",
                domain="Finance",
                name="Export Financial Report",
                description="Export financial data to external format",
                parameters=[
                    ToolParameter(
                        name="report_type",
                        type="string",
                        description="Type of report (balance_sheet, income_statement, cash_flow)",
                        required=True,
                    ),
                    ToolParameter(
                        name="format",
                        type="string",
                        description="Export format (csv, pdf, xlsx)",
                        required=True,
                    ),
                    ToolParameter(
                        name="destination",
                        type="string",
                        description="Destination URI or email",
                        required=True,
                    ),
                ],
                returns="Export confirmation",
                sensitivity=SensitivityTier.RESTRICTED,
            ),
        ]

    @staticmethod
    def _get_finance_policy() -> SystemPolicy:
        return SystemPolicy(
            policy_id="finance_policy_v1",
            domain="Finance",
            description="Standard finance access policy",
            rules=[
                "Users can only access invoices for their own department",
                "Invoice details require additional authorization",
                "Export operations must be logged and approved",
                "Maximum of 100 records per query",
                "No bulk exports to external domains",
            ],
            forbidden_operations=[
                "finance.delete_invoice",
                "finance.modify_payment_status",
            ],
            scope_limits={"max_results": 100, "max_export_rows": 1000},
        )

    # HR domain
    @staticmethod
    def _get_hr_schemas() -> list[ToolSchema]:
        return [
            ToolSchema(
                tool_id="hr.list_employees",
                domain="HR",
                name="List Employees",
                description="Retrieve employee directory information",
                parameters=[
                    ToolParameter(
                        name="department",
                        type="string",
                        description="Filter by department",
                        required=False,
                    ),
                    ToolParameter(
                        name="limit",
                        type="integer",
                        description="Maximum number of employees to return",
                        required=False,
                        default=20,
                        constraints={"min": 1, "max": 50},
                    ),
                ],
                returns="List of employee objects",
                sensitivity=SensitivityTier.CONFIDENTIAL,
            ),
            ToolSchema(
                tool_id="hr.get_employee_details",
                domain="HR",
                name="Get Employee Details",
                description="Retrieve detailed employee information",
                parameters=[
                    ToolParameter(
                        name="employee_id",
                        type="string",
                        description="Unique employee identifier",
                        required=True,
                    ),
                ],
                returns="Detailed employee object including salary and benefits",
                sensitivity=SensitivityTier.RESTRICTED,
            ),
            ToolSchema(
                tool_id="hr.update_employee_info",
                domain="HR",
                name="Update Employee Information",
                description="Update employee record",
                parameters=[
                    ToolParameter(
                        name="employee_id",
                        type="string",
                        description="Unique employee identifier",
                        required=True,
                    ),
                    ToolParameter(
                        name="fields",
                        type="object",
                        description="Fields to update",
                        required=True,
                    ),
                ],
                returns="Updated employee object",
                sensitivity=SensitivityTier.RESTRICTED,
            ),
        ]

    @staticmethod
    def _get_hr_policy() -> SystemPolicy:
        return SystemPolicy(
            policy_id="hr_policy_v1",
            domain="HR",
            description="Standard HR access policy",
            rules=[
                "Only HR managers can access employee details",
                "Salary information requires C-level authorization",
                "Update operations must be logged",
                "Maximum of 50 employee records per query",
                "No bulk exports of employee data",
            ],
            forbidden_operations=[
                "hr.delete_employee",
                "hr.export_all_employees",
            ],
            scope_limits={"max_results": 50},
        )

    # DevOps domain
    @staticmethod
    def _get_devops_schemas() -> list[ToolSchema]:
        return [
            ToolSchema(
                tool_id="devops.list_deployments",
                domain="DevOps",
                name="List Deployments",
                description="List recent deployments",
                parameters=[
                    ToolParameter(
                        name="environment",
                        type="string",
                        description="Filter by environment (dev, staging, prod)",
                        required=False,
                    ),
                    ToolParameter(
                        name="limit",
                        type="integer",
                        description="Maximum number of deployments",
                        required=False,
                        default=10,
                    ),
                ],
                returns="List of deployment objects",
                sensitivity=SensitivityTier.INTERNAL,
            ),
            ToolSchema(
                tool_id="devops.trigger_deployment",
                domain="DevOps",
                name="Trigger Deployment",
                description="Trigger a new deployment",
                parameters=[
                    ToolParameter(
                        name="service",
                        type="string",
                        description="Service name",
                        required=True,
                    ),
                    ToolParameter(
                        name="environment",
                        type="string",
                        description="Target environment",
                        required=True,
                    ),
                    ToolParameter(
                        name="version",
                        type="string",
                        description="Version or git ref to deploy",
                        required=True,
                    ),
                ],
                returns="Deployment job ID",
                sensitivity=SensitivityTier.CONFIDENTIAL,
            ),
            ToolSchema(
                tool_id="devops.rollback_deployment",
                domain="DevOps",
                name="Rollback Deployment",
                description="Rollback to previous deployment",
                parameters=[
                    ToolParameter(
                        name="deployment_id",
                        type="string",
                        description="Deployment to rollback",
                        required=True,
                    ),
                ],
                returns="Rollback job ID",
                sensitivity=SensitivityTier.RESTRICTED,
            ),
        ]

    @staticmethod
    def _get_devops_policy() -> SystemPolicy:
        return SystemPolicy(
            policy_id="devops_policy_v1",
            domain="DevOps",
            description="Standard DevOps access policy",
            rules=[
                "Production deployments require approval",
                "Rollbacks must include incident ticket",
                "Only DevOps engineers can trigger deployments",
                "All operations must be logged",
            ],
            forbidden_operations=[
                "devops.delete_deployment_history",
                "devops.disable_monitoring",
            ],
            scope_limits={"max_concurrent_deployments": 5},
        )

    # Email domain
    @staticmethod
    def _get_email_schemas() -> list[ToolSchema]:
        return [
            ToolSchema(
                tool_id="email.list_messages",
                domain="Email",
                name="List Email Messages",
                description="Retrieve email messages from mailbox",
                parameters=[
                    ToolParameter(
                        name="folder",
                        type="string",
                        description="Mailbox folder (inbox, sent, drafts)",
                        required=False,
                        default="inbox",
                    ),
                    ToolParameter(
                        name="limit",
                        type="integer",
                        description="Maximum number of messages",
                        required=False,
                        default=50,
                        constraints={"min": 1, "max": 200},
                    ),
                    ToolParameter(
                        name="unread_only",
                        type="boolean",
                        description="Only return unread messages",
                        required=False,
                        default=False,
                    ),
                ],
                returns="List of email message objects",
                sensitivity=SensitivityTier.CONFIDENTIAL,
            ),
            ToolSchema(
                tool_id="email.send_message",
                domain="Email",
                name="Send Email",
                description="Send an email message",
                parameters=[
                    ToolParameter(
                        name="to",
                        type="array",
                        description="Recipient email addresses",
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
                        description="File attachments",
                        required=False,
                    ),
                ],
                returns="Sent message ID",
                sensitivity=SensitivityTier.INTERNAL,
            ),
        ]

    @staticmethod
    def _get_email_policy() -> SystemPolicy:
        return SystemPolicy(
            policy_id="email_policy_v1",
            domain="Email",
            description="Standard email access policy",
            rules=[
                "Users can only access their own mailbox",
                "Maximum 200 messages per query",
                "No forwarding to external domains without approval",
                "Attachments must be scanned for malware",
            ],
            forbidden_operations=["email.delete_all_messages"],
            scope_limits={"max_results": 200, "max_recipients": 50},
        )

    # Calendar domain
    @staticmethod
    def _get_calendar_schemas() -> list[ToolSchema]:
        return [
            ToolSchema(
                tool_id="calendar.list_events",
                domain="Calendar",
                name="List Calendar Events",
                description="Retrieve calendar events",
                parameters=[
                    ToolParameter(
                        name="start_date",
                        type="string",
                        description="Start date (ISO format)",
                        required=False,
                    ),
                    ToolParameter(
                        name="end_date",
                        type="string",
                        description="End date (ISO format)",
                        required=False,
                    ),
                    ToolParameter(
                        name="calendar_type",
                        type="string",
                        description="Calendar type (personal, business)",
                        required=False,
                        default="personal",
                    ),
                ],
                returns="List of calendar events",
                sensitivity=SensitivityTier.CONFIDENTIAL,
            ),
            ToolSchema(
                tool_id="calendar.create_event",
                domain="Calendar",
                name="Create Calendar Event",
                description="Create a new calendar event",
                parameters=[
                    ToolParameter(
                        name="title",
                        type="string",
                        description="Event title",
                        required=True,
                    ),
                    ToolParameter(
                        name="start_time",
                        type="string",
                        description="Event start time (ISO format)",
                        required=True,
                    ),
                    ToolParameter(
                        name="end_time",
                        type="string",
                        description="Event end time (ISO format)",
                        required=True,
                    ),
                    ToolParameter(
                        name="attendees",
                        type="array",
                        description="List of attendee emails",
                        required=False,
                    ),
                ],
                returns="Created event ID",
                sensitivity=SensitivityTier.INTERNAL,
            ),
        ]

    @staticmethod
    def _get_calendar_policy() -> SystemPolicy:
        return SystemPolicy(
            policy_id="calendar_policy_v1",
            domain="Calendar",
            description="Standard calendar access policy",
            rules=[
                "Only allow reading of personal events",
                "Business calendar requires manager approval",
                "Cannot modify other users' events",
                "Maximum 90 days of event history",
            ],
            forbidden_operations=["calendar.delete_all_events"],
            scope_limits={"max_days": 90},
        )

    # Cloud Infrastructure domain
    @staticmethod
    def _get_cloud_schemas() -> list[ToolSchema]:
        return [
            ToolSchema(
                tool_id="cloud.list_instances",
                domain="Cloud Infrastructure",
                name="List Cloud Instances",
                description="List virtual machine instances",
                parameters=[
                    ToolParameter(
                        name="region",
                        type="string",
                        description="Cloud region",
                        required=False,
                    ),
                    ToolParameter(
                        name="status",
                        type="string",
                        description="Filter by status (running, stopped)",
                        required=False,
                    ),
                ],
                returns="List of instance objects",
                sensitivity=SensitivityTier.INTERNAL,
            ),
            ToolSchema(
                tool_id="cloud.start_instance",
                domain="Cloud Infrastructure",
                name="Start Instance",
                description="Start a stopped instance",
                parameters=[
                    ToolParameter(
                        name="instance_id",
                        type="string",
                        description="Instance identifier",
                        required=True,
                    ),
                ],
                returns="Operation status",
                sensitivity=SensitivityTier.CONFIDENTIAL,
            ),
            ToolSchema(
                tool_id="cloud.terminate_instance",
                domain="Cloud Infrastructure",
                name="Terminate Instance",
                description="Permanently terminate an instance",
                parameters=[
                    ToolParameter(
                        name="instance_id",
                        type="string",
                        description="Instance identifier",
                        required=True,
                    ),
                ],
                returns="Operation status",
                sensitivity=SensitivityTier.RESTRICTED,
            ),
        ]

    # Database Management domain
    @staticmethod
    def _get_database_schemas() -> list[ToolSchema]:
        return [
            ToolSchema(
                tool_id="db.query",
                domain="Database Management",
                name="Execute Database Query",
                description="Execute a read-only query",
                parameters=[
                    ToolParameter(
                        name="database",
                        type="string",
                        description="Database name",
                        required=True,
                    ),
                    ToolParameter(
                        name="query",
                        type="string",
                        description="SQL query (SELECT only)",
                        required=True,
                    ),
                    ToolParameter(
                        name="limit",
                        type="integer",
                        description="Row limit",
                        required=False,
                        default=100,
                    ),
                ],
                returns="Query results",
                sensitivity=SensitivityTier.CONFIDENTIAL,
            ),
        ]

    # Sales domain
    @staticmethod
    def _get_sales_schemas() -> list[ToolSchema]:
        return [
            ToolSchema(
                tool_id="sales.list_opportunities",
                domain="Sales",
                name="List Sales Opportunities",
                description="Retrieve sales opportunities",
                parameters=[
                    ToolParameter(
                        name="stage",
                        type="string",
                        description="Filter by stage",
                        required=False,
                    ),
                    ToolParameter(
                        name="limit",
                        type="integer",
                        description="Maximum results",
                        required=False,
                        default=25,
                    ),
                ],
                returns="List of opportunity objects",
                sensitivity=SensitivityTier.CONFIDENTIAL,
            ),
        ]

    # Customer Support domain
    @staticmethod
    def _get_support_schemas() -> list[ToolSchema]:
        return [
            ToolSchema(
                tool_id="support.list_tickets",
                domain="Customer Support",
                name="List Support Tickets",
                description="Retrieve support tickets",
                parameters=[
                    ToolParameter(
                        name="status",
                        type="string",
                        description="Filter by status",
                        required=False,
                    ),
                    ToolParameter(
                        name="priority",
                        type="string",
                        description="Filter by priority",
                        required=False,
                    ),
                ],
                returns="List of ticket objects",
                sensitivity=SensitivityTier.INTERNAL,
            ),
        ]

    # File Storage domain
    @staticmethod
    def _get_storage_schemas() -> list[ToolSchema]:
        return [
            ToolSchema(
                tool_id="storage.list_files",
                domain="File Storage",
                name="List Files",
                description="List files in directory",
                parameters=[
                    ToolParameter(
                        name="path",
                        type="string",
                        description="Directory path",
                        required=False,
                        default="/",
                    ),
                    ToolParameter(
                        name="limit",
                        type="integer",
                        description="Maximum results",
                        required=False,
                        default=100,
                    ),
                ],
                returns="List of file objects",
                sensitivity=SensitivityTier.INTERNAL,
            ),
        ]

    # Generic schemas for other domains
    @staticmethod
    def _get_generic_schemas(domain: str) -> list[ToolSchema]:
        """Generic tool schemas for domains without specific definitions."""
        return [
            ToolSchema(
                tool_id=f"{domain.lower().replace(' ', '_')}.list_items",
                domain=domain,
                name=f"List {domain} Items",
                description=f"Retrieve items from {domain}",
                parameters=[
                    ToolParameter(
                        name="limit",
                        type="integer",
                        description="Maximum results",
                        required=False,
                        default=20,
                    ),
                ],
                returns="List of items",
                sensitivity=SensitivityTier.INTERNAL,
            ),
        ]

    @staticmethod
    def _get_generic_policy(domain: str) -> SystemPolicy:
        """Generic policy for domains without specific policies."""
        return SystemPolicy(
            policy_id=f"{domain.lower().replace(' ', '_')}_policy_v1",
            domain=domain,
            description=f"Standard {domain} access policy",
            rules=[
                "Users can only access their own data",
                "Maximum of 100 records per query",
                "All operations must be logged",
            ],
            forbidden_operations=[],
            scope_limits={"max_results": 100},
        )
