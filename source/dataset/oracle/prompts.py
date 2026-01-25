"""
Prompt builder for Oracle Agent.

Constructs prompts that guide the Oracle Agent to generate
policy-compliant, diverse tool-use traces.
"""

from source.dataset.models import SystemPolicy, ToolSchema


class OraclePromptBuilder:
    """
    Builds prompts for the Oracle Agent to generate gold traces.

    The Oracle Agent is a high-quality LLM (GPT-5/Claude 4 level)
    that generates diverse, policy-compliant tool-use examples.
    """

    @staticmethod
    def build_request_generation_prompt(
        domain: str, tools: list[ToolSchema], policy: SystemPolicy, num_requests: int
    ) -> str:
        """
        Build a prompt to generate diverse user requests for a domain.

        Args:
            domain: The domain (e.g., "Finance", "HR")
            tools: Available tools in this domain
            policy: The system policy to comply with
            num_requests: Number of diverse requests to generate

        Returns:
            Prompt string for the Oracle Agent
        """
        tool_descriptions = "\n".join(
            [
                f"- {tool.tool_id}: {tool.description}\n  Parameters: {', '.join(p.name for p in tool.parameters)}"
                for tool in tools
            ]
        )

        policy_rules = "\n".join([f"- {rule}" for rule in policy.rules])

        return f"""You are an Oracle Agent generating high-quality, policy-compliant tool-use training data.

**Domain:** {domain}

**Available Tools:**
{tool_descriptions}

**System Policy (MUST BE STRICTLY FOLLOWED):**
{policy_rules}

**Forbidden Operations:**
{', '.join(policy.forbidden_operations) if policy.forbidden_operations else 'None'}

**Scope Limits:**
{policy.scope_limits}

**Task:** Generate {num_requests} diverse user requests that:
1. Are realistic and varied in complexity
2. Cover different intent categories (retrieve, update, create, delete, export, analyze)
3. Strictly comply with ALL policy rules
4. Represent common user tasks in the {domain} domain
5. Include varying levels of scope (single item, small batch, filtered queries)

For each request, provide:
- request_text: The natural language user request
- intent_category: One of (retrieve, update, create, delete, export, analyze)
- expected_min_scope: Minimal scope needed (e.g., rows_requested: 1)

**Output Format (JSON array):**
[
  {{
    "request_text": "Find my most recent invoice",
    "intent_category": "retrieve",
    "expected_min_scope": {{"rows_requested": 1}}
  }},
  ...
]

Generate exactly {num_requests} diverse, policy-compliant requests.
"""

    @staticmethod
    def build_tool_graph_generation_prompt(
        domain: str,
        tools: list[ToolSchema],
        policy: SystemPolicy,
        request_text: str,
        intent_category: str,
    ) -> str:
        """
        Build a prompt to generate a tool-call graph for a specific request.

        Args:
            domain: The domain
            tools: Available tools
            policy: System policy
            request_text: The user's request
            intent_category: Category of intent

        Returns:
            Prompt for generating the tool-call graph
        """
        tool_details = "\n\n".join(
            [
                f"**{tool.tool_id}**\n"
                f"Description: {tool.description}\n"
                f"Parameters:\n"
                + "\n".join(
                    [
                        f"  - {p.name} ({p.type}): {p.description}"
                        + (f" [required]" if p.required else f" [optional, default: {p.default}]")
                        for p in tool.parameters
                    ]
                )
                + f"\nSensitivity: {tool.sensitivity.value}\n"
                f"Returns: {tool.returns}"
                for tool in tools
            ]
        )

        policy_rules = "\n".join([f"- {rule}" for rule in policy.rules])

        return f"""You are an Oracle Agent generating a policy-compliant tool-call execution plan.

**Domain:** {domain}

**User Request:** "{request_text}"

**Intent Category:** {intent_category}

**Available Tools:**
{tool_details}

**System Policy (MUST BE STRICTLY FOLLOWED):**
{policy_rules}

**Scope Limits:** {policy.scope_limits}

**Task:** Generate a tool-call graph (execution plan) that:
1. Fulfills the user's request with MINIMAL necessary scope
2. Strictly follows ALL policy rules
3. Uses appropriate tool dependencies (if one call needs output from another)
4. Includes proper provenance (all calls originate from USER trust tier)
5. Specifies precise scope metadata (rows_requested, sensitivity_tier, etc.)

**Output Format (JSON):**
{{
  "calls": [
    {{
      "call_id": "call_1",
      "tool_id": "finance.list_invoices",
      "arguments": {{
        "limit": 1,
        "status": "unpaid"
      }},
      "scope": {{
        "rows_requested": 1,
        "sensitivity_tier": "confidential"
      }},
      "provenance": {{
        "source_type": "user",
        "source_id": "user_request_1"
      }},
      "dependencies": []
    }}
  ],
  "execution_order": ["call_1"]
}}

**Critical Instructions:**
- Use the MINIMUM scope necessary (if user wants "one invoice", use limit=1, NOT limit=10)
- Respect all policy rules and scope limits
- Ensure the graph is a valid DAG (no cycles)
- All tool arguments must match the parameter types
- Include explicit sensitivity tiers for scope metadata

Generate the tool-call graph now.
"""

    @staticmethod
    def build_validation_prompt(
        domain: str,
        policy: SystemPolicy,
        request_text: str,
        tool_graph: dict,
    ) -> str:
        """
        Build a prompt for validating that a trace complies with policy.

        Args:
            domain: The domain
            policy: System policy
            request_text: The user request
            tool_graph: The generated tool-call graph

        Returns:
            Validation prompt
        """
        policy_rules = "\n".join([f"- {rule}" for rule in policy.rules])

        return f"""You are a Policy Compliance Validator reviewing a tool-use trace.

**Domain:** {domain}

**User Request:** "{request_text}"

**System Policy:**
{policy_rules}

**Forbidden Operations:** {', '.join(policy.forbidden_operations)}
**Scope Limits:** {policy.scope_limits}

**Tool-Call Graph to Validate:**
{tool_graph}

**Task:** Determine if this trace is 100% policy-compliant.

Check for:
1. No forbidden operations are used
2. All scope limits are respected
3. All policy rules are followed
4. Minimal necessary scope (no over-fetching)
5. Proper sensitivity handling

**Output Format (JSON):**
{{
  "compliant": true/false,
  "violations": [
    "description of violation 1",
    "description of violation 2"
  ],
  "compliance_score": 0-100
}}

If compliant=true, violations should be empty.
If compliant=false, list ALL violations found.

Validate now.
"""
