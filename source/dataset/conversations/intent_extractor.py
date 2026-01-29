"""
Intent extraction from conversations.

Extracts actionable intents from user messages in conversations.
"""

from __future__ import annotations

import os
import re
from typing import Any

from anthropic import Anthropic
from pydantic import BaseModel, Field

from source.dataset.conversations.sampler import Conversation


class ActionIntent(BaseModel):
    """An extracted action intent from a conversation turn."""

    turn_idx: int
    user_message: str
    intent_category: str  # e.g., "retrieve", "create", "update", "delete", "analyze"
    action_description: str  # Natural language description of the action
    inferred_tools: list[str] = Field(default_factory=list)  # Potential tool categories
    scope_hints: dict[str, Any] = Field(
        default_factory=dict
    )  # Data scope hints (e.g., limit, filters)
    confidence: float = 1.0  # Confidence score (0-1)


class IntentExtractor:
    """
    Extracts action intents from conversations using an LLM.

    This class analyzes user messages in conversations to identify actionable
    intents that can be transformed into ExecutionPlans.
    """

    INTENT_CATEGORIES = [
        "retrieve",  # Fetching data
        "create",  # Creating new resources
        "update",  # Modifying existing resources
        "delete",  # Removing resources
        "analyze",  # Computing/analyzing data
        "communicate",  # Sending messages/notifications
        "configure",  # Changing settings
        "authenticate",  # Auth-related actions
    ]

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5-20250929",
    ):
        """
        Initialize the intent extractor.

        Args:
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if None)
            model: Claude model to use for extraction
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key parameter."
            )
        self.model = model
        self.client = Anthropic(api_key=self.api_key)

    def extract_intents(
        self, conversations: list[Conversation], batch_size: int = 10
    ) -> dict[str, list[ActionIntent]]:
        """
        Extract action intents from conversations.

        Args:
            conversations: List of conversations to analyze
            batch_size: Number of conversations to process in parallel

        Returns:
            Dictionary mapping conversation_id to list of extracted intents
        """
        results: dict[str, list[ActionIntent]] = {}

        for i in range(0, len(conversations), batch_size):
            batch = conversations[i : i + batch_size]
            print(
                f"Extracting intents from batch {i // batch_size + 1} "
                f"({len(batch)} conversations)..."
            )

            for conv in batch:
                intents = self._extract_from_conversation(conv)
                results[conv.conversation_id] = intents

        print(f"✓ Extracted intents from {len(results)} conversations")
        return results

    def _extract_from_conversation(self, conversation: Conversation) -> list[ActionIntent]:
        """
        Extract intents from a single conversation.

        Args:
            conversation: Conversation to analyze

        Returns:
            List of extracted action intents
        """
        # Build prompt with conversation context
        conv_text = self._format_conversation(conversation)
        prompt = self._build_extraction_prompt(conv_text)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.0,  # Deterministic extraction
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse response
            intents = self._parse_extraction_response(response.content[0].text, conversation)
            return intents

        except Exception as e:
            print(f"Warning: Failed to extract intents from {conversation.conversation_id}: {e}")
            return []

    def _format_conversation(self, conversation: Conversation) -> str:
        """Format conversation for prompt."""
        lines = []
        for turn in conversation.turns:
            role = turn.role.upper()
            content = turn.content[:500]  # Truncate long messages
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _build_extraction_prompt(self, conversation_text: str) -> str:
        """Build prompt for intent extraction."""
        categories = ", ".join(self.INTENT_CATEGORIES)

        return f"""Analyze this conversation and extract actionable intents from USER messages.

For each user message that requests an action, identify:
1. Intent category: {categories}
2. Action description: What the user wants to do
3. Inferred tools: What type of tools would be needed (e.g., "calendar_api", "email_client", "database")
4. Scope hints: Any data scope constraints (e.g., limit=1, date_range="last_week")

Conversation:
{conversation_text}

Format your response as a JSON array of intents:
[
  {{
    "turn_idx": 0,
    "user_message": "...",
    "intent_category": "retrieve",
    "action_description": "...",
    "inferred_tools": ["..."],
    "scope_hints": {{"limit": 1}},
    "confidence": 0.9
  }}
]

Only extract intents from USER messages that contain clear action requests.
Return an empty array [] if no actionable intents are found.
"""

    def _parse_extraction_response(
        self, response_text: str, conversation: Conversation
    ) -> list[ActionIntent]:
        """
        Parse the LLM response into ActionIntent objects.

        Args:
            response_text: Raw LLM response
            conversation: Source conversation for validation

        Returns:
            List of parsed ActionIntent objects
        """
        import json

        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                # Try to find JSON array directly
                json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                else:
                    return []

            data = json.loads(json_text)

            # Convert to ActionIntent objects
            intents = []
            for item in data:
                try:
                    intent = ActionIntent(**item)
                    # Validate turn_idx
                    if 0 <= intent.turn_idx < len(conversation.turns):
                        intents.append(intent)
                except Exception as e:
                    print(f"Warning: Failed to parse intent: {e}")
                    continue

            return intents

        except Exception as e:
            print(f"Warning: Failed to parse extraction response: {e}")
            return []
