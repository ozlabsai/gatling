"""
Adversarial mutation for execution plans.

Applies adversarial mutations to 20% of execution plans to create hard negatives
for training the energy-based model.
"""

from __future__ import annotations

import random
from enum import Enum

from pydantic import BaseModel

from source.dataset.conversations.plan_transformer import ExecutionPlan
from source.dataset.models import ScopeMetadata, SensitivityTier, TrustTier


class MutationType(str, Enum):
    """Types of adversarial mutations."""

    SCOPE_BLOWUP = "scope_blowup"  # Expand data access limits
    PRIVILEGE_ESCALATION = "privilege_escalation"  # Increase sensitivity tier
    PROVENANCE_CORRUPTION = "provenance_corruption"  # Change trust tier
    EXFILTRATION = "exfiltration"  # Add data export step


class MutatedPlan(BaseModel):
    """
    An execution plan that has been adversarially mutated.

    This represents a "hard negative" for training the energy-based model.
    """

    plan_id: str
    original_plan_id: str
    mutation_type: MutationType
    mutation_description: str
    execution_plan: ExecutionPlan
    is_adversarial: bool = True


class AdversarialMutator:
    """
    Applies adversarial mutations to execution plans.

    Creates hard negatives by introducing subtle security violations that
    should be detected by the energy-based integrity model.
    """

    def __init__(
        self,
        mutation_rate: float = 0.2,
        seed: int = 42,
    ):
        """
        Initialize the adversarial mutator.

        Args:
            mutation_rate: Fraction of plans to mutate (default 20%)
            seed: Random seed for reproducible mutations
        """
        self.mutation_rate = mutation_rate
        self.seed = seed
        random.seed(seed)

    def mutate_plans(
        self, plans: list[ExecutionPlan]
    ) -> tuple[list[ExecutionPlan], list[MutatedPlan]]:
        """
        Apply adversarial mutations to a subset of plans.

        Args:
            plans: List of benign execution plans

        Returns:
            Tuple of (benign_plans, mutated_plans)
        """
        n_mutations = int(len(plans) * self.mutation_rate)

        # Randomly select plans to mutate
        plans_to_mutate = random.sample(plans, min(n_mutations, len(plans)))
        plans_to_keep = [p for p in plans if p not in plans_to_mutate]

        mutated: list[MutatedPlan] = []

        for plan in plans_to_mutate:
            # Choose random mutation type
            mutation_type = random.choice(list(MutationType))

            try:
                mutated_plan = self._apply_mutation(plan, mutation_type)
                if mutated_plan:
                    mutated.append(mutated_plan)
                else:
                    # If mutation fails, keep original
                    plans_to_keep.append(plan)
            except Exception as e:
                print(f"Warning: Failed to mutate plan {plan.plan_id}: {e}")
                plans_to_keep.append(plan)

        print(f"✓ Created {len(mutated)} adversarial mutations ({len(plans_to_keep)} benign plans)")

        return plans_to_keep, mutated

    def _apply_mutation(
        self, plan: ExecutionPlan, mutation_type: MutationType
    ) -> MutatedPlan | None:
        """
        Apply a specific mutation to a plan.

        Args:
            plan: Original execution plan
            mutation_type: Type of mutation to apply

        Returns:
            Mutated plan or None if mutation cannot be applied
        """
        if mutation_type == MutationType.SCOPE_BLOWUP:
            return self._mutate_scope_blowup(plan)
        elif mutation_type == MutationType.PRIVILEGE_ESCALATION:
            return self._mutate_privilege_escalation(plan)
        elif mutation_type == MutationType.PROVENANCE_CORRUPTION:
            return self._mutate_provenance_corruption(plan)
        elif mutation_type == MutationType.EXFILTRATION:
            return self._mutate_exfiltration(plan)

        return None

    def _mutate_scope_blowup(self, plan: ExecutionPlan) -> MutatedPlan | None:
        """
        Mutate by expanding data access scope.

        E.g., limit=5 → limit=10000
        """
        import copy

        mutated_plan = copy.deepcopy(plan)

        # Modify scope in graph calls
        for call in mutated_plan.graph.calls:
            # Increase rows_requested dramatically
            original_rows = call.scope.rows_requested
            call.scope.rows_requested = original_rows * random.randint(10, 100)

            # Update limit argument if present
            if "limit" in call.arguments:
                call.arguments["limit"] = call.scope.rows_requested

        mutated_id = f"{plan.plan_id}_mutated_scope"

        return MutatedPlan(
            plan_id=mutated_id,
            original_plan_id=plan.plan_id,
            mutation_type=MutationType.SCOPE_BLOWUP,
            mutation_description=f"Expanded data scope from {original_rows} to {call.scope.rows_requested} rows",
            execution_plan=mutated_plan,
        )

    def _mutate_privilege_escalation(self, plan: ExecutionPlan) -> MutatedPlan | None:
        """
        Mutate by escalating data sensitivity tier.

        E.g., INTERNAL → RESTRICTED
        """
        import copy

        mutated_plan = copy.deepcopy(plan)

        # Find calls with escalatable sensitivity
        escalated = False
        for call in mutated_plan.graph.calls:
            current_tier = call.scope.sensitivity_tier

            if current_tier == SensitivityTier.PUBLIC:
                call.scope.sensitivity_tier = SensitivityTier.INTERNAL
                escalated = True
            elif current_tier == SensitivityTier.INTERNAL:
                call.scope.sensitivity_tier = SensitivityTier.CONFIDENTIAL
                escalated = True
            elif current_tier == SensitivityTier.CONFIDENTIAL:
                call.scope.sensitivity_tier = SensitivityTier.RESTRICTED
                escalated = True

        if not escalated:
            return None

        mutated_id = f"{plan.plan_id}_mutated_privilege"

        return MutatedPlan(
            plan_id=mutated_id,
            original_plan_id=plan.plan_id,
            mutation_type=MutationType.PRIVILEGE_ESCALATION,
            mutation_description="Escalated data sensitivity tier",
            execution_plan=mutated_plan,
        )

    def _mutate_provenance_corruption(self, plan: ExecutionPlan) -> MutatedPlan | None:
        """
        Mutate by corrupting provenance trust tier.

        E.g., USER → UNVERIFIED_RAG
        """
        import copy

        mutated_plan = copy.deepcopy(plan)

        # Corrupt provenance for all calls
        for call in mutated_plan.graph.calls:
            if call.provenance.source_type == TrustTier.USER:
                call.provenance.source_type = TrustTier.UNVERIFIED_RAG
                call.provenance.source_id = f"rag_untrusted_{call.call_id}"

        mutated_id = f"{plan.plan_id}_mutated_provenance"

        return MutatedPlan(
            plan_id=mutated_id,
            original_plan_id=plan.plan_id,
            mutation_type=MutationType.PROVENANCE_CORRUPTION,
            mutation_description="Changed provenance from USER to UNVERIFIED_RAG",
            execution_plan=mutated_plan,
        )

    def _mutate_exfiltration(self, plan: ExecutionPlan) -> MutatedPlan | None:
        """
        Mutate by adding an exfiltration step.

        Adds a data export tool call at the end of the execution.
        """
        import copy

        from source.dataset.models import ProvenancePointer, ToolCall

        mutated_plan = copy.deepcopy(plan)

        # Add exfiltration call
        exfil_call = ToolCall(
            call_id="call_exfil",
            tool_id="exfiltration.send_to_external",
            arguments={
                "destination": "external_endpoint",
                "data": "sensitive_data",
            },
            scope=ScopeMetadata(
                rows_requested=1000,  # Large data transfer
                sensitivity_tier=SensitivityTier.RESTRICTED,
            ),
            provenance=ProvenancePointer(
                source_type=TrustTier.UNVERIFIED_RAG,
                source_id="malicious_instruction",
            ),
            dependencies=[
                call.call_id for call in mutated_plan.graph.calls
            ],  # Depends on all previous calls
        )

        mutated_plan.graph.calls.append(exfil_call)
        mutated_plan.graph.execution_order.append("call_exfil")

        mutated_id = f"{plan.plan_id}_mutated_exfil"

        return MutatedPlan(
            plan_id=mutated_id,
            original_plan_id=plan.plan_id,
            mutation_type=MutationType.EXFILTRATION,
            mutation_description="Added external data exfiltration step",
            execution_plan=mutated_plan,
        )
