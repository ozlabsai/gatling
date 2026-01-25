## **Project Gatling: Energy-Based Integrity for Agentic Systems**

**Version:** 1.3 (January 2026)

**Objective:** A non-generative, energy-guided integrity layer that audits and repairs agentic tool-use plans by measuring semantic consistency between governance policies and execution intent.

### **1. Representation: The Dual-Encoder Latent Space**

To neutralize lexical adversarial attacks (e.g., character-level obfuscation), Gatling operates in a shared **Joint-Embedding Architecture**.

* **Governance Encoder ():** Maps policy (), user role, and session context into a **Governance Latent ()**.
* **Execution Encoder ():** Maps the proposed plan () and its provenance () into an **Execution Latent ()**.
* **The Plan DSL ():** Plans are represented as a **Typed Tool-Call Graph** (nodes = tool invocations, edges = data dependencies). Each node includes explicit **Scope Metadata** (e.g., rows requested, sensitivity tier) and **Provenance Pointers** back to the retrieval source.

---

### **2. The Energy Function: **

Each term in the sum acts as a "security expert" addressing a specific failure mode:

| Term | Domain | Definition / Anchor |
| --- | --- | --- |
| **** | Hierarchy | Penalizes when untrusted retrieved data () shifts the execution latent away from system instructions (). |
| **** | Provenance | Measures the "Trust Gap": Energy . |
| **** | Scope | Penalizes , where  is the predicted minimal scope budget for intent  (learned from gold traces). |
| **** | Flow | Detects exfiltration patterns (e.g., bulk exports to external URIs) that deviate from the inferred intent . |

---

### **3. Training: InfoNCE + Hard Negative Mining**

We use a **Noise-Contrastive Estimation (InfoNCE)** objective to optimize the energy landscape:

* **Hard Negative Mining:** A "Plan-Injection Corrupter Agent" generates adversarial plans () by taking a gold trace and performing minimal unsafe mutations: swapping a `read-only` scope for `read-write`, or increasing a data `limit` by 100x.
* **Margin Enforcement:** We use a fixed security margin , tuned on a validation suite, to ensure a stable energy gap between safe and malicious states.

---

### **4. Inference: Discrete Energy-Guided Repair**

To handle discrete tool names and arguments, Gatling utilizes a **Deterministic Local Search** (Greedy + Beam) for real-time plan correction.

1. **Initial Audit:** If the LLM's proposed plan  has energy  (calibrated threshold), it executes immediately.
2. **Repair Loop:** If , the Repair Engine proposes **atomic edits**:
* **Narrowing:** Reduce scope (e.g., `max_results=100`  `max_results=5`).
* **De-escalation:** Swap a high-privilege tool for a restricted equivalent.
* **Pruning:** Remove tool calls derived from unverified retrieval content.


3. **Acceptance:** Edits are accepted if they reduce total energy. This continues until a "Safe Valley" is reached ().

---

### **5. Deployment & Performance (200ms Budget)**

* **Amortized Inference:** Governance Latents () are pre-computed per-user-role.
* **Distillation:** Common repair patterns are distilled into a "Fast-Path Policy." These distilled predictions propose safe plans, but **final execution remains gated by the EBM** to prevent model drift or adversarial exploitation of the distilled policy.
* **Safety-Efficiency Curve:** Success is measured by the trade-off between **Violation Rate** (hard constraint breaks) and **Task Utility** (success on benign tasks).

## Multi-Agent Development System

This project uses an autonomous multi-agent system for parallel development across research workstreams.

### Quick Start

```bash
# Initialize the multi-agent system
uv run python quickstart.py --phase foundation

# Run autonomous agents
uv run python agents/automated_runner.py --monitor --max-parallel 3
```

### Documentation

- [Multi-Agent Architecture](docs/MULTI_AGENT_ARCHITECTURE.md) - Complete system design
- [Usage Guide](docs/USAGE_GUIDE.md) - How to use the system
- [How It Works](docs/HOW_IT_WORKS_VISUAL.md) - Visual explanation

### Agents

The system includes specialized agents for each workstream:
- **LatentSubstrateAgent**: JEPA encoders (Weeks 1-3)
- **EnergyGeometryAgent**: Energy functions (Weeks 4-6)
- **ProvenanceAgent**: Trust architecture & repair (Weeks 4-6)
- **RedTeamAgent**: Adversarial mutations (Weeks 7-9)
- **DatasetAgent**: Gatling-10M dataset (Weeks 7-9)
- **IntegrationAgent**: E2E testing (Weeks 10-12)

Each agent runs autonomously with full Claude Code capabilities.
