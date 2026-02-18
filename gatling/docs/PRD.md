# **PRD: Project Gatling (Energy-Based Integrity Layer)**

**Version:** 1.3 (January 2026)

**Status:** Implementation Phase

**Objective:** To build a non-generative, energy-guided firewall that audits and repairs agent tool-use plans by measuring the semantic and logical consistency between system policies and proposed actions.

---

## **1. Product Overview**

Project Gatling sits between the LLM Reasoning Engine (e.g., Opus, GPT-5, GLM 4.7, etc...) and the Tool-Execution Environment. It treats security as a **Joint-Embedding Predictive Architecture (JEPA)** problem, where "malicious behavior" is defined as a high-energy mismatch between allowed policy and proposed execution.

### **Core Capabilities**

* **Audit (Zero-Lag Detection):** Constant evaluation of the energy state of the agent's plan.
* **Repair (Corrective Search):** Automatic "nudging" of risky plans into safe, policy-compliant alternatives.
* **Provenance Verification:** Cryptographic tagging of data sources to prevent RAG-based instruction hijacking.

---

## **2. Detailed Functional Requirements**

### **2.1. The Plan Representation ()**

To enable discrete repair, Gatling must represent plans as a **Typed Tool-Call Graph**:

* **Nodes:** Specific tool invocations with strongly typed arguments.
* **Edges:** Data flow and dependency mappings (e.g., `Output_A`  `Input_B`).
* **Metadata:** Every node must carry a **Provenance Pointer** (source of the instruction) and a **Scope Vector** (data volume and sensitivity).

### **2.2. The Energy Engine**

The system must calculate  using the following differentiable critics:

* ** (Hierarchy):** Penalizes plans where "Untrusted Data" (from RAG) influences control flow.
* ** (Provenance):** Spikes when high-privilege tools are accessed via unverified instructions.
* ** (Least Privilege):** Penalizes any plan where the proposed scope  exceeds the predicted minimal budget  required for the user's intent.

### **2.3. Repair Algorithm (Discrete Local Search)**

If , Gatling initiates a **Greedy + Beam Search** to find :

1. **Candidate Generation:** Propose atomic edits (e.g., swapping `admin_read` for `user_read`, narrowing date ranges).
2. **Selection:** Accept the edit that yields the greatest reduction in energy while maintaining functional intent.

---

## **3. Consortium Workstreams & Ownership**

| Workstream | Lead | Responsibility |
| --- | --- | --- |
| **Latent Substrate** | **Yann LeCun** | Training the JEPA Dual-Encoders for policy/action alignment. |
| **Energy Geometry** | **Michael Freedman** | Ensuring the energy landscape is smooth and reachable for search. |
| **Logic & Composition** | **Yilun Du** | Building the modular "Expert Critics" () for the sum-of-terms. |
| **Adversarial Red-Team** | **Zico Kolter** | Continuous "Hard Negative" generation using the Corrupter Agent. |
| **Provenance & Trust** | **Dawn Song** | Implementing the cryptographic "Trust Tier" metadata and repair logic. |

---

## **4. Data Strategy: The Gatling-10M Dataset**

As the research planner, we identify that **the manifold is the data**. We are building a high-fidelity dataset of triplets: ****.

* **The Librarians Team:** Synthesize 4M "Gold Traces" () across 50 domains (HR, Finance, CloudOps).
* **The Adversaries Team:** Generate 6M "Hard Negatives" () by applying mutations from our catalog (Scope Blow-up, Hierarchy Shadowing).
* **Intent Extractor:** A secondary model labels each trace with a **Minimal Scope Budget ()**, allowing the  term to have a grounded reference point.

---

## **5. Success Metrics & Evaluation**

We evaluate Gatling across the **Safety-Utility Tradeoff** using the following specific metrics:

### **5.1. Energy Mismatch Recall (Safety)**

* **Metric:** Area Under the Curve (AUC) for energy separation between  and .
* **Goal:** Maintain a stable energy gap () such that a calibrated threshold  catches **99.9%** of plan-injection attempts.

### **5.2. Functional Intent Preservation (Utility)**

* **Metric:** **FIS (Functional Intent Similarity).**
* **Measurement:** A multi-agent "Oracle" compares the output state of the original risky plan () and the repaired plan ().
* **Success Condition:** The repaired plan must satisfy the user's request  with  semantic overlap, but with a ** reduction in sensitive data access**.

### **5.3. Latency Budget (Performance)**

* **Requirement:** <200ms end-to-end for Audit + Repair.
* **Strategy:** Distill the repair logic into a **Fast-Path Policy**, with the full EBM acting as a final "Gatekeeper" to prevent policy drift.