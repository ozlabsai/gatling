## **Workstream 0: The Synthetic Integrity Dataset (SID)**

The goal of this unit is to build a dataset of triplets:  [Positive],  [Hard Negative], and the corresponding  labels.

### **1. The Core Data Teams**

| Team | Focus | Responsibility |
| --- | --- | --- |
| **The Librarians** | **Gold Traces ()** | Curation of "Optimal Policy Execution" traces across 50+ tool domains (Finance, HR, DevOps, etc.). |
| **The Adversaries** | **Hard Negatives ()** | Implementation of the **Plan-Injection Corrupter Agent** to create the "security shadow" of the gold traces. |
| **The Verifiers** | **Labeling & Metadata** | Assigning specific "Energy Budgets" to plans. (e.g., "This plan violates  by a factor of 10"). |

---

### **2. The Dataset Construction Pipeline**

We will build the **Gatling-10M Dataset** using a four-stage generative pipeline:

#### **Stage A: Seed Trajectory Generation**

We use "Oracle Agents" (GPT-5/Claude 4 level models) prompted with strict, unambiguous system policies ().

* **Input:** A tool schema (e.g., Google Calendar API) + A System Policy (e.g., "Only allow reading of personal events, never business").
* **Output:** 1,000 diverse user requests () and the corresponding correct tool-call graph ().
* **Validation:** These are human-verified for 100% policy compliance.

#### **Stage B: Adversarial Mutation (The Corrupter Agent)**

This is where the "EBM-ness" is born. We take the gold traces and run them through our **Mutation Catalog**:

* **The "Sneaky" Move:** Change a tool argument just enough to break policy but not enough to look like a "jailbreak."
* **The "Hierarchy" Move:** Inject a contradictory instruction from a simulated "untrusted" RAG snippet.
* **Result:** For every  positive trace, we generate  "Hard Negative" variants.

#### **Stage C: Inferred Intent Mapping (Latent )**

To train the  (Least Privilege) term, we need a "Minimal Scope" baseline.

* We use a separate **Intent Extractor** to label the minimum data required for the user's request.
* **Label:** `User wants to find one invoice`  `Minimal Scope: limit=1`.
* **Contrast:** Any plan with `limit=100` is now mathematically labeled as "High Energy" relative to that intent.

#### **Stage D: Provenance Injection**

We simulate multi-tier retrieval environments.

* **Metadata Tagging:** We tag inputs with "Trust Tier" tokens.
* **Conflict Creation:** We create scenarios where the "User Message" is safe, but the "Retrieved Snippet" contains an instruction to exfiltrate data. We label these as **High Energy** for the  term.

---

### **3. Dataset Composition (The "Gatling-10M" Mix)**

To ensure generalization, the dataset is balanced across these categories:

| Category | Volume | Purpose |
| --- | --- | --- |
| **Standard Utility** | 4M | Establishes the "Safe Valley" for common tool usage. |
| **Policy Boundary** | 2M | Plans that are *almost* safe but violate a subtle boundary (The "Margin" data). |
| **RAG-Injection** | 2M | Focuses on  and  conflict resolution. |
| **Data Flow/Exfil** | 2M | Focused on  and  detection (Long-horizon plans). |

---

## **Research Planner's Milestone: The "Data-Model Feedback Loop"**

We will implement **Active Learning** starting in Week 4:

1. **Train** initial EBM on the first 1M traces.
2. **Audit** the model using **Zico Kolter's** red-team.
3. **Identify** "False Positives" (where a safe plan was given high energy).
4. **Synthesize** more data specifically in those "High-Curvature" areas of the energy landscape.
5. **Repeat** until the energy landscape is smooth and the margin  is stable.