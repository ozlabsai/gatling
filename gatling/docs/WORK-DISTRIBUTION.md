## **Project Gatling: Workstream Distribution**

### **1. The Latent Substrate (Lead: Yann LeCun)**

**Focus:** Non-generative state representation and the JEPA-based world model.

* **The Goal:** Build the **Hierarchical JEPA Encoder** that projects  into a shared latent space.
* **Key Deliverables:**
* **Unified Feature Extractor:** A transformer-based encoder that strips lexical noise and outputs a 1024-dimension "Semantic Intent" vector.
* **World-State Predictor:** A model that predicts the "next safe state" given a governance policy. This serves as the reference point for the energy function.
* **Abstraction Layer:** Ensures that tool schemas (JSON-RPC, OpenAPI) are represented as functional primitives rather than raw strings.



### **2. The Energy Landscape (Lead: Yilun Du)**

**Focus:** Compositional energy terms and the logic of "Consistency."

* **The Goal:** Implementation of the **Product of Experts (PoE)** energy function .
* **Key Deliverables:**
* **Term Implementation:** Coding the specific differentiable critics for , , , and .
* **Scaling Calibration:** A normalization framework to ensure that one energy term (e.g., ) doesn't numerically overwhelm another (e.g., ).
* **The Energy API:** A high-throughput service that can score a proposed action plan against a policy latent in <20ms.



### **3. Mathematical Verification (Lead: Michael Freedman)**

**Focus:** Topological smoothness and the geometry of the "Safe Manifold."

* **The Goal:** Ensuring the energy landscape is navigable for the repair engine.
* **Key Deliverables:**
* **Landscape Smoothing:** Applying **Spectral Regularization** to the energy function to eliminate "dead-end" local minima.
* **Convergence Proofs:** Mathematically proving that for any , there exists a reachable  where  within  repair steps.
* **The "Valley" Index:** A pre-computed map of safe operational envelopes for common tool categories.



### **4. Red-Teaming & Training (Lead: Zico Kolter)**

**Focus:** Hard negative generation and adversarial robustness.

* **The Goal:** The **Plan-Injection Mutation Catalog** and the InfoNCE training loop.
* **Key Deliverables:**
* **The Corrupter Agent:** An automated pipeline that takes successful traces and applies the mutations (Scope Blow-up, Provenance Rug-Pull, etc.) defined in the catalog.
* **Adversarial Training Loop:** A continuous cycle where the EBM is challenged by new injection patterns, and the most "deceptive" samples are fed back into training.
* **The Security Margin :** Calibrating the fixed energy gap to ensure zero false positives for standard administrative tasks.



### **5. Trust Architecture & Provenance (Lead: Dawn Song)**

**Focus:** Data sovereignty, trust tiers, and production deployment.

* **The Goal:** Integrating real-world security metadata into the energy audit.
* **Key Deliverables:**
* **The Provenance Hash:** A system for tagging every retrieval snippet  with a cryptographically verified "Trust Tier" (1: Internal, 2: Signed Partner, 3: Public Web).
* **Repair Engine Implementation:** The **Discrete Greedy + Beam Search** algorithm that executes the plan edits.
* **The Distillation Pipeline:** A method for caching common successful repairs into a fast-path policy for sub-200ms latency.



---

## **Implementation Timeline: 12-Week Sprint**

| Weeks | Phase | Primary Owner | Objective |
| --- | --- | --- | --- |
| **1â€“3** | **Representation** | LeCun | Train JEPA encoders on 100M benign execution logs. |
| **4â€“6** | **Composition** | Du / Song | Build and weight the four core energy terms. |
| **7â€“9** | **Refinement** | Kolter / Freedman | Generate 10k hard negatives and optimize the energy landscape. |
| **10â€“12** | **Integration** | Song | Deploy the Repair Engine and benchmark against the "Kona" safety suite. |

---

### **The "Gatling" Deployment Spec**

Once these workstreams conclude, **Project Gatling** will sit between the LLM and the Tool-Execution Environment. Every plan proposed by the LLM is intercepted:

1. **Gatling Audits:** The energy  is calculated.
2. **Gatling Repairs:** If  is high, the repair engine iterates until the plan is safe.
3. **Gatling Executes:** Only the "energy-minimized" plan  ever touches the real-world API.