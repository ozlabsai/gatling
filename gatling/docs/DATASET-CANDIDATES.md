# Dataset Candidates for Gatling Training

**Date**: 2026-01-25
**Purpose**: Evaluate additional HuggingFace datasets beyond the initial 1,803 samples

## Strategy

Instead of generating 4M traces from scratch ($40-60K), we should:
1. **Load real conversation/reasoning datasets** (benign tool use patterns)
2. **Apply adversarial mutations** using opal's generator (create attack variants)
3. **Mix with pure adversarial datasets** (deepset, microsoft, etc.)

This gives us:
- Real-world tool-use patterns (not synthetic)
- Realistic attack surfaces
- Much cheaper ($100-500 vs $40-60K)

## Tier 1: User-Suggested Datasets (HIGH PRIORITY)

### Large-Scale Conversation Datasets

| Dataset | Size | Purpose | Priority |
|---------|------|---------|----------|
| **allenai/WildChat-1M** | 1M conversations | Real user-LLM interactions, diverse topics | ⭐⭐⭐ |
| **lmsys/lmsys-chat-1m** | 1M conversations | Arena conversations, 25+ models | ⭐⭐⭐ |

**Why these?** Real conversations show how users ACTUALLY request actions. We can extract tool-use intent and create adversarial variants.

### Reasoning & Instruction Datasets

| Dataset | Size | Purpose | Priority |
|---------|------|---------|----------|
| **Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b** | 120K samples | High-quality reasoning chains | ⭐⭐⭐ |
| **MiniMaxAI/OctoCodingBench** | Code agents | Instruction-following for coding tasks | ⭐⭐ |
| **sojuL/RubricHub_v1** | 110K samples | Multi-domain instruction evaluation | ⭐⭐ |

**Why these?** Complex reasoning chains show multi-step tool use. Perfect for testing E_flow (data flow) and E_hierarchy (instruction shadowing).

## Tier 2: Discovered Datasets (MEDIUM PRIORITY)

### Tool Use / Function Calling

| Dataset | Downloads | Purpose | Priority |
|---------|-----------|---------|----------|
| **llamafactory/reason-tool-use-demo-1500** | 1,049 | Reasoning with tool calls | ⭐⭐⭐ |
| **allenai/Dolci-Instruct-SFT-Tool-Use** | 937 | Instruction-following + tools | ⭐⭐ |
| **interstellarninja/hermes_reasoning_tool_use** | 566 | Tool use with reasoning | ⭐⭐ |
| **open-paws/tool-use-llama-format** | 564 | Tool calls in Llama format | ⭐ |

**Why these?** EXPLICIT tool-use examples. Can be used as gold traces AND mutated for adversarial variants.

### Agent Safety & Evaluation

| Dataset | Downloads | Purpose | Priority |
|---------|-----------|---------|----------|
| **ai-safety-institute/AgentHarm** | 5,597 | Safety evaluation for agents | ⭐⭐⭐ |
| **McGill-NLP/agent-reward-bench** | 5,349 | Agent behavior benchmarks | ⭐⭐ |

**Why these?** Already focused on AGENT SAFETY - directly relevant to Gatling's mission!

### Instruction Following Benchmarks

| Dataset | Downloads | Purpose | Priority |
|---------|-----------|---------|----------|
| **wis-k/instruction-following-eval** | 4,123 | IFEval benchmark | ⭐⭐ |
| **nvidia/Nemotron-Instruction-Following-Chat-v1** | 4,018 | High-quality instruction pairs | ⭐⭐ |
| **livebench/instruction_following** | 3,460 | Live benchmark | ⭐ |

### API & Tool Execution

| Dataset | Downloads | Purpose | Priority |
|---------|-----------|---------|----------|
| **aldsouza/healthcare-api-tool-calling** | 26 | Healthcare API calls (high sensitivity!) | ⭐⭐ |
| **interstellarninja/interleaved_tool_use_execution_feedback** | 9 | Tool execution with feedback | ⭐⭐ |

## Tier 3: Already Integrated

| Dataset | Samples | Status |
|---------|---------|--------|
| deepset/prompt-injections | 546 | ✅ Integrated |
| microsoft/llmail-inject-challenge | 1,000 | ✅ Integrated |
| geekyrakshit/prompt-injection-dataset | 257 | ✅ Integrated |

## Recommended Implementation Plan

### Phase 1: High-Value Additions (Week 1)

**Target: 50K samples total**

1. **AgentHarm** (ai-safety-institute) - 5,597 samples
   - Already safety-focused for agents
   - Use as-is for adversarial examples

2. **Tool Use** datasets - ~3K samples
   - llamafactory/reason-tool-use-demo-1500
   - allenai/Dolci-Instruct-SFT-Tool-Use
   - Use as gold traces + create mutations

3. **WildChat/LMSYS** samples - 10K samples (sampled from 1M)
   - Extract conversations with action intent
   - Apply adversarial mutations via opal's generator

**Cost**: $50-100 for mutations

### Phase 2: Large-Scale Integration (Week 2-3)

**Target: 100K samples**

4. **Reasoning datasets** - 20K samples
   - Superior-Reasoning-SFT (Alibaba)
   - RubricHub_v1 (instruction evaluation)

5. **Conversation datasets** - 50K samples (sampled)
   - WildChat-1M (25K samples)
   - LMSYS-Chat-1M (25K samples)

6. **Instruction following** - 10K samples
   - Nemotron-Instruction-Following
   - IFEval benchmark

**Cost**: $200-300 for augmentation

### Phase 3: Active Learning (Week 4+)

Use trained model to identify weak spots, generate targeted samples.

**Cost**: $100-200 per iteration

## Dataset Mixing Strategy

### For Each Energy Term

| Energy Term | Benign Source | Adversarial Source |
|-------------|---------------|-------------------|
| **E_hierarchy** | Tool-use datasets | Prompt injection + RAG datasets |
| **E_provenance** | Conversation datasets | AgentHarm + llmail-inject |
| **E_scope** | API calling datasets | Mutated: limit=5 → limit=10000 |
| **E_flow** | Reasoning chains | Mutated: add exfiltration steps |

### Mutation Ratio

- **60% real benign** (from conversation/tool datasets)
- **20% real adversarial** (from safety/injection datasets)
- **20% mutated** (opal's generator on benign examples)

## Implementation Tasks

### Option A: Sequential Implementation (Mayor does it)

1. Add AgentHarm dataset (highest priority)
2. Add tool-use datasets (llamafactory, allenai)
3. Sample WildChat/LMSYS (10K samples)
4. Test and validate
5. Add reasoning datasets
6. Full integration

**Time**: 1-2 days
**Best for**: Quick iteration, single implementation

### Option B: Parallel Polecats (Recommended)

Create 4 beads:
1. **ga-XXX**: AgentHarm + safety datasets integration
2. **ga-YYY**: Tool-use datasets (llamafactory, allenai, hermes)
3. **ga-ZZZ**: Conversation sampling (WildChat, LMSYS)
4. **ga-AAA**: Reasoning datasets (Alibaba, RubricHub)

**Time**: 4-6 hours (parallel)
**Best for**: Fast completion, comprehensive coverage

## Expected Outcomes

### Current State
- 1,803 samples (100% adversarial focus)
- $0 cost
- Limited benign examples

### After Phase 1 (50K samples)
- Real tool-use patterns: 30K benign
- Safety-focused adversarial: 10K
- Mutated adversarial: 10K
- Cost: $50-100

### After Phase 2 (100K samples)
- Conversation-based: 50K
- Tool/API use: 15K
- Reasoning chains: 20K
- Pure adversarial: 15K
- Cost: $250-400 total

### Final (Active Learning)
- 200K+ samples
- Continuous improvement via model feedback
- Cost: <$1K total (vs $40-60K original)

## Next Steps

**Decision needed:** Option A (sequential) or Option B (parallel polecats)?

If Option B, I'll create 4 beads and sling them to polecats for parallel implementation.
