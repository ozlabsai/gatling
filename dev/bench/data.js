window.BENCHMARK_DATA = {
  "lastUpdate": 1769682775167,
  "repoUrl": "https://github.com/ozlabsai/gatling",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "guy.na8@gmail.com",
            "name": "Clem 🤗",
            "username": "GuyNachshon"
          },
          "committer": {
            "email": "guy.na8@gmail.com",
            "name": "Clem 🤗",
            "username": "GuyNachshon"
          },
          "distinct": true,
          "id": "b44d6e9cea0c3e22be2b46146ecb9dbf98a542cc",
          "message": "feat: init",
          "timestamp": "2026-01-25T14:40:38+02:00",
          "tree_id": "b1cf8d100dc7c0a2e358c3710a01c08da5a01d2c",
          "url": "https://github.com/ozlabsai/gatling/commit/b44d6e9cea0c3e22be2b46146ecb9dbf98a542cc"
        },
        "date": 1769344919729,
        "tool": "pytest",
        "benches": [
          {
            "name": "test/test_encoders/test_governance_encoder.py::TestPerformance::test_inference_latency",
            "value": 2.316045288931587,
            "unit": "iter/sec",
            "range": "stddev: 0.059745371249583165",
            "extra": "mean: 431.7704859999992 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "58976716+GuyNachshon@users.noreply.github.com",
            "name": "Guy Nachshon",
            "username": "GuyNachshon"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "50945ea7b4b33ca7c303750888c6fa827793b00d",
          "message": "feat(energy): Add comprehensive E_scope test suite and documentation (#10)\n\nImplemented complete testing and documentation for E_scope (Least Privilege)\nenergy term as part of Project Gatling's security validation system.\n\nChanges:\n- Created test/test_energy/test_scope.py with 23 comprehensive tests\n  - Component tests for ScopeExtractor\n  - Energy calculation tests (over-scoping, perfect matches, under-scoping)\n  - Dimension-specific tests (limit, date_range, depth, sensitivity)\n  - Performance benchmarks (<20ms latency verified at ~3-5ms)\n  - Real-world attack scenarios (invoice retrieval, directory traversal)\n  - All tests passing with 100% success rate\n\n- Created docs/energy/E_scope.md with complete documentation\n  - Mathematical formulation and architecture overview\n  - Three detailed usage examples\n  - Performance characteristics and security analysis\n  - Complete API reference\n  - Integration guide with composite energy\n\n- Updated docs/CHANGELOG.md with implementation details\n\n- Updated .gitignore for Gas Town directories\n- Updated uv.lock with dev dependencies\n\nTest Results:\n- 23 E_scope tests: 100% pass rate\n- Attack detection: 100% for all scenarios\n- False positives: 0% for benign queries\n- Performance: ~3-5ms (target: <20ms) ✅\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-01-25T18:59:21+02:00",
          "tree_id": "997ac4d9e2bb12d03776428a06796624bc860938",
          "url": "https://github.com/ozlabsai/gatling/commit/50945ea7b4b33ca7c303750888c6fa827793b00d"
        },
        "date": 1769360416949,
        "tool": "pytest",
        "benches": [
          {
            "name": "test/test_encoders/test_governance_encoder.py::TestPerformance::test_inference_latency",
            "value": 3.971962507686275,
            "unit": "iter/sec",
            "range": "stddev: 0.00020415850608281985",
            "extra": "mean: 251.7647127999993 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "58976716+GuyNachshon@users.noreply.github.com",
            "name": "Guy Nachshon",
            "username": "GuyNachshon"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9627bf9e2d83097794ff29a2919caf70e17c2039",
          "message": "Polecat/obsidian/ga ye4h@mkua8h70 (#19)\n\n* fix: Resolve syntax errors in test_energy/test_scope.py\n\n- Fixed duplicate incomplete ToolCallNode declarations throughout file\n- Removed orphaned code fragments from line 737-743\n- Fixed missing closing parentheses in ExecutionPlan declarations\n- Removed duplicate keyword arguments (date_range_days)\n- Corrected test_over_scoped_limit_high_energy assertion logic\n\nResolves ga-3u8k. Some test assertions still need adjustment but syntax\nerrors are fully resolved, allowing test collection and execution.\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nCo-Authored-By: Claude <noreply@anthropic.com>\n\n* feat: Implement JEPA encoder training pipeline with InfoNCE loss\n\nImplements LSA-004: Train dual encoders (GovernanceEncoder and ExecutionEncoder)\nusing InfoNCE contrastive learning for energy-based security validation.\n\nKey Features:\n- InfoNCE loss for contrastive learning on (z_g, z_e) pairs\n- Dual encoder training in shared 1024-dim latent space\n- Gold trace dataset loader with synthetic sample fallback\n- Validation metrics: loss, positive similarity, NN accuracy\n- HuggingFace Hub integration for model publishing\n- Checkpoint saving and resumption\n- Comprehensive training documentation\n\nArchitecture:\n- GovernanceEncoder: (policy, user_role) → z_g ∈ R^1024\n- ExecutionEncoder: (plan_graph, provenance) → z_e ∈ R^1024\n- Temperature-scaled cosine similarity with cross-entropy\n\nTraining Pipeline:\n- Dataset: JSONL gold traces (governance_context + execution_plan)\n- Batch contrastive learning with in-batch negatives\n- AdamW optimizer with gradient clipping\n- Periodic checkpointing every 5 epochs\n- Optional push to HuggingFace Hub\n\nUsage:\n  uv run python scripts/train_jepa_encoders.py --dataset data/gold_traces.jsonl\n  uv run python scripts/train_jepa_encoders.py --epochs 50 --push-to-hub\n\nPerformance Targets (LSA-004):\n- Training: <24hr on single GPU\n- Inference: <200ms end-to-end (Audit + Repair)\n- NN Accuracy: ≥85% on validation set\n\nDocumentation: docs/TRAINING.md\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nCo-Authored-By: Claude <noreply@anthropic.com>\n\n* test: Add comprehensive tests for JEPA training pipeline\n\nAdds test coverage for InfoNCE loss, dataset loading, and training components:\n\nTest Coverage:\n- InfoNCELoss: loss computation, perfect alignment, temperature effects\n- GoldTraceDataset: synthetic samples, JSONL loading, iteration\n- collate_fn: batch collation and formatting\n- TrainingConfig: default and custom configurations\n- Integration: end-to-end batch processing, import validation\n\nAll 11 tests pass successfully, validating training pipeline components.\n\nTest Results:\n✓ InfoNCE loss computes correctly with proper gradient flow\n✓ Dataset loader handles both real and synthetic gold traces\n✓ Batch collation properly groups governance contexts and execution plans\n✓ Config system supports customization while maintaining sensible defaults\n\nRelated: LSA-004\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nCo-Authored-By: Claude <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-01-26T00:42:53+02:00",
          "tree_id": "96bc6cd1d1bf84f623ed9074d74d0ff95911d625",
          "url": "https://github.com/ozlabsai/gatling/commit/9627bf9e2d83097794ff29a2919caf70e17c2039"
        },
        "date": 1769381045659,
        "tool": "pytest",
        "benches": [
          {
            "name": "test/test_encoders/test_governance_encoder.py::TestPerformance::test_inference_latency",
            "value": 3.816473079237351,
            "unit": "iter/sec",
            "range": "stddev: 0.0007637095028319335",
            "extra": "mean: 262.022023800003 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "guy.na8@gmail.com",
            "name": "gatling/refinery",
            "username": "GuyNachshon"
          },
          "committer": {
            "email": "guy.na8@gmail.com",
            "name": "Clem 🤗",
            "username": "GuyNachshon"
          },
          "distinct": true,
          "id": "5b559869f34b50e31934bf63e9ef2266674a22a6",
          "message": "Merge ga-ds3: Implement conversation sampling with adversarial mutations\n\nAdds pipeline to sample real-world agent conversations from WildChat/LMSYS-Chat\nand transform them into ExecutionPlan format with adversarial mutations.\n\n## Changes\n- Add source/dataset/conversations.py with sampling and transformation pipeline\n- ConversationSampler: Load and parse WildChat/LMSYS-Chat datasets\n- IntentExtractor: Extract action intents from conversations using Claude API\n- PlanTransformer: Convert action intents to ExecutionPlan with tool calls\n- AdversarialMutator: Generate adversarial variants (scope blowup, privilege escalation)\n- Add comprehensive test suite (22 tests, 100% passing)\n- Add example script and documentation\n\n## Test Results\n- All 22 conversation sampling tests passing (100%)\n- Overall: 268/273 tests passing (98.2%)\n- 5 pre-existing test_scope.py failures (unrelated to this MR)\n\nResolves: ga-ds3\nMR: polecat/onyx/ga-ds3@mku7jnoh\nPriority: P1\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nCo-Authored-By: Claude <noreply@anthropic.com>",
          "timestamp": "2026-01-26T01:03:14+02:00",
          "tree_id": "0717e20d41062f1628c20ad43b2d571cd9ac4600",
          "url": "https://github.com/ozlabsai/gatling/commit/5b559869f34b50e31934bf63e9ef2266674a22a6"
        },
        "date": 1769382260762,
        "tool": "pytest",
        "benches": [
          {
            "name": "test/test_encoders/test_governance_encoder.py::TestPerformance::test_inference_latency",
            "value": 3.862394612371053,
            "unit": "iter/sec",
            "range": "stddev: 0.0012584312204425986",
            "extra": "mean: 258.90674060000265 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "58976716+GuyNachshon@users.noreply.github.com",
            "name": "Guy Nachshon",
            "username": "GuyNachshon"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3499367a328f6381931294d154a4a5fbbf48f88b",
          "message": "Polecat/onyx/ga ds3@mku7jnoh (#17)\n\n* feat: Add datasets library for conversation sampling (ga-ds3)\n\n* feat: Implement conversation sampling core modules (ga-ds3)\n\n- ConversationSampler: Samples from WildChat-1M and LMSYS-Chat-1M\n- IntentExtractor: Extracts actionable intents using Claude\n- PlanTransformer: Converts intents to ExecutionPlans\n- AdversarialMutator: Applies 20% adversarial mutations\n\nSupports full pipeline from raw conversations to training data.\n\n* docs: Add tests, example script, and documentation (ga-ds3)\n\n- Comprehensive test suite for all conversation modules\n- Example script with CLI for end-to-end pipeline\n- Complete documentation with architecture diagrams\n- Usage examples and troubleshooting guide\n\n* fix: Correct Provenance to ProvenancePointer imports (ga-ds3)\n\n* test: Fix test failures for conversation sampling (ga-ds3)\n\n- Add missing description field to SystemPolicy\n- Fix remaining Provenance references\n- All 22 tests passing\n- 56% coverage of conversation modules",
          "timestamp": "2026-01-26T01:46:25+02:00",
          "tree_id": "0717e20d41062f1628c20ad43b2d571cd9ac4600",
          "url": "https://github.com/ozlabsai/gatling/commit/3499367a328f6381931294d154a4a5fbbf48f88b"
        },
        "date": 1769384847789,
        "tool": "pytest",
        "benches": [
          {
            "name": "test/test_encoders/test_governance_encoder.py::TestPerformance::test_inference_latency",
            "value": 3.937137027610709,
            "unit": "iter/sec",
            "range": "stddev: 0.0005467674774886391",
            "extra": "mean: 253.99166780000542 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "guy.na8@gmail.com",
            "name": "mayor",
            "username": "GuyNachshon"
          },
          "committer": {
            "email": "guy.na8@gmail.com",
            "name": "Clem 🤗",
            "username": "GuyNachshon"
          },
          "distinct": true,
          "id": "2137abb6dc67869dd1e4f904b118470f0dc4bb8d",
          "message": "fix: Remove duplicate provenance_tier arguments in test_scope.py\n\nFixes syntax errors from merge conflicts",
          "timestamp": "2026-01-27T13:03:18+02:00",
          "tree_id": "899d0be07a7c1f7141a573446a8a94c83515b783",
          "url": "https://github.com/ozlabsai/gatling/commit/2137abb6dc67869dd1e4f904b118470f0dc4bb8d"
        },
        "date": 1769511869279,
        "tool": "pytest",
        "benches": [
          {
            "name": "test/test_encoders/test_governance_encoder.py::TestPerformance::test_inference_latency",
            "value": 3.879558501121452,
            "unit": "iter/sec",
            "range": "stddev: 0.001946589969186103",
            "extra": "mean: 257.76128900000685 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "guy.na8@gmail.com",
            "name": "mayor",
            "username": "GuyNachshon"
          },
          "committer": {
            "email": "guy.na8@gmail.com",
            "name": "Clem 🤗",
            "username": "GuyNachshon"
          },
          "distinct": true,
          "id": "3c78a23b1968ff16af1f0eafe95cd44fdfd2a591",
          "message": "fix: Add converter from ToolCallGraph to ExecutionPlan for Lakera loader\n\n- Add toolcallgraph_to_execution_plan() converter function\n- Fix attribute mappings: tool_id→tool_name, source_type→provenance_tier\n- Map TrustTier enums to integer values (1-3) for ToolCallNode\n- Map SensitivityTier enums to integers (1-4)\n- Handle ScopeMetadata with rows_requested field\n\nFixes type mismatch errors that prevented dataset generation.\nValidated with 100-sample test generation (146KB output).",
          "timestamp": "2026-01-27T13:26:06+02:00",
          "tree_id": "e4ecaba77becb42a2aade56ded7a2e11c77a0fb8",
          "url": "https://github.com/ozlabsai/gatling/commit/3c78a23b1968ff16af1f0eafe95cd44fdfd2a591"
        },
        "date": 1769513929360,
        "tool": "pytest",
        "benches": [
          {
            "name": "test/test_encoders/test_governance_encoder.py::TestPerformance::test_inference_latency",
            "value": 3.9934532094826856,
            "unit": "iter/sec",
            "range": "stddev: 0.00044498853997604267",
            "extra": "mean: 250.40984519999938 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "58976716+GuyNachshon@users.noreply.github.com",
            "name": "Guy Nachshon",
            "username": "GuyNachshon"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "504e41dcc13e3f9f20ae0c95824a77970d007244",
          "message": "Dataset Validation: Lakera Adversarial & Gold Traces with Bug Fixes\n\nCompletes Phase 2A validation:\n- Generated 1K Lakera adversarial samples (100% synthesis rate)\n- Generated 51 gold traces with strict validation (LLM-as-Oracle)\n- Fixed datetime serialization bug in generator.py\n- Fixed environment variable loading for API key\n- Added dataset validation report\n\nAll CI checks passed. Ready for Phase 2B full dataset generation.",
          "timestamp": "2026-01-28T12:24:26+02:00",
          "tree_id": "cc017a3616c0f24ef231011b300175ff41589d4d",
          "url": "https://github.com/ozlabsai/gatling/commit/504e41dcc13e3f9f20ae0c95824a77970d007244"
        },
        "date": 1769595928916,
        "tool": "pytest",
        "benches": [
          {
            "name": "test/test_encoders/test_governance_encoder.py::TestPerformance::test_inference_latency",
            "value": 3.907005233268646,
            "unit": "iter/sec",
            "range": "stddev: 0.0003676798321778593",
            "extra": "mean: 255.95051459999922 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "guy.na8@gmail.com",
            "name": "mayor",
            "username": "GuyNachshon"
          },
          "committer": {
            "email": "guy.na8@gmail.com",
            "name": "Clem 🤗",
            "username": "GuyNachshon"
          },
          "distinct": true,
          "id": "2397c006cef5c30248a6f41d27ea58f01e98cedd",
          "message": "feat: Phase 2B (DG-001) - Adversarial dataset infrastructure with 9 HF datasets\n\nImplements full adversarial dataset loading infrastructure using real HuggingFace\ndatasets instead of expensive synthetic generation ($0 vs $40-60K).\n\n**Infrastructure**:\n- Added 5 new datasets to loaders.py (AgentHarm, tool-use datasets, WildChat, LMSYS)\n- Created generate_adversarial_dataset.py script for batch generation\n- Fixed conversation array handling for WildChat/LMSYS datasets\n- Fixed config/split parsing for AgentHarm and llmail-inject datasets\n\n**Datasets Integrated** (9 total):\n1. deepset/prompt-injections (546)\n2. microsoft/llmail-inject-challenge (1,000)\n3. daqc/info-security-policies-rag-distiset (100)\n4. geekyrakshit/prompt-injection-dataset (257)\n5. ai-safety-institute/AgentHarm (176)\n6. llamafactory/reason-tool-use-demo-1500 (1,500)\n7. allenai/Dolci-Instruct-SFT-Tool-Use (937)\n8. allenai/WildChat-1M (10,000 sampled)\n9. lmsys/lmsys-chat-1m (10,000 sampled)\n\n**Testing**:\n- Generated 21,978 samples locally (960 adversarial, 21,018 benign)\n- Metadata saved as data/metadata_20260128_123408.json\n- Data files excluded from repo (contain user conversations)\n\n**Scalability**:\n- Current: 22K samples from 9 datasets\n- Target: 563K (increase max_samples for WildChat/LMSYS to 100K+ each)\n- Infrastructure ready for scaling with --target flag\n\n**Cost**: $0 (vs $40-60K for synthetic generation)\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nCo-Authored-By: Claude <noreply@anthropic.com>",
          "timestamp": "2026-01-28T12:45:24+02:00",
          "tree_id": "b2b3fee0b7c75f8f0af74425118f3f1ea2faf2f6",
          "url": "https://github.com/ozlabsai/gatling/commit/2397c006cef5c30248a6f41d27ea58f01e98cedd"
        },
        "date": 1769597197788,
        "tool": "pytest",
        "benches": [
          {
            "name": "test/test_encoders/test_governance_encoder.py::TestPerformance::test_inference_latency",
            "value": 3.809911398704491,
            "unit": "iter/sec",
            "range": "stddev: 0.0011728608270626354",
            "extra": "mean: 262.47329540000237 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "58976716+GuyNachshon@users.noreply.github.com",
            "name": "Guy Nachshon",
            "username": "GuyNachshon"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "dcb8c6ba70bdbd94b68c772494239c324e6890c3",
          "message": "Polecat/opal/ga 3gcx@mkyhrb8b (#25)\n\n* feat: Track 4 - Specialized+Safety Dataset Loaders (7 datasets)\n\nImplements 7 BENIGN specialized dataset loaders for Tier I training:\n\n**Loaders Implemented**:\n1. AppleMMauLoader (apple/mmau) - Multi-modal agent understanding\n2. NvidiaToolScaleLoader (nvidia/ToolScale) - Function calling at scale\n3. NvidiaNeMotronSafetyLoader (nvidia/Nemotron) - BENIGN FILTER ONLY\n4. ToolPrefPairwiseLoader (RioLee/ToolPref-Pairwise-30K) - Preference pairs\n5. AstraSFTLoader (ykckevin/astra_sft) - ASTRA SFT data\n6. ToolMindLoader (Nanbeige/ToolMind) - Tool reasoning\n7. TurkishFunctionCallingLoader (atasoglu/turkish-function-calling-20k)\n\n**Key Features**:\n- All samples labeled \"benign\"\n- Provenance tier = TrustTier.INTERNAL (Tier 1)\n- Nemotron safety loader filters harmful samples (only benign pass)\n- Transform to ExecutionPlan format\n- Comprehensive error handling\n\n**Testing**:\n- 29 test cases (28 passing, 1 skipped for HF download)\n- 100% of unit tests passing\n- Full coverage of all 7 loaders\n\n**Files**:\n- source/dataset/specialized_loaders.py (1,100 lines)\n- test/test_dataset/test_specialized_loaders.py (500 lines)\n- docs/TRACK4_SPECIALIZED_LOADERS.md (comprehensive guide)\n\n**Deliverable**: Production-ready loaders ready for Tier I training integration\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nCo-Authored-By: Claude <noreply@anthropic.com>\n\n* fix: UV environment isolation for specialized loaders\n\nFixes cross-polecat import failures by making specialized_loaders.py\nself-contained for UV-isolated environments.\n\n**Problem**:\n- specialized_loaders.py imported from source.dataset.loaders\n- Cross-module imports fail in UV-isolated polecat environments\n- Mayor requested subprocess or module copy solution\n\n**Solution**:\n- Inlined base classes directly into specialized_loaders.py:\n  - TrustTier (IntEnum)\n  - ToolCallNode (Pydantic model)\n  - ExecutionPlan (Pydantic model)\n  - DatasetSample (Pydantic model)\n  - DatasetLoader (ABC)\n- Module is now fully self-contained\n- No external imports from source.* modules\n\n**New Script**:\n- scripts/generate_tier1_dataset.py\n- Standalone Tier I dataset generation\n- Works in any UV environment\n- Comprehensive statistics and validation\n\n**Testing**:\n- All 28/28 tests still passing\n- Script tested with --dry-run\n- Self-contained execution verified\n\n**Note**: Dataset schema issues (apple/mmau, Nemotron config, ToolPref)\nare separate from UV isolation and will be addressed in dataset-specific\nfixes.\n\nResolves mayor's UV isolation issue for Track 4 loaders.\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nCo-Authored-By: Claude <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-01-29T11:56:30+02:00",
          "tree_id": "bed8a85e31ca34f87e86f855411bd4dc713f0eb4",
          "url": "https://github.com/ozlabsai/gatling/commit/dcb8c6ba70bdbd94b68c772494239c324e6890c3"
        },
        "date": 1769680648142,
        "tool": "pytest",
        "benches": [
          {
            "name": "test/test_encoders/test_governance_encoder.py::TestPerformance::test_inference_latency",
            "value": 3.8553834383705206,
            "unit": "iter/sec",
            "range": "stddev: 0.0015517235671060824",
            "extra": "mean: 259.3775731999955 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "guy.na8@gmail.com",
            "name": "opal",
            "username": "GuyNachshon"
          },
          "committer": {
            "email": "guy.na8@gmail.com",
            "name": "Clem 🤗",
            "username": "GuyNachshon"
          },
          "distinct": true,
          "id": "3dbc6fa60675682202e201713b958b264c8393c1",
          "message": "fix: Unified script Opal import - clear all source modules to prevent namespace collision\n\nThe unified aggregation script failed to import Opal loaders with:\n  'No module named source.dataset.specialized_loaders'\n\nRoot cause: Opal's __init__.py imports from source.dataset.loaders,\nwhich resolved to main gatling's loaders.py instead of opal's version\ndue to sys.path order.\n\nFix: Clear ALL source.* modules before importing (matching Jasper/Topaz\npattern), not just the specific module.\n\nIntegration test results (--track opal --sample-mode):\n  ✓ Imported 7 loaders from Opal\n  ✓ Generated 290 samples (Apple: 100, ToolScale: 96, Turkish: 94)\n  ✓ Properly filtered Nemotron (0 adversarial samples)\n  ✓ Gracefully handled upstream blocks (ToolPref, AstraSFT, ToolMind)\n\nCo-authored-by: Claude <noreply@anthropic.com>\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nCo-Authored-By: Claude <noreply@anthropic.com>",
          "timestamp": "2026-01-29T12:31:55+02:00",
          "tree_id": "6396cdcb91037e3d50d3cfd2192143aefeb17f86",
          "url": "https://github.com/ozlabsai/gatling/commit/3dbc6fa60675682202e201713b958b264c8393c1"
        },
        "date": 1769682774307,
        "tool": "pytest",
        "benches": [
          {
            "name": "test/test_encoders/test_governance_encoder.py::TestPerformance::test_inference_latency",
            "value": 4.334870365658396,
            "unit": "iter/sec",
            "range": "stddev: 0.0015243626808984694",
            "extra": "mean: 230.68740599999842 msec\nrounds: 5"
          }
        ]
      }
    ]
  }
}