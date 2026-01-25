window.BENCHMARK_DATA = {
  "lastUpdate": 1769360417693,
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
      }
    ]
  }
}