# CI/CD Pipeline for Project Gatling

## Overview

Project Gatling uses a **lightweight CI/CD pipeline** optimized for research workflows. The pipeline focuses on:
- ✅ Code quality and consistency
- ✅ Test coverage and correctness
- ✅ Performance regression detection
- ✅ Reproducibility across environments

## Pipeline Architecture

### GitHub Actions Workflows

#### 1. **Main CI Pipeline** (`.github/workflows/ci.yml`)

Runs on every push and pull request to `main` and `develop` branches.

**Jobs:**

**a) Test** (Multi-platform, Python 3.13)
- Platforms: Ubuntu, macOS
- Runs full test suite (excluding benchmarks)
- Generates coverage report (98% target)
- Uploads coverage to Codecov

**b) Lint & Type Check**
- Ruff linting (code quality)
- Ruff formatting check
- MyPy type checking
- Continues on error (warnings only)

**c) Benchmark** (Main branch only)
- Runs performance benchmarks
- Tracks performance over time
- Alerts on >20% degradation

### Local Development

#### Pre-commit Hooks (`.pre-commit-config.yaml`)

Automatically run before each commit:
- Trailing whitespace removal
- End-of-file fixer
- YAML/JSON/TOML validation
- Large file detection (>10MB)
- Merge conflict detection
- Debug statement detection
- **Ruff linting & formatting**
- **MyPy type checking**
- **Secret detection**

#### Installation

```bash
# Install pre-commit
uv pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Common Commands

### Using Makefile

```bash
# Show all available commands
make help

# Install dependencies
make install

# Install with dev tools and pre-commit
make install-dev

# Run tests
make test

# Run tests with coverage
make test-cov

# Run benchmarks
make benchmark

# Lint code
make lint

# Format code
make format

# Type check
make type-check

# Run all CI checks locally
make ci

# Clean cache files
make clean
```

### Using UV directly

```bash
# Run tests
uv run pytest test/ -v -m "not benchmark"

# Run with coverage
uv run pytest test/ --cov=source --cov-report=term-missing

# Run benchmarks only
uv run pytest test/ -m benchmark --benchmark-only

# Lint
uv run ruff check source/ test/

# Format
uv run ruff format source/ test/

# Type check
uv run mypy source/ --ignore-missing-imports
```

## CI Configuration

### Test Markers

Tests can be marked with pytest markers:
- `@pytest.mark.benchmark`: Performance benchmarks (slow)
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.slow`: Slow tests

Run specific markers:
```bash
# Run only benchmarks
uv run pytest -m benchmark

# Exclude benchmarks
uv run pytest -m "not benchmark"

# Run integration tests
uv run pytest -m integration
```

### Coverage Requirements

- **Target**: 90% code coverage
- **Current**: 98% (GovernanceEncoder)
- **Enforcement**: Reported but not enforced (research flexibility)

### Performance Benchmarks

Benchmarks track:
- Inference latency (mean, min, max)
- Memory usage
- Throughput

**Alert triggers:**
- >20% performance degradation
- >50% memory increase

## CI Best Practices

### For Commits

1. **Run tests locally first**:
   ```bash
   make ci
   ```

2. **Use pre-commit hooks**:
   - Auto-formats code
   - Catches issues before commit
   - Prevents secrets from being committed

3. **Write meaningful commit messages**:
   ```
   feat: add execution encoder for JEPA architecture

   - Implements ExecutionEncoder (LSA-002)
   - Adds sparse attention for plan graphs
   - Test coverage: 95%
   ```

### For Pull Requests

1. **Ensure CI passes** before requesting review
2. **Update tests** for new functionality
3. **Document** significant changes in CHANGELOG.md
4. **Benchmark** performance-critical changes

### For Research Code

CI is **flexible for research**:
- Linting errors are **warnings** (not blockers)
- Type checking is **advisory** (not strict)
- Benchmarks alert but **don't fail** builds

**Philosophy**: CI assists research, it doesn't block it.

## Troubleshooting

### CI Fails on Import Errors

**Problem**: Module not found in CI
```
ModuleNotFoundError: No module named 'source'
```

**Solution**: Ensure `pyproject.toml` has `pythonpath = ["."]` in `[tool.pytest.ini_options]`

### Pre-commit Hooks Slow

**Problem**: Hooks take too long
**Solution**: Skip slow hooks for quick commits:
```bash
SKIP=mypy git commit -m "Quick fix"
```

### Benchmark Alerts

**Problem**: Performance degradation alert
**Solution**:
1. Check if change was intentional
2. Optimize if needed
3. Update baseline if acceptable

## Future Enhancements

### Phase 2: Research CI (When Training Starts)

Planned additions:
- **Experiment Tracking**: MLflow/Weights & Biases integration
- **Model Versioning**: Hugging Face Hub integration
- **Dataset Validation**: Automated data quality checks
- **Training Pipelines**: Distributed training on GPU runners
- **Artifact Management**: Model checkpoint storage

### Phase 3: Deployment CI

When moving to production:
- **Docker builds**: Containerized deployment
- **Security scanning**: Snyk/Trivy for vulnerabilities
- **Integration tests**: End-to-end system tests
- **Staging deployment**: Automatic staging deploys
- **Load testing**: Performance under load

## Resources

- **GitHub Actions Docs**: https://docs.github.com/en/actions
- **UV Docs**: https://docs.astral.sh/uv/
- **Ruff Docs**: https://docs.astral.sh/ruff/
- **Pre-commit Docs**: https://pre-commit.com/
- **Pytest Docs**: https://docs.pytest.org/

## Questions?

- Check `Makefile` for available commands
- Review test files for examples
- See `.github/workflows/ci.yml` for CI configuration
