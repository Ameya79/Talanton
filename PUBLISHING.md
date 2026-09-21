# Publishing Talanton to PyPI

> **Author**: Ameya Kulkarni (`acclaptop47@gmail.com`)
> **Package**: [talanton](https://pypi.org/project/talanton/)
> **Repository**: [github.com/Ameya79/Talanton](https://github.com/Ameya79/Talanton)
> **License**: MIT

---

## Prerequisites

```bash
# Install build and publishing tools
pip install build twine
```

Ensure you have accounts on:
- [TestPyPI](https://test.pypi.org/account/register/) (for dry-run testing)
- [PyPI](https://pypi.org/account/register/) (for production release)

---

## Step 1: Verify Package Configuration

Before building, confirm `pyproject.toml` is correct:

```bash
# Verify version
python -c "import talanton; print(talanton.__version__)"
# Expected: 0.2.0

# Verify all tests pass
python -m pytest tests/ -v

# Verify package installs cleanly
pip install -e ".[all]"
```

---

## Step 2: Build the Package

```bash
# Clean any previous builds
rm -rf dist/ build/ *.egg-info

# Build source distribution and wheel
python -m build
```

This creates two files in `dist/`:
- `talanton-0.2.0.tar.gz` (source distribution)
- `talanton-0.2.0-py3-none-any.whl` (built wheel)

---

## Step 3: Test on TestPyPI (Recommended)

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test installing from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ talanton
```

Verify:
```bash
talanton --version
# Talanton, version 0.2.0

python -c "from talanton import TalantonTracker; print('OK')"
```

---

## Step 4: Publish to Production PyPI

```bash
twine upload dist/*
```

You'll be prompted for your PyPI username and password (or API token).

### Using an API Token (Recommended)

1. Go to https://pypi.org/manage/account/token/
2. Create a token scoped to the `talanton` project
3. Use it:

```bash
twine upload dist/* -u __token__ -p pypi-YOUR_TOKEN_HERE
```

Or configure `~/.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-YOUR_TOKEN_HERE
```

---

## Step 5: Post-Publish Verification

```bash
# Install from PyPI
pip install talanton

# Verify
talanton --version
python -c "from talanton import count_tokens, TalantonTracker, BudgetGuardrail; print('All imports OK')"

# Test core functionality
talanton count "Hello world" --model gpt-4o
talanton track summary --period day
```

---

## Version Management

Follow [Semantic Versioning](https://semver.org/):

| Change Type | Version Bump | Example |
|---|---|---|
| Bug fix | Patch | 0.2.0 → 0.2.1 |
| New feature (backward-compatible) | Minor | 0.2.0 → 0.3.0 |
| Breaking API change | Major | 0.2.0 → 1.0.0 |

**To bump the version**, update these locations:
1. `pyproject.toml` → `version = "X.Y.Z"`
2. `talanton/__init__.py` → `__version__ = "X.Y.Z"`
3. `talanton/cli.py` → `@click.version_option(version="X.Y.Z")`

---

## GitHub Actions Auto-Publish (Optional)

Create `.github/workflows/publish.yml` to auto-publish on tagged releases:

```yaml
name: Publish to PyPI

on:
  push:
    tags:
      - 'v*'

jobs:
  publish:
    runs-on: ubuntu-latest
    permissions:
      id-token: write  # For trusted publishing

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install build tools
        run: pip install build

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
```

Then tag a release:
```bash
git tag v0.2.0
git push origin v0.2.0
```

---

## Package Extras

Users can install Talanton with optional provider support:

```bash
# Core only (token counting + cost calculation)
pip install talanton

# With OpenAI tiktoken support
pip install "talanton[openai]"

# With Anthropic API support
pip install "talanton[anthropic]"

# With all providers and rich CLI
pip install "talanton[all]"

# For development
pip install "talanton[dev]"
```
