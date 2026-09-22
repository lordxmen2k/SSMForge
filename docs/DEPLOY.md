# Deploying ssmforge to PyPI

This document explains how to publish `ssmforge` v0.1.0 to PyPI.

## What you'll get

After running through the steps below, anyone can install ssmforge with:

```bash
pip install ssmforge
ssmforge arch Qwen/Qwen2-1.5B-Instruct
```

## Why we ship to PyPI

`pip install ssmforge` is the standard install path. PyPI gives the package:

- A canonical URL (https://pypi.org/project/ssmforge/)
- Versioning with installable constraints (`pip install ssmforge>=0.1.0`)
- Signature verification
- Discoverability via the PyPI search

## Prerequisites

- A PyPI account (https://pypi.org/account/register/)
- A PyPI API token (https://pypi.org/manage/account/token/) — scope to this project
- `twine` installed: `pip install twine build`
- This repo's `dist/` directory (already built; see below)

## Step 1: Verify the build

The artifacts already exist at `dist/`:

```bash
ls -lh dist/
# ssmforge-0.1.0-py3-none-any.whl   ~25 KB
# ssmforge-0.1.0.tar.gz             ~33 KB
```

Both passed `twine check`:

```bash
twine check dist/*
# PASSED for both
```

If you need to rebuild (e.g. after edits):

```bash
# From the repo root
python -m pip install --upgrade build  # one-time
python -m build --no-isolation
```

`--no-isolation` uses your existing venv's setuptools/wheel — faster and doesn't
require network access during build.

## Step 2: Sanity-check the package contents

```bash
# List files inside the wheel
python -c "import zipfile; z=zipfile.ZipFile('dist/ssmforge-0.1.0-py3-none-any.whl'); print('\n'.join(sorted(z.namelist())))"

# Show the metadata
python -c "import zipfile; z=zipfile.ZipFile('dist/ssmforge-0.1.0-py3-none-any.whl'); print(z.read('ssmforge-0.1.0.dist-info/METADATA').decode())"
```

Verify:
- ✅ Author is `SSMForge Contributors` (not a platform name)
- ✅ License is `Apache License`
- ✅ Dependencies: `transformers>=4.45`, `huggingface_hub>=0.24`
- ✅ Entry point: `ssmforge = ssmforge.cli:main`
- ✅ Has all source files: `ssmforge/cli.py`, `ssmforge/analyze/*`

## Step 3: Upload to PyPI

```bash
# Test the upload (optional but recommended for first time)
twine upload --repository testpypi dist/ssmforge-0.1.0*

# Then upload to production
twine upload dist/ssmforge-0.1.0*
```

When prompted for credentials, paste the API token (not your password).

If you'd rather use the API token non-interactively, set it in `~/.pypirc`:

```ini
[distutils]
index-servers = pypi

[pypi]
username = __token__
password = pypi-XXXXXXXXXXXXXXXXXXXXXX
```

Then `twine upload` will use it automatically.

## Step 4: Verify the install

In a fresh Python environment:

```bash
python -m venv /tmp/ssmforge-fresh
source /tmp/ssmforge-fresh/bin/activate   # or: /tmp/ssmforge-fresh\Scripts\activate on Windows
pip install ssmforge
ssmforge doctor
ssmforge arch hf-internal-testing/tiny-random-LlamaForCausalLM --quiet
```

You should see `ssmforge_version: 0.1.0` and a valid JSON report.

## Step 5: Tag the release

After the upload succeeds:

```bash
git tag -a v0.1.0 -m "ssmforge v0.1.0 — architecture analyzer"
git push origin v0.1.0
```

## Troubleshooting

### `twine` complains about long_description

Run `twine check dist/*` and look for warnings. The description is sourced from
README.md (Markdown). Common issues:

- **404 URLs in README.** Make sure all URLs point to live docs (e.g. docs.ssmforge).
- **License classifier mismatch.** We use SPDX: `License :: OSI Approved :: Apache Software License`.

### Version already exists

If `twine upload` says the version already exists on PyPI:

```bash
# Bump version in pyproject.toml
# version = "0.1.1"   # or 0.2.0 if breaking changes

# Rebuild and upload
python -m build --no-isolation
twine upload dist/ssmforge-0.1.1*
```

We never reuse a version number on PyPI — once uploaded, it's permanent.

### Need to yank a release

If a release is broken:

- https://pypi.org/manage/project/ssmforge/releases/ → click "Yank"
- Yanked releases still install by default but show a warning, or fail with
  `--strict` mode.

## Project links

- Source: https://github.com/lordxmen2k/SSMForge
- Issues: https://github.com/lordxmen2k/SSMForge/issues
- PyPI: https://pypi.org/project/ssmforge/
- Docs: https://github.com/lordxmen2k/SSMForge/blob/main/README.md

## License

Apache 2.0
