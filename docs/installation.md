# Installation

## Requirements

- Python >= 3.10
- A working compiler toolchain (for numpy / scipy)
- Virtual environment strongly recommended

## Development installation (recommended)

```bash
pip3 install -e .
```

This installs MxL-GEN in editable mode, so changes to the source code are immediately reflected.

## Documentation tooling

```bash
pip3 install mkdocs mkdocs-material mkdocstrings mkdocstrings-python
```

## Verify installation

```bash
python3 - <<'PY'
import MxL_GEN
print(MxL_GEN.__file__)
PY
```
