# Contributing to emmy-ptx

Thanks for your interest. This is a small, focused project and contributions are welcome.

## Getting started

```bash
git clone https://github.com/richardokonicha/emmy-ptx.git
cd emmy-ptx
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m unittest discover -s tests -v
```

You'll also need `emmy-ml` installed (it provides `Graph.from_dict` for IR loading):

```bash
pip install emmy-ml
```

## Development workflow

1. Fork the repo and create a feature branch.
2. Make your change. Keep commits small and focused.
3. Run the full test suite: `python -m unittest discover -s tests -v`
4. If you touched the compiler, also run it against the Emmy hackathon problems:
   ```bash
   python -m emmy_ptx /path/to/kernel.ir.json > /tmp/out.ptx
   ```
5. Open a pull request with a clear description of what changed and why.

## Code style

- Python 3.10+ compatible.
- No external dependencies in the compiler itself (stdlib only). `emmy-ml` is allowed for IR loading.
- Keep the compiler as a single module (`src/emmy_ptx/compiler.py`). Do not split it unless there's a strong reason.
- No comments unless they explain *why*, not *what*.

## Testing

- Every new op or code path should have a test.
- Tests that need problem data should skip gracefully if the data isn't available.
- Network-dependent tests (GPU runner) should be marked clearly.

## Reporting bugs

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md). Include:
- The IR file (or a minimal reproduction)
- The PTX output (or the error)
- Your Python version and `emmy-ml` version
