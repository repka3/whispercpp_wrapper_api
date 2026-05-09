# Agent Notes

- Use the project virtualenv for all Python commands: `.venv/bin/python`.
- Do not fall back to system `python` or `python3` for tests, scripts, or app checks in this repo unless the user explicitly asks.
- The test suite is `unittest` based. Run it with:

```bash
.venv/bin/python -m unittest discover -v
```

- `pytest` is not required for the current test suite and may not be installed in the virtualenv.
