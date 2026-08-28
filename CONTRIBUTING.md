# Contributing to MemStrata

Thanks for improving MemStrata. Keep the method repository self-contained:
`memstrata` must not import `vmem_bench`; cross-repository evaluation belongs
in VMem-Bench adapters.

## Before opening a pull request

```bash
python3 -m pip install -e ".[dev]"
PYTHONPATH=src python3 -m pytest -q
```

For production changes, also run the recording smoke described in
[`README.md`](README.md). Do not commit model weights, source videos, generated
outputs, credentials, or machine-specific absolute paths.

## Documentation and compatibility

Update the English `README.md` and the corresponding `README.zh.md` when a
user-facing workflow changes. Preserve the `paper-reproduction` branch as a
paper-metric freeze; production improvements belong on `main` unless they are
explicitly backported without changing the frozen protocol.
