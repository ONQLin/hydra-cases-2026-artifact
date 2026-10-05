# Tests and validation

`artifact/` is reserved for reproducing the original HYDRA paper. Tests for
subsequent simulator features live here.

From the repository root, using the Python environment described in the
[main README](../README.md#environment-setup):

```bash
python -m unittest discover -s tests
```

`test_*.py` contains unit and contract tests. Three real Garnet tests require
`RUN_CHIPSIM_GARNET_TESTS=1` and the [CHIPSIM setup](../integrations/chipsim/README.md).
These optional tests are skipped by default.

[Validation suites](validation/README.md) contain simulation runners, small
input fixtures, reproduction instructions and curated reference results.
Write new run outputs to ignored `output_sanity_checks/` directories.
