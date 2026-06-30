# Tests

Runnable, hardware-free checks for the toolchain's scripts and config. No AWS
credentials, GPU, or robot needed — AWS clients are stubbed (`conftest.py`), so
`--dry-run` paths are asserted to make **zero** real AWS calls.

## Run

```bash
pip install -r tests/requirements.txt   # pytest, pyyaml, torch
pytest tests/ -q
```

## Coverage

| File | What it checks |
|------|----------------|
| `test_env_registration.py` | `training.envs` imports on a bare box; `PickAndPlaceUR3-v0` registers when gymnasium is present. |
| `test_dry_runs.py` | `launch_rl.py`, `cosmos_setup.py` (launch/generate), `cosmos3_generate.py`, and the edge scripts produce valid `--dry-run` output and make no real AWS calls. |
| `test_repo_hygiene.py` | Buildspecs are valid YAML; no merge-conflict markers; no hardcoded AWS account in runnable code. |

## Notes

- Tests that need an optional dep (`torch`, `gymnasium`) **skip** cleanly when it's
  absent rather than fail — so the suite runs anywhere.
- These do NOT validate hardware-gated runtime (Jetson inference, p5 Cosmos runs,
  Isaac Sim on a GPU). Those remain manual — see `docs/ROADMAP.md`.
