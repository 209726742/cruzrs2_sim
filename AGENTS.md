# Repository Guidelines

## Project Structure & Module Organization

- `src/lerobot/`: LeRobot models, policies, dataset processing, and training entry points.
- `cruzr_mujoco_sim/scripts/collection/`: current dual-material collection and dataset pipeline.
- `cruzr_mujoco_sim/scripts/core/`: simulation control, task contracts, object tools, and quality checks.
- `cruzr_mujoco_sim/scripts/tests/`: Python `unittest` regression tests.
- `cruzr_mujoco_sim/scripts/training/`: formal π0.5 canary, training, and resume wrappers.
- `cruzr_mujoco_sim/scripts/archive/`: historical workflows; do not use for new work without confirming applicability.
- `cruzr_mujoco_sim/assets/`, datasets, weights, environments, logs, and outputs are local/external artifacts and are ignored by Git.

## Build, Test, and Development Commands

There is no separate build system. Run commands from the repository root unless a script says otherwise.

```bash
# Show the generic training configuration without starting a job
bash pi05_train.sh dry-run

# Run the current test suite
cd cruzr_mujoco_sim
python -m unittest discover -s scripts/tests -p 'test_*.py' -v

# Check the formal training setup, then start training
bash scripts/training/pi05_formal300_train.sh canary
bash scripts/training/pi05_formal300_train.sh start
```

Provide external dataset, policy, Isaac Sim, and GPU paths explicitly on another machine. Run a canary before a long job; use `resume` for recovery.

## Coding Style & Naming Conventions

Use 4-space indentation for Python and Bash-compatible shell style with `set -euo pipefail` for new scripts. Use `snake_case` for Python functions, variables, and modules; use descriptive `UPPER_SNAKE_CASE` names for environment variables and shell configuration. Match surrounding code and avoid unrelated refactors. No repository-wide formatter or linter is configured; keep imports, comments, and line formatting consistent with adjacent files.

## Testing Guidelines

Add or update a `test_*.py` file under `cruzr_mujoco_sim/scripts/tests/` for behavior changes. Prefer small deterministic tests for contracts, data validation, geometry, and parameter parsing. Run the full `unittest` command before submitting changes; run the relevant script-level smoke or canary when changing training or simulation behavior. No formal coverage threshold is configured.

## Commit & Pull Request Guidelines

Use short, imperative commit subjects that identify the change, for example `Add training README` or `Fix dataset split validation`. Keep commits focused. Pull requests should describe the affected pipeline, commands and environment used, tests/canary results, dataset or checkpoint assumptions, and any behavior or compatibility risks. Never include secrets, private SDK material, model weights, datasets, logs, or generated outputs.

## Security & Configuration Tips

Keep `.env`, keys, tokens, private paths, credentials, and external assets outside commits. Use command-line arguments or environment variables for machine-specific paths, and inspect `git status` and the staged file list before pushing.


## Milestone Publishing

After a substantial change or completed milestone, run the relevant checks, review `git status` and the staged file list, then commit and push the intended tracked changes to `origin/main`. Respect `.gitignore`; never force-add excluded data, weights, assets, logs, credentials, or generated outputs. If authentication or network access prevents pushing, preserve the local work and report the exact blocker.
