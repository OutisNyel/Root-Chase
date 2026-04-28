# AGENTS.md

This file is the shared working memory for different CLI agents/tools in this repository.

## 1) Project Baseline (Current)

- Active algorithm: `code/agent_ppo` (PPO + Mamba hidden state).
- Current observation/model contract (from `doc/ppo`):
  - `DIM_OF_OBSERVATION = 9647`
  - local: `7x21x21`
  - global: `6x32x32`
  - scalar: `288`
  - hidden state input: `128`
  - action space: `16` (`0-7` move, `8-15` flash)
  - `MAMBA_TBPTT_LEN = 32`
- Runtime mode remains near on-policy PPO on top of off-policy framework config:
  - `algorithm_on_policy_or_off_policy = "off-policy"`
  - `local_aisrv_workflow` is required in small-scale Docker runtime.

## 2) Hard Rules (Must Follow)

1. Documentation-first for meaningful changes.
   - Update `doc/ppo` design docs first, then code, then sync docs with actual implementation.
2. Do not modify these paths under `code/`:
   - `common_python/`
   - `kaiwudrl/`
   - `tools/`
3. Final runtime is a custom Docker container, not this local machine.
   - Avoid local-only assumptions.
   - Keep paths/configs compatible with packaged/containerized execution.

## 3) Allowed Change Surface (Default)

- `code/agent_ppo/**`
- `code/conf/**` (when required by algorithm/config changes)
- `doc/ppo/**` (and related docs for doc-first workflow)
- Repo-level docs like this file (`AGENTS.md`) when team memory needs update.

If a task appears to require edits in blocked paths, stop and ask for explicit override before changing scope.

## 4) Doc-First Mapping (Quick)

- Observation/layout/memory changes: `doc/ppo/02_observation_and_memory.md`
- Model/hidden state/head changes: `doc/ppo/03_model_and_hidden_state.md`
- Reward/loss/SampleData changes: `doc/ppo/04_reward_training_aux_loss.md`
- Workflow/architecture changes: `doc/ppo/01_architecture.md`
- Config changes affecting behavior: `doc/ppo/07_config_snapshot.md`
- Process/governance updates: `doc/ppo/05_doc_first_workflow.md`

## 5) Packaging / Deployment Context

- Release packaging includes only:
  - `code/conf`
  - `code/agent_ppo`
  - `code/.vscode`
  - `code/kaiwu.json`
  - `code/train_test.py`
- Container preload model convention:
  - checkpoint path: `code/agent_ppo/ckpt/model.ckpt-*.pkl`
  - preload settings are managed via `code/conf/configure_app.toml`
  - keep exactly one preload checkpoint in `agent_ppo/ckpt` when using auto preload id.

## 6) Pre-Commit Checklist (Agent Side)

- Did I update docs first for behavior-changing work?
- Did I avoid `code/common_python`, `code/kaiwudrl`, and `code/tools`?
- Are config/path assumptions valid in Docker runtime?
- If observation/model/reward/config changed, did I sync the corresponding `doc/ppo/*` files?
