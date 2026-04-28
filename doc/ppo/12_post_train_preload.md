# 12 Initial-Model Preload And Candidate Selection

Date: 2026-04-27

## Goal

Run the next fine-tuning job from the evaluated initial `agent_ppo` checkpoint,
not from the degraded checkpoint produced by the previous training round. This
keeps the framework and normal save path unchanged while making checkpoint
selection explicit.

The initial model is the current baseline to beat:

| Model | Avg score | Terminal steps | Treasures | Skill uses |
|---|---:|---:|---:|---:|
| Initial preload model | `854.98` | `450.65` | `1.79` | `0.17` |
| Previous trained candidate | `297.79` | `179.86` | `0.28` | `0.98` |

Do not copy checkpoints from the rejected previous training candidate into the
preload directory for the next run.

The checkpoint used by the container is copied into `agent_ppo` as the only
preload checkpoint:

`code/agent_ppo/ckpt/model.ckpt-<id>.pkl`

This location is intentional. The small-scale Docker code package can read files
under `agent_ppo`, while external backup folders are not guaranteed to be mounted
into the learner container.

## Runtime Config

`code/conf/configure_app.toml` uses the framework preload hook:

```toml
preload_model = true
preload_model_dir = "agent_ppo/ckpt"
preload_model_id = 0
```

`preload_model_id = 0` is an `agent_ppo` auto mode. During `load_model`, the
agent reads the only `model.ckpt-*.pkl` file under `agent_ppo/ckpt`. If zero or
multiple checkpoint files exist there, loading fails fast instead of silently
starting from the wrong model.

`algorithm_on_policy_or_off_policy` stays `off-policy`, because
`local_aisrv_workflow` rejects native kaiwudrl on-policy mode.

## Load And Save Semantics

Only the initial learner model is loaded from the unique
`code/agent_ppo/ckpt/model.ckpt-*.pkl`.

After the learner starts, `agent_ppo.Agent.save_model()` is unchanged. New models
continue to be written through the normal framework-provided save path and model
id. The preload directory should contain only one `model.ckpt-*.pkl`.

## Helper Script

Use:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\prepare_agent_ppo_post_train.ps1 `
  -ZipPath "<verified_initial_model_zip>"
```

The script is idempotent:

- Creates `code/agent_ppo/ckpt` if needed.
- Keeps an existing unique `model.ckpt-*.pkl` unless `-ForceRefresh` is passed.
- Infers `ModelId` from the unique local checkpoint or the unique checkpoint in
  the zip. Pass `-ModelId` only when there is ambiguity.
- Sets `preload_model=true`, `preload_model_dir="agent_ppo/ckpt"`, and
  `preload_model_id=0`.

## Verification

Required checks before restarting training:

- `code/conf/configure_app.toml` parses as TOML.
- `preload_model` is `true`.
- `preload_model_dir` is `agent_ppo/ckpt`.
- `preload_model_id` is `0`.
- `code/agent_ppo/ckpt` contains exactly one `model.ckpt-*.pkl`.
- Current `agent_ppo.model.Model` can strict-load the checkpoint.
- No files under `code/common_python/`, `code/kaiwudrl/`, or `code/tools/` changed.

## Candidate Policy

Treat every later checkpoint as `train_candidate`. Keep the initial model as
`best_eval_model` until a candidate beats it on platform or validation behavior:
score, terminal steps, treasure count, and skill usage. If a candidate increases
skill usage while score, survival, and treasure count fall, reject it and restart
from the initial preload model again.
