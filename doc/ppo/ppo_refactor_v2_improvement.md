# Archived PPO Refactor v2 Proposal

Date: archived on 2026-04-27

This file is retained only as historical context. It is not the current training
plan and its old hyperparameter suggestions must not be used as source of truth.

Current active values are tracked in:

- [07_config_snapshot.md](07_config_snapshot.md)
- [04_reward_training_aux_loss.md](04_reward_training_aux_loss.md)
- [09_sequence_global32_smallscale.md](09_sequence_global32_smallscale.md)
- [10_survival_killshot_update.md](10_survival_killshot_update.md)

## Superseded Ideas

The original proposal suggested changes such as lowering `CLIP_PARAM`, using a
large entropy coefficient, and restarting training from scratch. Those ideas were
superseded by later implementation work:

- PPO currently uses `CLIP_PARAM = 0.2`, `BETA_START = 0.02`, and
  `TARGET_KL = 0.01`.
- Learning rate currently starts at `INIT_LEARNING_RATE_START = 1e-5`.
- Training rollout sampling now uses `TRAIN_SAMPLE_TOP_K = 16` and
  `TRAIN_SAMPLE_TEMPERATURE = 1.0`.
- Action behavior during PPO rollout is shaped with model-side priors only;
  external runtime guard/resource override helpers are disabled unless learner
  loss is updated to reproduce the same behavior distribution.
- Mamba training uses non-overlapping `MAMBA_TBPTT_LEN = 48` sequence windows.
- Small-scale training remains framework `off-policy` with FIFO replay,
  `reverb_samples_per_insert = 1`, and `reverb_error_buffer = 8`.

Do not copy constants from older experiment notes into code. Use
`07_config_snapshot.md` before changing `code/agent_ppo/conf/conf.py` or
`code/conf/configure_app.toml`.
