# 09 48-Step Sequence Windows And Near On-policy PPO

Date: 2026-04-27

## Constraint

Small-scale Docker runs force:

```toml
remote_agent_default_runtime_mode = "local_aisrv_workflow"
```

The kaiwudrl runtime rejects native `on-policy` mode with
`local_aisrv_workflow`, so the active app config intentionally remains:

```toml
algorithm_on_policy_or_off_policy = "off-policy"
```

The current design is "near on-policy PPO": app-level sequence windows plus a
small FIFO replay buffer and a low sample-reuse ratio. This avoids changes under
`code/kaiwudrl/`.

## Sequence Window Contract

`sample_process` computes returns and advantages inside a complete episode, then
cuts the episode into non-overlapping windows:

```text
[0..47] [48..95] [96..143] ...
```

The last window is padded to `MAMBA_TBPTT_LEN = 48`. Padding timesteps carry
`seq_mask = 0` and do not contribute to PPO, value, entropy, or auxiliary
losses.

Each replay item is one sequence window, not one environment step.

| Field | Shape per replay item | Meaning |
|---|---:|---|
| `obs` | `FEATURE_LEN * 48` | 48 observations, each already containing `h_in` |
| `legal_action` | `ACTION_NUM * 48` | legal action mask per timestep |
| `act` | `48` | sampled action id per timestep |
| `prob` | `ACTION_NUM * 48` | behavior policy probability per timestep |
| `reward`, `value`, `reward_sum`, `advantage`, `done` | `48` | PPO scalars |
| `monster_pos_target` | `2 * 48` | auxiliary position target |
| `monster_pos_mask`, `monster_dist_target`, `monster_dist_mask` | `48` | auxiliary masks and labels |
| `seq_id` | `1` | episode sequence id |
| `seq_pos` | `48` | 0-based episode position |
| `seq_mask` | `48` | 1 for real timesteps, 0 for padding |
| `seq_len` | `1` | real length of the window |

The learner reshapes to `[batch, time, field]`, runs visual/scalar encoders in a
single flattened pass, and loops only the Mamba hidden transition over time.
This keeps the external model and PPO contracts unchanged while removing most
per-timestep Python overhead.

## Behavior Probability

`prob` is the active rollout behavior distribution after model-side action
priors and legal masking. The current actor path intentionally does not apply
external flash suppression, near-monster move regression suppression, or a safe
resource/frontier boost before training sampling:

```text
model logits with model-side action prior
-> legal mask
-> TRAIN_SAMPLE_TOP_K=16
-> TRAIN_SAMPLE_TEMPERATURE=1.0
```

This is intentional. Learner-side `pi_new` is recomputed from model logits plus
`legal_action`; actor-side `pi_old` must use that same policy basis. With
`TRAIN_SAMPLE_TOP_K=16` over 16 actions and `TRAIN_SAMPLE_TEMPERATURE=1.0`, the
current training distribution is not truncated or sharpened.

## Current FIFO Replay Config

The runtime config is interpreted in window units:

```toml
replay_buffer_capacity = 128
preload_ratio = 0.125
reverb_remover = "reverb.selectors.Fifo"
reverb_sampler = "reverb.selectors.Fifo"
reverb_rate_limiter = "SampleToInsertRatio"
reverb_samples_per_insert = 1
reverb_error_buffer = 8
train_batch_size = 8
```

This starts learner consumption after roughly 16 windows are inserted. Each
learner batch consumes 8 windows, up to 384 real or padded timesteps.

`sample_production_and_consumption_ratio` should now be interpreted around the
configured `reverb_samples_per_insert = 1`. A ratio near 1 means the intended
"train about once per produced window" mode is active.

`reverb_error_buffer = 8` is deliberately tied to `train_batch_size = 8`. Reverb
computes:

```text
error = (inserts - min_size_to_sample) * samples_per_insert - samples
```

and allows sampling while `error >= -error_buffer`. Keeping the buffer at one
batch lets the learner fetch a complete batch when the table first reaches
`min_size_to_sample`, but it prevents the learner from running multiple batches
ahead of newly inserted windows.

## Why It Is Still Only Near On-policy

This setup reduces staleness, but it is not equivalent to a fully synchronous
on-policy PPO implementation:

- actors and learner are still decoupled by replay;
- each window is targeted for about one learner use, with one batch of limiter
  slack;
- learner-side `pi_new` remains the current legal-masked model distribution,
  while `pi_old` is the stored legal-masked actor model distribution;
- kaiwudrl native on-policy synchronization remains disabled.

The practical goal is to keep samples fresh enough for PPO while preserving the
small-scale runtime that can actually start.

## Acceptance Checks

Mechanism checks:

- `algorithm_on_policy_or_off_policy = "off-policy"` in
  `code/conf/configure_app.toml`.
- `seq_max_len = 48` in learner logs.
- `seq_mean_len` is far above 1 when episode lengths are above one window.
- `sample_production_and_consumption_ratio` stays close to 1.
- `clip_frac` and `grad_norm` do not stay saturated for long periods.

Training-quality checks:

- `total_reward` should align better with `sim_score` and `episode_steps`.
- `success_rate`, `post500_survival_rate`, `sim_score`, `episode_steps`, and
  `treasure_count` remain the primary result metrics.
- Loss curves alone are not sufficient to judge effectiveness.

## Window Length Note

The current code snapshot uses `MAMBA_TBPTT_LEN = 48`, raised from 32 to give
Mamba a slightly longer credit-assignment horizon without jumping all the way to
64. Moving to 64 can still be reasonable later once most episodes are
comfortably longer than 64 steps and learner throughput is not blocked by sample
starvation. Any window-length change must update both code and this document
because replay item sizes, batch memory, and sequence metrics all change.
