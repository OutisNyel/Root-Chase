# 10 Survival Killshot Update

Date: 2026-04-25

## 2026-04-27 Current Overrides

This page started as a 2026-04-25 change note. The current active code has a few
later overrides:

- FIFO replay is configured with `reverb_samples_per_insert=1` and
  `reverb_error_buffer=8`, so `sample_production_and_consumption_ratio` should
  be read around 1.
- Training rollout sampling uses `TRAIN_SAMPLE_TOP_K=16` and
  `TRAIN_SAMPLE_TEMPERATURE=1.0`.
- PPO uses `INIT_LEARNING_RATE_START=1e-5`, `CLIP_PARAM=0.2`,
  `VALUE_CLIP_PARAM=2.0`, `VF_COEF=0.5`, `GRAD_CLIP_RANGE=0.5`, and
  `BETA_START=0.02`; `TARGET_KL=0.01`.
- Flash prior uses `ACTION_PRIOR_FLASH_BASE_LOGIT=-6.0`,
  `ACTION_PRIOR_FLASH_ESCAPE_LOGIT=1.0`, and
  `ACTION_PRIOR_FLASH_DANGER_RADIUS=6.0`.
- Monster move pressure is now weaker and close-range gated:
  `ACTION_PRIOR_MONSTER_ESCAPE_LOGIT=0.5`,
  `ACTION_PRIOR_MONSTER_ESCAPE_RADIUS=7.0`, and BFS-delta reward fades out by
  BFS `12`.
- BFS feature depth is reduced to `GLOBAL_BFS_THRESHOLD=160.0` and
  `LOCAL_BFS_THRESHOLD=24.0` to cut feature-process latency.
- Runtime flash guard, runtime monster guard, and safe resource/frontier override
  are disabled during PPO rollout. Learner currently recomputes `pi_new` only
  from model logits plus `legal_action`, so actor-side `old_prob` must use the
  same model-distribution basis.
- Safe resource/frontier rollout boost is disabled with
  `RESOURCE_OVERRIDE_TRAIN_LOGIT=0.0`.

## Diagnosis

The 15:25-18:55 run proved that the plumbing is no longer the main blocker:

- 48-step Mamba windows are active: `seq_max_len=48` for all learner updates.
- Near on-policy FIFO replay is active; the current 1:1 consumption config should
  show `sample_production_and_consumption_ratio` close to 1.
- Reward ranking is fixed: `steps/reward corr=0.960`, `score/reward corr=0.983`.

The remaining failure is behavioral: all 715 sampled episodes failed, `post500_count=0`, only 4 episodes crossed 300 steps, and median survival stayed at 18 steps. PPO also barely moved the policy (`approx_kl≈0`, `clip_frac≈0`), so continuing the same run mostly repeats the same early-death distribution.

## Changes

The current active code keeps these mechanisms, but the active fine-tuning
strength is now governed by the initial-model conservative update below.

### Model Action Prior

`Model.forward` now adds a deterministic prior to the logits before masked softmax. Actors and learner share the same model-side prior. Rollout sampling then applies only the legal mask, so `SampleData.prob` stores the same kind of distribution that learner recomputes as `pi_new`.

- Adjacent wall/blocked cells add `ACTION_PRIOR_WALL_LOGIT=-2.0` to the corresponding move logit.
- Visible local monsters add a very mild escape-direction prior to move logits,
  radius `7`, strength `0.5`.
- Flash logits are suppressed by default with base strength `-6.0`; very close
  visible monsters add only a mild escape-direction prior to flash logits,
  radius `6`, strength `1.0`.

The action order follows the environment contract: move `0..7` and flash `8..15` are `E, NE, N, NW, W, SW, S, SE`.

### Reward

Reward now prioritizes early survival, safe distance, escape, and exploration:

- Dense survival: `REWARD_STEP=0.04` plus
  `REWARD_SURVIVE_ALIVE_PER_STEP=0.03`.
- Early milestones: `25/50/75/100/150/200/250/300/400/500`.
- Terminal caught penalty is `-10.0`.
- Monster distance delta strengthened: away `0.12`, toward `0.18`, clip `8`.
- Safe-distance bonus starts at BFS `8`, capped at BFS `12`.
- Escaping from `<=8` to `>8` BFS gives `+0.4`.
- Treasure and buff rewards are raised: treasure pickup `8.0`, buff before 500 `3.0`, after 500 `10.0`.
- Treasure/buff approach rewards are reduced to `0.04/0.04` until the policy
  can reliably survive and explore; frontier/new-cell exploration is stronger.

New reward debug metrics:

- `safety_bonus`
- `monster_bfs_reward`
- `explore_approach_bonus`
- `resource_approach_bonus`
- `flash_reward`

### PPO

Current PPO fine-tuning is conservative to preserve the evaluated initial model:

- `INIT_LEARNING_RATE_START=1e-5`
- `CLIP_PARAM=0.2`
- `VALUE_CLIP_PARAM=2.0`
- `VF_COEF=0.5`
- `GRAD_CLIP_RANGE=0.5`
- `BETA_START=0.02`
- `TARGET_KL=0.01`
- active-timestep GAE advantage normalization is enabled

Learner logs now include raw `adv_mean` and `adv_std` before normalization.

### Training Action Sampling

Training rollout currently samples from the legal-masked model distribution:

```text
model logits with model-side action prior
-> legal masked policy
-> top_k=16
-> temperature=1.0
-> sample action
```

`top_k=16` over a 16-action space and `temperature=1.0` leave the distribution
unchanged. `ActData.prob` stores this sampling distribution, so PPO compares
`pi_new(action)` against the same policy basis used to collect the sample. External
runtime guards must stay out of PPO rollout unless learner loss also reproduces
them.

## Short-run Acceptance

Do not judge this by loss alone. In the first 30-45 minutes after restart, check:

- `post200_count` and `post300_count` should rise visibly.
- `median steps` should move away from the 18-step failure mode.
- `wall_or_still_after_move_rate` should drop.
- `avg_min_monster_bfs` should rise or `monster_bfs_reward` should become less negative.
- `approx_kl` and `clip_frac` should no longer stay pinned at zero for the whole run.

If these do not move, the next blocker is likely environment observation/action semantics or needing an explicit planner policy, not another reward coefficient sweep.

## 2026-04-25 Learner Sequence Forward Speedup

The learner no longer calls full `Model.forward()` once per active timestep. `Model.forward_sequence(obs_seq, seq_mask)` keeps the external `SampleData` and PPO loss contracts unchanged, but changes the internal compute layout:

```text
[B, T, obs] -> [B*T, obs]
batch local/global CNN + scalar MLP + action prior
reshape back to [B, T, hidden]
loop only the Mamba hidden-state cell over T=48
mask padding timesteps before PPO/value/aux losses
```

Actor inference is unchanged and still uses the single-step `Model.forward(obs)`. The optimization is therefore isolated to `code/agent_ppo/model/model.py` and `code/agent_ppo/algorithm/algorithm.py`; it does not require any kaiwudrl runtime or replay schema change.

Smoke equivalence check compared `forward_sequence()` against the old step-by-step loop on the same model and batch. Max output differences were around `1e-6`, which is expected floating-point noise.

## 2026-04-26 Platform Behavior Guard

Platform testing showed a gap between training-dashboard scores and submission
behavior: the model can average much higher on the platform, but it tends to
camp until chased and often spends flash at the opening.

The 2026-04-26 competition-facing fix was intentionally pragmatic:

- Flash is suppressed by default with `ACTION_PRIOR_FLASH_BASE_LOGIT=-4.0`.
- Flash escape prior is stronger but only within BFS radius `6`.
- Runtime flash guard was disabled by default at that point
  (`FLASH_GUARD_ENABLED=False`).
- Deterministic `exploit()` used policy argmax by default; resource/frontier
  override was disabled by default at that point
  (`RESOURCE_OVERRIDE_ENABLED=False`).
- Reward now punishes non-danger flash much harder and raises resource/frontier
  approach plus anti-camping pressure.

This does not change the action space or model checkpoint format. It is meant to
make the submitted policy actively collect score while preserving the learned
escape reflex near monsters.

## 2026-04-26 Reward Density Monitoring

Episode logs and monitor panels now include reward density metrics:

- `reward_component_mean`
- `reward_component_std`
- `reward_component_var`
- `reward_positive_component_mean`
- `reward_negative_component_mean`

See `doc/ppo/11_reward_density_monitoring.md` for the exact definition.

## 2026-04-26 Global Exploration Progress Reward

Training now adds a full-map exploration schedule reward. The target is to have
seen the whole `128x128` map by `4 * 128 = 512` environment steps:

```text
explored_ratio = explored_cells / (128 * 128)
target_ratio = min(step / (4 * 128), 1.0)
gap = explored_ratio - target_ratio
```

If `gap` is negative, the agent receives a lag penalty capped at `-0.35` per
step. If `gap` is non-negative, it receives an on-track reward plus a small
ahead-of-schedule bonus. Logs and monitor panels include
`global_explore_reward`, `global_explore_ratio`, `global_explore_target_ratio`,
and `global_explore_gap`.

The same progress signal is also exposed to the policy as one scalar feature:

```text
global_explore_progress_feature =
    clip(gap, -0.25, 0.25) mapped from [-0.25, 0.25] to [0.0, 1.0]
```

`0.5` means the current full-map exploration pace is on target, lower values
mean the agent is behind schedule, and higher values mean it is ahead of
schedule. This uses existing scalar padding, so observation shape and checkpoint
compatibility are unchanged.

## 2026-04-27 Monster Pressure Gating And Route Safety

Platform review showed a "magnet repulsion" failure mode: when a monster is
visible but not actually close, the policy can keep maximizing monster distance
instead of following the S-curve route, finding an exit, or using the safe window
to explore.

The fix changes monster pressure from an always-on repulsion field into a
danger-gated signal:

- move-logit monster escape prior is weaker and only close-range;
- monster BFS delta reward is full at BFS `<=6`, linearly decays, and is zero by
  BFS `12`;
- safe-distance reward is capped at BFS `12`, so the agent is not paid to keep
  fleeing forever;
- resource/frontier override logic remains available as dormant helper code, but
  is disabled during PPO rollout;
- training rollout does not boost the safe resource/frontier action
  (`RESOURCE_OVERRIDE_TRAIN_LOGIT=0.0`).

Reward code now adds three route-quality step signals:

- `post_escape_scout_reward`: after danger, keep moving away from the last danger
  point instead of drifting back into the same pocket;
- `trail_reward`: reward real displacement from the recent trail and penalize
  low-unique local loops;
- `route_safety_reward`: reward increasing adjacent open exits and penalize
  low-exit cells under monster pressure.

The intended behavior is: fear close monsters, stop over-fearing far monsters,
and spend safe windows on route transfer, frontier exploration, and resource
collection.

## 2026-04-27 Small Fix Restart

This plan was superseded by the initial-model conservative fine-tune below
after platform evaluation showed the previous trained candidate had degraded
badly. Keep this section as run history, not as the active next-run config.

The 3-aisrv / 12-gamecore run reached `global_step=6261` with corrected PPO
probabilities, but the policy still averaged about 41 steps in the last 100
episodes, `caught_rate=1.0`, `post500_survival_rate=0.0`, and treasure/buff
pickup near zero. Learner-side `approx_kl` averaged around `0.0007` and
`clip_frac=0.0`, so the policy was barely moving while aisrv feature processing
spent roughly 260-280 ms per step, almost all in BFS.

That superseded restart changed only the low-risk training knobs:

- raise learning rate to `5e-5`, lower entropy coefficient to `0.01`, and lower
  `TARGET_KL` to `0.02`;
- reduce model-side flash dependence: flash base prior `-6.0`, flash escape
  prior `2.0`, monster escape prior `1.0`;
- add `REWARD_SURVIVE_ALIVE_PER_STEP=0.03`, keep the base step reward at `0.04`,
  and increase terminal caught penalty to `-10.0`;
- shift early behavior pressure from treasure/buff approach toward exploration:
  new cells `0.05`, frontier approach `0.08`, treasure/buff approach `0.04/0.04`;
- reduce BFS search thresholds to `GLOBAL_BFS_THRESHOLD=160.0` and
  `LOCAL_BFS_THRESHOLD=24.0`.

Those thresholds were only for that superseded restart. The active policy now is
to preserve the evaluated initial model and accept a candidate only if platform
or validation behavior beats that baseline.

## 2026-04-27 Exploration Guard Follow-up

Review of the 04:49-06:28 training logs and evaluation playback showed two
remaining behavior failures:

- global exploration lag persisted: the last 100 parsed episodes averaged
  `global_explore_ratio=0.0511` against `target=0.0912`;
- the policy still sometimes moved back toward visible monster pressure, keeping
  `caught_rate=1.0` and `post500_survival_rate=0.0`.

The first attempted follow-up made the planner helpers active in rollout, but
that would compare externally rewritten actor `old_prob` against learner-side
raw model `new_prob`. The corrected follow-up keeps those helpers dormant during
PPO training and only keeps changes that preserve the learner-recomputed
distribution basis:

- `predict()` samples from the legal-masked model distribution and stores that
  same distribution in `ActData.prob`.
- Runtime flash guard, runtime monster guard, and resource/frontier boost are
  disabled by default.
- Resource/frontier helper code can still use reachable BFS and frontier-first
  target choice if it is explicitly re-enabled in a future learner-matched
  implementation.
- Exploration rewards and post-escape regress penalties are raised so short
  local dithering and returning to danger are less competitive with actual map
  transfer.

If external rules are needed later, the required flash/monster/resource state
must be serialized in `SampleData` and `Algorithm._compute_loss()` must recreate
the exact same behavior distribution before computing PPO ratio, KL, and
clip fraction.

## 2026-04-27 Explore Direction Observation

To give the policy a lower-noise exploration signal without hard-selecting
actions, the preprocessor now fills six scalar padding slots with a reachable
frontier direction:

```text
explore_vec_x
explore_vec_z
explore_vec_norm
nearest_frontier_bfs_norm
frontier_count_norm
explore_safety_factor
```

The frontier is computed from global passability, visited memory, and the
current hero BFS map. Direction strength is reduced when visible monster BFS is
inside the danger band. The six raw features are scaled by
`EXPLORE_VECTOR_FEATURE_SCALE=0.25` before entering the model because these slots
were previously zero padding in old checkpoints. No exploration action-prior or
logits bias is active in this stage; PPO still samples from model logits plus
legal masking only.

The previous observation's frontier vector also provides a small dense reward on
the next transition: true displacement aligned with that vector gets up to
`REWARD_EXPLORE_VECTOR_ALIGN=0.04`, while moving against it gets up to
`PENALTY_EXPLORE_VECTOR_OPPOSE=-0.02`. This reward is strength-gated and inherits
the monster-safety downweight from the vector.

## 2026-04-27 Initial-Model Conservative Fine-Tune

Platform evaluation showed the evaluated initial model is still the best
baseline:

| Model | Avg score | Terminal steps | Treasures | Skill uses |
|---|---:|---:|---:|---:|
| Initial preload model | `854.98` | `450.65` | `1.79` | `0.17` |
| Previous trained candidate | `297.79` | `179.86` | `0.28` | `0.98` |

The previous trained candidate is therefore rejected as a preload source. The
next run must start from the evaluated initial model in `code/agent_ppo/ckpt`,
and any newly trained checkpoint remains only a candidate until platform or
validation evaluation beats the initial baseline.

The fine-tuning knobs are deliberately conservative:

- `INIT_LEARNING_RATE_START=1e-5`
- `BETA_START=0.02`
- `TARGET_KL=0.01`
- `ACTION_PRIOR_FLASH_BASE_LOGIT=-6.0`
- `ACTION_PRIOR_FLASH_ESCAPE_LOGIT=1.0`
- `ACTION_PRIOR_MONSTER_ESCAPE_LOGIT=0.5`

Monitor candidate quality by behavior first. Skill uses should not climb while
terminal steps, score, and treasure count drop. If the candidate is below the
initial-model baseline, discard it instead of chaining another training round
from it.
