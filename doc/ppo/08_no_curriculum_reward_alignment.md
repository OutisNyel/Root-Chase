# 08 No Curriculum And Reward Alignment

Date: 2026-04-27

This page records the removal of the earlier curriculum-learning environment
override and the current reward-alignment direction. For the exact latest
constants, `doc/ppo/07_config_snapshot.md` is the source-of-truth snapshot.

## Diagnosis

The curriculum run did not improve the result metrics that matter:

- `success_rate` stayed at 0.
- `post500_survival_rate` stayed at 0.
- short failures could receive better shaped reward than longer high-score
  failures.

That reward ordering was harmful for PPO because it made "die quickly" look
better than "survive, explore, collect score, then fail".

## Formal Training Distribution

Training no longer lowers difficulty by episode count. Every episode uses
`code/agent_ppo/conf/train_env_conf.toml` directly:

```toml
monster_interval = 300
monster_speedup = 500
max_step = 1000
```

`train_workflow.py` should not contain episode-stage overrides such as:

- `episode_cnt < 500`
- `episode_cnt < 1000`
- `monster_interval = 1000`
- `monster_interval = 600`
- `monster_speedup = 2000`
- `monster_speedup = 1000`

If difficulty needs to change, update the environment config explicitly and
record it in the training notes.

## Current Reward Alignment

The current reward design makes four things consistently valuable:

- real score increases;
- survival progress;
- map exploration and resource collection;
- escaping danger without camping or wasting flash.

Active key constants:

| Config | Current value | Purpose |
|---|---:|---|
| `REWARD_STEP` | `0.04` | base dense survival reward |
| `REWARD_SURVIVE_ALIVE_PER_STEP` | `0.03` | extra live-transition survival reward |
| `REWARD_SCORE_DELTA` | `0.04` | positive `env_info.total_score` delta |
| `REWARD_TREASURE` | `8.0` | treasure pickup |
| `REWARD_BUFF_BEFORE_500` | `3.0` | early buff pickup |
| `REWARD_BUFF_AFTER_500` | `10.0` | late buff pickup |
| `REWARD_TREASURE_APPROACH` | `0.04` | safe treasure approach, currently reduced until survival improves |
| `REWARD_BUFF_APPROACH` | `0.04` | safe buff approach, currently reduced until survival improves |
| `REWARD_MONSTER_PRESSURE_FULL_BFS` | `6.0` | full monster-pressure shaping only in close danger |
| `REWARD_MONSTER_PRESSURE_ZERO_BFS` | `12.0` | monster BFS delta shaping fades to zero by this distance |
| `FLASH_GUARD_ENABLED` | `False` | runtime flash guard disabled during PPO training |
| `MONSTER_GUARD_ENABLED` | `False` | runtime monster guard disabled during PPO training |
| `RESOURCE_OVERRIDE_ENABLED` | `False` | safe resource/frontier override disabled during PPO training |
| `RESOURCE_OVERRIDE_SAFE_BFS` | `9.0` | dormant safe resource/frontier override threshold |
| `RESOURCE_OVERRIDE_TRAIN_LOGIT` | `0.0` | train-time safe resource/frontier boost is disabled |
| `REWARD_POST_ESCAPE_PROGRESS` | `0.12` | keep transferring after escaping danger |
| `REWARD_TRAIL_PROGRESS` | `0.08` | reward real displacement from recent trail |
| `PENALTY_TRAIL_STALL` | `-0.08` | penalize low-unique local loops |
| `PENALTY_LOW_OPEN_SPACE` | `-0.10` | penalize low-exit cells under pressure |
| `REWARD_EXPLORE_PER_CELL` | `0.05` | newly explored passable cells |
| `REWARD_EXPLORE_APPROACH` | `0.08` | frontier approach |
| `REWARD_EXPLORE_VECTOR_ALIGN` | `0.04` | dense reward for movement aligned with the previous frontier vector |
| `PENALTY_EXPLORE_VECTOR_OPPOSE` | `-0.02` | mild penalty for movement opposite the previous frontier vector |
| `REWARD_GLOBAL_EXPLORE_ON_TRACK` | `0.08` | full-map exploration pace is on target |
| `REWARD_GLOBAL_EXPLORE_AHEAD_COEF` | `0.40` | exploration ahead of schedule |
| `PENALTY_GLOBAL_EXPLORE_LAG_COEF` | `1.20` | exploration behind schedule |
| `PENALTY_GLOBAL_EXPLORE_LAG_MAX` | `-0.35` | per-step lag penalty cap |
| `PENALTY_NEAR_MONSTER_THRESHOLD` | `6.0` | close-monster penalty range |
| `PENALTY_NEAR_MONSTER_SLOPE` | `0.06` | close-monster penalty slope |
| `PENALTY_STILL` | `-0.04` | standing still |
| `PENALTY_MOVE_STILL` | `-0.08` | move action that does not move |
| `PENALTY_CONSECUTIVE_STILL_STEP` | `-0.02` | extra repeated-still penalty |
| `PENALTY_CONSECUTIVE_STILL_MAX` | `-0.20` | repeated-still penalty cap |
| `PENALTY_FLASH_SAFE` | `-2.0` | non-danger flash |
| `PENALTY_CAUGHT` | `-10.0` | terminal caught penalty |

Survival milestones:

```text
25/50/75/100/150/200/250/300/400/500
-> 1.5/2.5/3.5/5/7/9/11/14/18/25
```

`score_delta` only uses positive adjacent-observation increases in
`env_info.total_score`. Negative score noise or missing score fields should not
create negative score reward.

## Global Exploration Pace

Full-map exploration reward compares actual map coverage with the expected pace:

```text
explored_ratio = explored_cells / (128 * 128)
target_ratio = min(step / (4 * 128), 1.0)
gap = explored_ratio - target_ratio
```

Behind-schedule exploration is penalized; on-track or ahead-of-schedule
exploration is rewarded. The same gap is exposed as a scalar feature through
existing scalar padding, so observation shape and checkpoint compatibility are
unchanged.

The observation also exposes a low-cost reachable-frontier direction vector in
the existing scalar padding. It uses global passability, visited memory, and the
current hero BFS map to point toward nearby unexplored reachable frontier cells.
This is an input feature only; action logits are not externally biased.
The previous-frame vector also supplies a small dense reward when the next true
displacement follows it, making frontier progress less sparse than waiting only
for new-cell or nearest-frontier BFS events.

## Runtime Behavior Alignment

Training behavior keeps PPO actor/learner probability basis aligned:

- flash logits receive `ACTION_PRIOR_FLASH_BASE_LOGIT = -6.0` by default, with
  only a very mild close-danger escape prior of
  `ACTION_PRIOR_FLASH_ESCAPE_LOGIT = 1.0`;
- non-danger flash has `PENALTY_FLASH_SAFE = -2.0`;
- runtime flash guard, runtime monster guard, and safe resource/frontier override
  are disabled during PPO rollout because learner currently recomputes
  `pi_new` only from model logits plus `legal_action`;
- train-time safe resource/frontier boost stays disabled with
  `RESOURCE_OVERRIDE_TRAIN_LOGIT = 0.0`.

## Next-Run Metrics

Judge this change by result and behavior metrics first:

- `success_rate`
- `post500_survival_rate`
- `sim_score`
- `episode_steps`
- `treasure_count`
- `buff_count`
- `reward`
- `global_explore_ratio`
- `reward_component_mean`
- `reward_component_std`

The expected direction is that reward becomes positively aligned with score,
survival length, exploration, and resources. Loss curves alone are not enough.
