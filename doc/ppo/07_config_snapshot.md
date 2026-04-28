# 07 Config Snapshot

Date: 2026-04-27

This snapshot tracks the `agent_ppo` constants that affect observation shape,
action selection, reward ranking, PPO stability, and competition behavior.

## Observation And Model

| Config | Value |
|---|---:|
| `MAP_SIZE` | `128` |
| `LOCAL_MAP_SIZE` | `21` |
| `GLOBAL_MAP_SIZE` | `32` |
| `LOCAL_CHANNELS` | `7` |
| `GLOBAL_CHANNELS` | `6` |
| `SCALAR_DIM` | `288` |
| `MAMBA_HIDDEN_DIM` | `128` |
| `FEATURE_LEN` / `DIM_OF_OBSERVATION` | `9647` |
| `ACTION_NUM` | `16` |
| `MOVE_ACTION_NUM` | `8` |
| `FLASH_ACTION_NUM` | `8` |
| `CONV_CHANNEL` | `64` |
| `VISUAL_FUSION_DIMS` | `[64]` |
| `SCALAR_MLP_DIMS` | `[192, 96]` |
| `VIEW_FUSION_DIMS` | `[192, 128]` |
| `MAMBA_TBPTT_LEN` | `48` |
| `GLOBAL_BFS_THRESHOLD` | `160.0` |
| `LOCAL_BFS_THRESHOLD` | `24.0` |
| `EXPLORE_VECTOR_DANGER_BFS` | `4.0` |
| `EXPLORE_VECTOR_SAFE_BFS` | `8.0` |
| `EXPLORE_VECTOR_DANGER_SCALE` | `0.3` |
| `EXPLORE_VECTOR_FRONTIER_COUNT_NORM` | `256.0` |
| `EXPLORE_VECTOR_FEATURE_SCALE` | `0.25` |

## Action Prior And Runtime Guard

| Config | Value | Purpose |
|---|---:|---|
| `ACTION_PRIOR_WALL_LOGIT` | `-2.0` | Reduce blocked/wall move logits |
| `ACTION_PRIOR_MONSTER_ESCAPE_LOGIT` | `0.5` | Very mild move-direction nudge away from close visible monsters |
| `ACTION_PRIOR_MONSTER_ESCAPE_RADIUS` | `7.0` | Monster escape move prior radius |
| `ACTION_PRIOR_FLASH_BASE_LOGIT` | `-6.0` | Strongly suppress flash by default |
| `ACTION_PRIOR_FLASH_ESCAPE_LOGIT` | `1.0` | Very mild close-danger flash escape prior |
| `ACTION_PRIOR_FLASH_DANGER_RADIUS` | `6.0` | Flash prior danger radius |
| `FLASH_GUARD_ENABLED` | `False` | Runtime flash guard is disabled for PPO probability-basis consistency |
| `FLASH_GUARD_DANGER_BFS` | `6.0` | Flash allowed only when visible monster BFS is at most this value |
| `MONSTER_GUARD_ENABLED` | `False` | Runtime monster guard is disabled for PPO probability-basis consistency |
| `MONSTER_GUARD_DANGER_BFS` | `12.0` | Monster guard active radius |
| `MONSTER_GUARD_L2_EPS` | `0.05` | Allowed numerical slack for no-regress monster-distance checks |
| `RESOURCE_OVERRIDE_ENABLED` | `False` | Greedy reachable resource/frontier override is disabled during PPO training |
| `RESOURCE_OVERRIDE_SAFE_BFS` | `9.0` | Resource override runs only when monster distance is safely above this |
| `RESOURCE_OVERRIDE_TRAIN_LOGIT` | `0.0` | No train-time external boost outside learner-recomputed model logits |
| `RESOURCE_OVERRIDE_EXPLORE_GAP` | `-0.02` | Dormant frontier-first threshold if override is explicitly re-enabled |
| `TRAIN_SAMPLE_TOP_K` | `16` | No top-k truncation for the 16-action space |
| `TRAIN_SAMPLE_TEMPERATURE` | `1.0` | No temperature sharpening before sampling |

Exploration direction is exposed as scalar observation features only. The active
code does not add exploration-direction bias to action logits, keeping PPO
actor/learner probabilities on the same model-distribution basis. The feature
scale is kept below `1.0` because these slots were padding in older checkpoints.

## Reward

| Config | Value |
|---|---:|
| `REWARD_CLIP_MIN` | `-20.0` |
| `REWARD_CLIP_MAX` | `30.0` |
| `REWARD_DENSITY_EPS` | `1.0e-6` |
| `REWARD_STEP` | `0.04` |
| `REWARD_SURVIVE_ALIVE_PER_STEP` | `0.03` |
| `REWARD_SCORE_DELTA` | `0.04` |
| `REWARD_TREASURE` | `8.0` |
| `REWARD_EXPLORE_PER_CELL` | `0.05` |
| `REWARD_EXPLORE_APPROACH` | `0.08` |
| `REWARD_EXPLORE_VECTOR_ALIGN` | `0.04` |
| `PENALTY_EXPLORE_VECTOR_OPPOSE` | `-0.02` |
| `EXPLORE_VECTOR_REWARD_MIN_STRENGTH` | `0.05` |
| `GLOBAL_EXPLORE_TARGET_STEPS` | `512` |
| `GLOBAL_EXPLORE_GAP_CLIP` | `0.25` |
| `REWARD_GLOBAL_EXPLORE_ON_TRACK` | `0.08` |
| `REWARD_GLOBAL_EXPLORE_AHEAD_COEF` | `0.40` |
| `PENALTY_GLOBAL_EXPLORE_LAG_COEF` | `1.20` |
| `PENALTY_GLOBAL_EXPLORE_LAG_MAX` | `-0.35` |
| `REWARD_BUFF_MAINTAIN` | `0.01` |
| `REWARD_BUFF_BEFORE_500` | `3.0` |
| `REWARD_BUFF_AFTER_500` | `10.0` |
| `REWARD_MONSTER_BFS_DELTA_POS` | `0.12` |
| `REWARD_MONSTER_BFS_DELTA_NEG` | `0.18` |
| `REWARD_MONSTER_BFS_DELTA_CLIP` | `8.0` |
| `REWARD_MONSTER_PRESSURE_FULL_BFS` | `6.0` |
| `REWARD_MONSTER_PRESSURE_ZERO_BFS` | `12.0` |
| `REWARD_FLASH_DANGER_BFS_THRESHOLD` | `6.0` |
| `REWARD_FLASH_SUCCESS_DELTA` | `2.0` |
| `REWARD_FLASH_DANGER_SUCCESS` | `3.0` |
| `PENALTY_FLASH_DANGER_FAIL` | `-0.5` |
| `PENALTY_FLASH_SAFE` | `-2.0` |
| `REWARD_NO_VISIBLE_MONSTER` | `0.03` |
| `REWARD_SAFE_MONSTER_BFS_THRESHOLD` | `8.0` |
| `REWARD_SAFE_MONSTER_BFS_CAP` | `12.0` |
| `REWARD_SAFE_MONSTER_BFS_COEF` | `0.004` |
| `REWARD_DANGER_ESCAPE_THRESHOLD` | `8.0` |
| `REWARD_DANGER_ESCAPE_BONUS` | `0.4` |
| `REWARD_RESOURCE_SAFE_BFS_HIGH` | `10.0` |
| `REWARD_RESOURCE_SAFE_BFS_LOW` | `5.0` |
| `REWARD_RESOURCE_DANGER_FACTOR` | `0.5` |
| `REWARD_TREASURE_APPROACH` | `0.04` |
| `REWARD_BUFF_APPROACH` | `0.04` |
| `POST_ESCAPE_SCOUT_STEPS` | `64` |
| `POST_ESCAPE_DANGER_BFS` | `8.0` |
| `REWARD_POST_ESCAPE_PROGRESS` | `0.12` |
| `PENALTY_POST_ESCAPE_REGRESS` | `-0.12` |
| `POST_ESCAPE_PROGRESS_CLIP` | `4.0` |
| `PENALTY_STILL` | `-0.04` |
| `PENALTY_MOVE_STILL` | `-0.08` |
| `PENALTY_CONSECUTIVE_STILL_STEP` | `-0.02` |
| `PENALTY_CONSECUTIVE_STILL_MAX` | `-0.20` |
| `PENALTY_ANTI_LOOP_CLOSE` | `-0.08` |
| `PENALTY_ANTI_LOOP_NEAR` | `-0.03` |
| `REWARD_ANTI_LOOP_PROGRESS` | `0.12` |
| `TRAIL_PROGRESS_WINDOW` | `16` |
| `TRAIL_PROGRESS_DIST` | `8.0` |
| `TRAIL_STALL_DIST` | `4.0` |
| `TRAIL_LOW_UNIQUE_RATIO` | `0.45` |
| `REWARD_TRAIL_PROGRESS` | `0.08` |
| `PENALTY_TRAIL_STALL` | `-0.08` |
| `REWARD_OPEN_MOVE_DELTA` | `0.04` |
| `PENALTY_OPEN_MOVE_LOSS` | `-0.05` |
| `OPEN_SPACE_DANGER_BFS` | `12.0` |
| `OPEN_SPACE_LOW_NEIGHBORS` | `3` |
| `PENALTY_LOW_OPEN_SPACE` | `-0.10` |
| `PENALTY_NEAR_MONSTER_THRESHOLD` | `6.0` |
| `PENALTY_NEAR_MONSTER_SLOPE` | `0.06` |
| `PENALTY_CAUGHT` | `-10.0` |
| `PENALTY_FLASH_ON_CD` | `-0.5` |

Survival milestones are unchanged:
`25/50/75/100/150/200/250/300/400/500 -> 1.5/2.5/3.5/5/7/9/11/14/18/25`.

## PPO And Learner Checks

| Config | Value |
|---|---:|
| `GAMMA` | `0.99` |
| `LAMDA` | `0.95` |
| `INIT_LEARNING_RATE_START` | `0.00001` |
| `BETA_START` | `0.02` |
| `CLIP_PARAM` | `0.2` |
| `VALUE_CLIP_PARAM` | `2.0` |
| `VF_COEF` | `0.5` |
| `GRAD_CLIP_RANGE` | `0.5` |
| `ADVANTAGE_NORMALIZE` | `True` |
| `ADVANTAGE_NORM_EPS` | `1.0e-8` |
| `LEARNER_VALIDATE_BATCH` | `True` |
| `LEARNER_STRICT_BATCH_VALIDATION` | `False` |
| `LEARNER_VALIDATE_MODEL_OUTPUTS` | `True` |
| `LEARNER_CUDA_SYNC_DEBUG` | `False` |
| `TARGET_KL` | `0.01` |
| `AUX_MONSTER_POS_COEF` | `0.2` |
| `AUX_MONSTER_DIST_COEF` | `0.1` |
| `AUX_LOSS_VISIBLE_MONSTER_WEIGHT` | `5.0` |

## Small-scale Runtime And Preload

| Config | Value |
|---|---:|
| `algorithm_on_policy_or_off_policy` | `off-policy` |
| `replay_buffer_capacity` | `128` |
| `preload_ratio` | `0.125` |
| `reverb_remover` | `reverb.selectors.Fifo` |
| `reverb_sampler` | `reverb.selectors.Fifo` |
| `reverb_rate_limiter` | `SampleToInsertRatio` |
| `reverb_samples_per_insert` | `1` |
| `reverb_error_buffer` | `8` |
| `train_batch_size` | `8` |
| `dump_model_freq` | `1` |
| `preload_model` | `true` |
| `preload_model_dir` | `agent_ppo/ckpt` |
| `preload_model_id` | `0` |

`preload_model_id=0` means `agent_ppo` auto-loads the unique
`model.ckpt-*.pkl` in `agent_ppo/ckpt`. The next run should preload the
evaluated initial model that scored higher on platform evaluation, not any
checkpoint produced by the degraded previous training round. Later checkpoints
still use the normal framework save path and must be treated as candidates until
evaluation beats the initial-model baseline.

`reverb_samples_per_insert=1` means the target sample consumption rate is near
one learner use per inserted sequence window. `reverb_error_buffer=8` matches
one learner batch, giving the rate limiter enough slack to fetch a full batch
without allowing multiple stale batches to run ahead of inserts.
`dump_model_freq=1` is the framework's configured interval value; wall-clock
save spacing still depends on learner progress and the framework checkpoint
process.
