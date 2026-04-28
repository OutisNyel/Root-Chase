# 11 Reward Density Monitoring

Date: 2026-04-26

This update adds episode-level monitoring for reward signal density.

## Definition

For each environment step, `Preprocessor` decomposes shaped reward into named
components such as:

- `step_reward`
- `score_reward`
- `treasure_reward`
- `explore_cell_reward`
- `buff_reward`
- `survival_bonus`
- `safety_bonus`
- `monster_bfs_reward`
- `explore_approach_bonus`
- `global_explore_reward`
- `resource_approach_bonus`
- `flash_reward`
- `anti_loop_reward`
- penalties

A component is active when `abs(component) > Config.REWARD_DENSITY_EPS`.

## Episode Metrics

Each `[GAMEOVER]` line and monitor report now includes:

| Metric | Meaning |
|---|---|
| `reward_component_mean` | Average active reward components per step |
| `reward_component_std` | Standard deviation of active component count per step |
| `reward_component_var` | Per-episode variance of active component count |
| `reward_positive_component_mean` | Average positive reward components per step |
| `reward_negative_component_mean` | Average penalty components per step |

The variance is computed as:

```text
mean(count_t^2) - mean(count_t)^2
```

The dashboard keeps mean-family metrics and standard deviation in separate
panels. This avoids plotting the center and spread on one line chart, where the
comparison is visually misleading.

## Usage

Healthy dense shaping should show `reward_component_mean` clearly above 1, not
just the base step reward. If `reward_negative_component_mean` dominates, the
agent is mostly being trained by penalties. If variance is very high, reward
signals are concentrated in rare events instead of providing steady direction.
