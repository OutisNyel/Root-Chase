# 04 运维验收清单（Ops Checklist）

本页是训练运行和验收时的“先后顺序清单”，按优先级排查。

## 1. 集成自检（启动前）

1. `code/conf/algo_conf_gorge_chase.toml` 的 `[diy]` 指向 `agent_diy`。
2. `code/agent_diy/workflow/train_workflow.py` 读取 `agent_diy/conf/train_env_conf.toml`。
3. `code/agent_diy/conf/conf.py` 中关键值确认：
   - `MAX_MONSTER_SPEED=2.0`
   - `MONSTER_DANGER_*` 一组参数存在
   - `flash late fusion` 相关开关不存在
4. 行为指标已在面板中配置：
   - `danger_treasure_chase_rate`
   - `stuck_event_rate`
   - `corner_stuck_duration`
   - `first_pass_treasure_pick_rate`
   - `return_path_caught_rate`
   - `post500_survival_rate`
   - `early_jump_step`

## 2. 运行健康（训练中）

1. `train_global_step`、`predict_succ_cnt`、`sample_receive_cnt` 持续增长。
2. `sample_production_and_consumption_ratio` 不长期塌陷。
3. `total_loss`、`value_loss` 先下降后平台。
4. `policy_loss` 不应过早长期贴近 0。
5. `reward` 观察移动均值，不看单点噪声。

## 3. 行为验收（先看行为再看 loss）

```mermaid
flowchart LR
    A["回放观感"] --> B["行为指标趋势"]
    B --> C["是否满足验收阈值"]
    C --> D["通过/继续重训"]
```

### 3.1 怪物相关

1. 危险逼近时，是否先脱险再拿宝箱。
2. `danger_treasure_chase_rate` 是否稳定下降。
3. `return_path_caught_rate` 是否明显下降。
4. `post500_survival_rate` 是否提升。

### 3.2 卡死相关

1. 墙角驻留是否减少。
2. `stuck_event_rate` 是否下降。
3. `corner_stuck_duration` 是否下降。

## 4. 危险函数专项排障

出现“过于怂不拿宝箱”时，优先调：
1. 降低 `MONSTER_DANGER_EXP_K`（危险曲线更平缓）。
2. 降低 `MONSTER_DANGER_TERRAIN_COEF`（地形放大减弱）。
3. 降低 `MONSTER_DANGER_HIGH_THRESHOLD` 会让高风险更易触发，通常反向操作是升高它。
4. 提高 `DANGER_TREASURE_MIN_REWARD_SCALE`（危险态仍保留更多宝箱收益）。

出现“仍然顶怪追宝箱”时，优先调：
1. 提高 `MONSTER_DANGER_EXP_K`。
2. 提高 `MONSTER_DANGER_TERRAIN_COEF`。
3. 降低 `MONSTER_DANGER_HIGH_THRESHOLD`。
4. 适度提高 `RISK_PENALTY_COEF` 或 `RISK_WORSE_PENALTY_UNIT`。

## 5. 短验 A/B（120k）通过条件

A/B：
1. A：`temporal_enable=false`
2. B：`temporal_enable=true`，`temporal_sampling_mode=mixed21`，`temporal_input_mode=compressed_obs`

至少满足 3/5：
1. `early_jump_step` 提前 `>= 25%`
2. `post500_survival_rate` 提升 `>= 20pct`
3. `return_path_caught_rate` 下降 `>= 35%`
4. `danger_treasure_chase_rate` 下降 `>= 30%`
5. `reward` 移动均值提升 `>= 10%`

非回归：
1. `stuck_event_rate`、`corner_stuck_duration` 不得劣于基线 `> 5%`

## 6. 全量重训停止规则

连续 3 个窗口（每窗 50k）都满足时可停：
1. `reward` 移动均值改善 `< 2%`
2. `|policy_loss| < 0.01`
3. `entropy` 移动窗变化 `< 5%`

