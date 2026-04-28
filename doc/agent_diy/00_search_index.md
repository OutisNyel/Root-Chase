# 00 检索索引（Search Index）

用于快速检索 `agent_diy` 文档与代码关键词，适合 `rg` / IDE 全局搜索。

## 1. 主题索引

| 主题 | 关键词（可直接检索） | 入口文档 |
|---|---|---|
| 怪物危险函数 | `danger_score` `MONSTER_DANGER_TRIGGER_DIST_NORM` `MONSTER_DANGER_EXP_K` `MONSTER_DANGER_TERRAIN_COEF` `MONSTER_DANGER_RISK_BLEND` | `03_reward_and_training.md` |
| 地形卷积风险 | `terrain_trap_score` `_terrain_trap_score` `global_passable` | `03_reward_and_training.md` |
| 可见怪物逻辑 | `is_in_view` `_update_monster_risk` `visible_count` | `03_reward_and_training.md` |
| 危险态追宝箱 | `danger_treasure_chase_rate` `high_risk_decision_count` `danger_treasure_chase_count` | `04_ops_checklist.md` |
| 回头被抓 | `return_path_caught_rate` `return_path_window` | `04_ops_checklist.md` |
| 卡死与乒乓 | `stuck_event_rate` `corner_stuck_duration` `PINGPONG_PENALTY` | `04_ops_checklist.md` |
| 闪现收益判定 | `REWARD_FLASH_GOOD` `REWARD_FLASH_BAD` `MONSTER_DANGER_HIGH_THRESHOLD` | `03_reward_and_training.md` |
| 动作风险过滤 | `_build_legal_action` `_apply_risk_aware_filter` `RISK_FILTER_DELTA` | `01_architecture.md` |
| 时序分支 | `TEMPORAL_ENABLE` `mixed21` `compressed_obs` `temporal_valid_mask` | `07_temporal_mamba.md` |
| 训练超参 | `PHASE2_START_TRAIN_STEP` `PHASE1_TRAIN_BATCH_SIZE` `PHASE2_TRAIN_BATCH_SIZE` | `06_retrain_playbook.md` |

## 2. 文件索引

- `doc/agent_diy/README.md`: 文档总览与阅读路径。
- `doc/agent_diy/01_architecture.md`: 全链路架构图与模块边界。
- `doc/agent_diy/03_reward_and_training.md`: 奖励塑形、危险函数与参数说明。
- `doc/agent_diy/04_ops_checklist.md`: 运维验收与排障流程。
- `doc/agent_diy/06_retrain_playbook.md`: 重训策略、调参顺序、回滚条件。
- `doc/agent_diy/08_legend_quick_ref.md`: 术语图例、指标方向、快速对照。

## 3. 代码定位索引

- `code/agent_diy/conf/conf.py`: 所有奖励和危险函数配置项。
- `code/agent_diy/feature/preprocessor.py`: `is_in_view`、`danger_score`、奖励计算与动作过滤。
- `code/agent_diy/model/model.py`: 当前帧分支 + 时序分支 + 融合结构。
- `code/agent_diy/algorithm/algorithm.py`: PPO 损失和两阶段训练超参切换。
- `code/agent_diy/workflow/train_workflow.py`: 采样、学习、指标上报链路。

