# 第一次训练结果记录（默认 PPO）

## 1. 训练背景
- 训练算法：默认 `agent_ppo`（PPO）。
- 训练流程：`workflow` 持续与环境交互采样，并按固定间隔上报监控指标。
- 环境关键配置（来自 `train_env_conf.toml`）：
  - `map = [1..10]`
  - `map_random = false`
  - `max_step = 1000`
  - `treasure_count = 10`
  - `buff_count = 2`
  - `talent_cooldown = 100`
  - `monster_interval = 300`
  - `buff_cooldown = 200`

## 2. 数据来源
- `train_doc/Screenshot 2026-04-13 210533.png`（基础指标 + 部分环境指标）
- `train_doc/Screenshot 2026-04-13 210629.png`（环境指标）
- `train_doc/Screenshot 2026-04-13 210647.png`（算法指标）

训练时段从图上看约为 `2026-04-13 19:21` 到 `21:04`（43分钟）。

## 3. 基础指标表现（约值）
- `train_global_step`：线性上升到约 `1.1 万`，训练过程连续无中断迹象。
- `episode_cnt`：线性上升到约 `11.0~11.2 万`，对局采样充足。
- `predict_succ_cnt` 与 `sample_receive_cnt`：均线性上升到约 `440 万`，采样/接收链路稳定。
- `load_model_succ_cnt`：上升到约 `11 万`，模型加载正常。

## 4. 环境指标表现（约值）
- 地图相关：`total_map=10`、`map_random=0`，与配置一致（顺序使用 10 张训练图）。
- 步数相关：`max_steps=1000` 固定，`finished_steps` 大多在 `50~150` 区间，少量波动到更高值，说明多数对局较早结束。
- 得分相关：
  - `step_score` 波动较大，常见在 `50~180`，偶尔到 `250+`。
  - `treasure_score` 大多接近 `0`，偶尔出现小幅上升。
  - `total_score` 与 `step_score`趋势基本一致，说明当前得分主要来自生存步数而非宝箱。
- 宝箱/技能/Buff/怪物参数：
  - `total_treasure=10`、`flash_cooldown=100`、`total_buff=2`、`buff_refresh_time=200`、`monster_interval=300` 均为常量，配置生效。
  - `treasures_collected` 与 `collected_buff` 基本贴近 `0`，资源利用偏弱。

## 5. 算法指标表现（约值）
- `reward`：全程为负，约在 `-8` 附近波动（截图示例点：`-7.63558`）。
- `total_loss`：约 `15 -> 0.8~1.0`，明显下降后趋于平稳。在19:58（37分钟后）后趋于稳定
- `value_loss`：约 `12 -> 0.8~1.0`，与总损失同向收敛。在19:58（37分钟后）后趋于稳定
- `policy_loss`：约 `3.8 -> 0` 附近，后期略有负值波动。在19:58（37分钟后）后趋于稳定
- `entropy_loss`：约 `2.0 -> 0.75`，策略随机性下降，探索逐步减弱。

## 6. 第一次训练结论
- 从优化角度看：默认 PPO 的损失项收敛趋势正常，训练过程稳定。
- 从任务完成质量看：平均回报仍明显为负，单局结束步数偏短，宝箱和 buff 获取较少。
- 综合判断：第一次训练已完成“可稳定训练并收敛”，但策略仍处于早期阶段，当前主要问题是生存能力和资源获取能力不足。
