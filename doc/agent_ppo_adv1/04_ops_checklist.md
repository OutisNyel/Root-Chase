# 04 运维与检查清单

## 1. 接入检查

1. `code/conf/algo_conf_gorge_chase.toml` 存在 `[ppo_adv1]` 且 agent/workflow 指向 `agent_ppo_adv1`。
2. `code/train_test.py` 中 `algorithm_name_list` 包含 `ppo_adv1`。
3. `agent_ppo_adv1/workflow/train_workflow.py` 读取 `agent_ppo_adv1/conf/train_env_conf.toml`。

## 2. 容器内最小烟测

1. 启动短训练，确认能进入 `reset -> predict -> step -> learn` 闭环。
2. 监控日志确认存在 `policy_loss/value_loss/entropy/reward`。
3. 观测检查：
   - `obs` 长度固定 `11626`
   - `legal_action` 长度固定 `16`
   - `logits` 输出维度 `16`
   - `value` 输出维度 `1`
4. 行为检查：动作值始终落在 `[0, 15]`。

## 3. 问题定位顺序

1. 看 `obs` 长度是否与 `Config.DIM_OF_OBSERVATION` 一致。
2. 看 `legal_action` 是否被错误解析成全 0 或仅两维可用。
3. 看 reward 是否被 clip 频繁打满（`-20` 或 `20`）。
4. 看 `global_monster_risk` 与 `global_explored` 是否随步数更新。
5. 再看 PPO 比例 `ratio` 是否出现极端值导致损失震荡。

## 4. 常见配置改动注意

- 改通道数时，必须同步更新：
  - `conf.py` 的 `LOCAL_CHANNELS/GLOBAL_CHANNELS/SCALAR_DIM`
  - `preprocessor.py` 对应张量堆叠顺序
  - `model.py` 输入切分与分支输入通道
- 改动作数时，必须同步更新：
  - `ACTION_NUM`
  - `SampleData.legal_action/prob`
  - actor 输出维度
  - legal mask 逻辑

