# 01 架构总览

![architecture pipeline](assets/pipeline.svg)

## 1. 训练闭环

1. `workflow/train_workflow.py` 调用 `env.reset`，读取 `agent_ppo_adv1/conf/train_env_conf.toml`。
2. `agent.observation_process` 调用 preprocessor，输出 `feature + legal_action + reward`。
3. `agent.predict` 用模型前向得到 `logits/value`，再按 legal mask 采样动作。
4. `env.step(action)` 返回下一帧观测，写入 `SampleData`。
5. `sample_process` 计算 GAE，`algorithm.learn` 执行 PPO 更新。

## 2. 模块职责

| 模块 | 职责 |
|---|---|
| `feature/preprocessor.py` | 全局记忆维护、局部/全局/标量特征构建、reward shaping |
| `model/model.py` | 对扁平观测反切分并进行双 CNN + MLP 融合 |
| `algorithm/algorithm.py` | legal mask softmax、PPO policy/value/entropy loss |
| `agent.py` | 预测、采样、learn 调用、模型读写 |
| `workflow/train_workflow.py` | 采样循环、样本打包、监控指标上报 |

## 3. 模型结构（实现对齐）

![feature layout](assets/feature_layout.svg)

- Local 分支：`ConvGNReLU(10->64)` + `ResBlock x4 (dilation: 1/2/3/1)` + GAP
- Global 分支：`ConvGNReLU(7->64)` + `ResBlock x4 (stride/downsample + dilation)` + GAP
- Scalar 分支：`MLP(48->256->128)`
- 融合层：`MLP((64+64+128)->512->256->128)`
- 输出头：`Actor(128->16)`，`Critic(128->1)`

## 4. 动作空间协议（16 维）

动作定义来自环境协议，`adv1` 在算法层完整使用 16 维掩码。

| 动作值 | 类型 | 方向 |
|---|---|---|
| 0 | 移动 | 右 |
| 1 | 移动 | 右上 |
| 2 | 移动 | 上 |
| 3 | 移动 | 左上 |
| 4 | 移动 | 左 |
| 5 | 移动 | 左下 |
| 6 | 移动 | 下 |
| 7 | 移动 | 右下 |
| 8 | 闪现 | 右 |
| 9 | 闪现 | 右上 |
| 10 | 闪现 | 上 |
| 11 | 闪现 | 左上 |
| 12 | 闪现 | 左 |
| 13 | 闪现 | 左下 |
| 14 | 闪现 | 下 |
| 15 | 闪现 | 右下 |

## 5. 接入点

- 算法注册：`code/conf/algo_conf_gorge_chase.toml` 的 `[ppo_adv1]`
- 训练入口可选项：`code/train_test.py` 的 `algorithm_name_list`

