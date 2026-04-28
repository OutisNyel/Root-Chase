# lzy_ppo_mamba_refactor_3 训练评估

评估时间：2026-04-25 04:33  
数据来源：`train_monitor/tensorboard/events.out.tfevents.1777062557.Outis.41980.0`  
转换 manifest：`train_monitor/tensorboard/aisrv_rl_helper_manifest.json`

## 数据范围

| 项目 | 数值 |
|---|---:|
| TensorBoard scalar tags | 50 |
| 转换记录数 | 513 |
| gameover 记录 | 426 |
| training 记录 | 87 |
| wall time | 2026-04-25 03:36:12 ~ 04:29:10 |
| 持续时间 | 52.97 分钟 |
| 源 aisrv helper 日志 | pid304, pid310 |

注意：本次 TensorBoard 文件来自 `monitor/aisrv_rl_helper_to_tensorboard.py`，只包含 helper 日志中的 `[GAMEOVER]` 和 `training_metrics is ...`。event 中没有 `approx_kl`、`clip_frac`、`grad_norm`、`aux_pos_loss`、`aux_dist_loss`。用户随后补充了在线监控截图，截图中包含这些 PPO 诊断项；本报告的 PPO 诊断部分基于该截图做定性评估。

## 总体结论

这次训练没有形成可用策略。426 局全部失败，`success=0`、`caught_rate=1`、`post500_survival_rate=0` 全程不变。

虽然 `episode_steps` 和 `sim_score` 的均值在后段略有抬升，但中位数没有改善，且 reward 没有变好。更像是偶发长局拉高了均值，而不是策略稳定进步。

loss 明显下降，尤其是 value loss 从高位降到低位；在线监控截图显示 `approx_kl` 已不再长期异常，`clip_frac` 也不再顶 100%。这说明前面关于 PPO 概率同源/样本新鲜度的修复方向基本有效，但对局质量没有同步提升。当前主要问题已经从“PPO 更新异常”转为“奖励、行为信用分配和探索目标没有把训练推向成功策略”。

## 关键指标

### 结果指标

| 指标 | 全量 | 前 100 局 | 后 100 局 | 判断 |
|---|---:|---:|---:|---|
| success | 0 / 426 | 0 | 0 | 无成功样本 |
| caught_rate | 100% | 100% | 100% | 全部被抓 |
| post500_survival_rate | 0% | 0% | 0% | 没有活过 500 步 |
| total_reward mean | -12.295 | -12.220 | -12.423 | 没有改善 |
| episode_steps mean | 38.71 | 33.66 | 44.52 | 均值略升 |
| episode_steps median | 21 | 20 | 19 | 稳定性未改善 |
| episode_steps max | 304 | 302 | 280 | 偶发长局，不稳定 |
| sim_score mean | 62.29 | 53.49 | 70.78 | 分数略升但不转化为成功 |

生存长度分布：

| 条件 | 局数 | 占比 |
|---|---:|---:|
| steps >= 50 | 78 | 18.31% |
| steps >= 100 | 39 | 9.15% |
| steps >= 150 | 25 | 5.87% |
| steps >= 200 | 12 | 2.82% |
| steps >= 300 | 3 | 0.70% |
| steps >= 500 | 0 | 0.00% |

### 探索与资源

| 指标 | 全量均值 | 前 100 局 | 后 100 局 | 判断 |
|---|---:|---:|---:|---|
| explore_cells mean | 116.32 | 103.80 | 116.82 | 略升，但波动大 |
| explore_cells median | 85.5 | 83 | 78 | 中位数未提升 |
| treasure_count total | 18 | 3 | 4 | 获取率很低 |
| buff_count total | 3 | 0 | 1 | 几乎没有拿 buff |

426 局里只有 18 局拿到宝箱，3 局拿到 buff。当前策略没有稳定学会资源路线或资源优先级。

### 行为指标

| 指标 | 全量均值 | 早期 20% | 后期 20% | 变化 |
|---|---:|---:|---:|---:|
| move_action_rate | 0.909 | 0.894 | 0.913 | +0.019 |
| flash_rate | 0.091 | 0.106 | 0.087 | -0.019 |
| flash_success_rate | 0.061 | 0.100 | 0.063 | -0.037 |
| still_rate | 0.182 | 0.167 | 0.216 | +0.049 |
| position_changed_rate | 0.818 | 0.833 | 0.784 | -0.049 |
| wall_or_still_after_move_rate | 0.194 | 0.173 | 0.231 | +0.059 |

行为质量在后期没有变好。`wall_or_still_after_move_rate` 上升，`position_changed_rate` 下降，说明无效移动或卡墙类行为变多。`flash_success_rate` 很低，并且后期低于早期，闪现没有被学成有效逃生动作。

补充统计：

| 条件 | 局数 | 占比 |
|---|---:|---:|
| flash_success_rate > 0 | 32 | 7.51% |
| wall_or_still_after_move_rate > 0.3 | 82 | 19.25% |

### 怪物压力与 legal mask

| 指标 | 均值 | 说明 |
|---|---:|---|
| avg_min_monster_bfs | 6.20 | 距怪物距离大体稳定，没有明显拉开 |
| visible_monster_count | 0.72 | 大部分对局中经常能看到怪物 |
| legal_action_count | 8.41 | 平均只比 8 个移动动作多一点 |
| legal_move_count | 8.00 | 移动动作始终全部合法 |
| legal_flash_count | 0.41 | 闪现可用窗口较少 |

`legal_move_count` 恒为 8，说明当前移动 legal mask 不是墙体规避信号；撞墙/原地问题需要靠 reward、特征或动作后果学习解决。

## 学习指标

| 指标 | 首值 | 末值 | 早期 20% 均值 | 后期 20% 均值 | 判断 |
|---|---:|---:|---:|---:|---|
| total_loss | 18.33 | 0.77 | 16.22 | 1.41 | 明显下降 |
| policy_loss | 4.39 | 0.38 | 4.11 | 0.66 | 明显下降 |
| value_loss | 13.96 | 0.49 | 12.19 | 0.85 | 明显下降 |
| entropy_loss | -2.01 | -2.00 | -2.036 | -2.042 | 基本稳定 |
| training/algorithm/reward | -10.48 | -9.88 | -10.915 | -10.961 | 无实质改善 |

`entropy_loss` 在 `-2.0` 附近稳定。结合平均 legal action 数约 8.4，这不是明显的熵崩塌；策略仍有较高随机性，但随机探索没有转化为有效逃生、资源获取或成功样本。

因此本轮主要问题不是“策略过早确定化”，而是：学习信号没有把探索行为塑造成有效策略，value/policy loss 降低后，对局指标仍停留在失败分布。

### 在线 PPO 诊断截图补充

用户补充的在线监控截图覆盖 03:38 ~ 04:33，包含 TensorBoard event 未转换出的诊断项。基于截图读数：

| 指标 | 截图表现 | 结论 |
|---|---|---|
| `approx_kl` | 大多在 0 ~ 0.06，少数峰值约 0.08 | 已脱离此前 KL 约 2 的异常状态 |
| `clip_frac` | 多数在 0.2 ~ 0.4，峰值约 0.5 | 不再是 100%，PPO clip 异常基本解除 |
| `grad_norm` | 初期有接近 95 的尖峰，后续多在 10 ~ 35，末段约 8 ~ 20 | 梯度仍有波动，但没有持续爆炸 |
| `aux_pos_loss` | 初期快速从高位降到接近 0 | 位置辅助头很快拟合或信号变弱 |
| `aux_dist_loss` | 初期下降后接近 0 | 距离辅助头不再提供持续强梯度 |

修正判断：本轮不再像早期异常那样由 `old_prob/new_prob` 不同源导致 `clip_frac=100%` 和 `approx_kl≈2`。PPO 更新本身看起来已经回到可训练区间；但结果层仍然失败，说明下一步不应优先继续改 `clip/lr`，而应优先处理 reward、无效移动、闪现学习和资源目标。

## 框架计数与吞吐

| 指标 | 末值 |
|---|---:|
| train_global_step | 41619 |
| sample_receive_cnt | 93363 |
| predict_succ_cnt | 94675 |
| framework episode_cnt | 3066 |
| sample_production_and_consumption_ratio | 14.26 |

这里的 `sample_production_and_consumption_ratio` 来自 local helper，公式是 `(train_count - preload_model_train_count) / predict_count`。它可以作为训练/预测链路是否在增长的参考，但不是严格的“样本池生产消费平衡”。本次它从启动值上升到约 14.3，说明训练链路在跑，但不能据此判断 PPO 样本新鲜度。

`framework episode_cnt=3066` 与本报告中的 426 条 gameover 不一致是正常的：前者来自框架 training metrics 的累计计数，后者是这次 event 文件里实际转换到的 `[GAMEOVER]` 记录。

## 诊断

1. 结果层没有突破：成功率、500 步生存率全为 0，继续加长同配置训练的收益很低。
2. reward 没有改善：`total_reward` 后 100 局均值比前 100 局更低，说明训练没有朝目标行为稳定移动。
3. loss 下降不等于策略提升：value loss 下降主要表示拟合了当前失败轨迹分布。
4. 行为质量后期变差：卡墙/原地移动比例上升，位置变化率下降。
5. 闪现没有学会：使用率低、成功率低，且后期成功率下降。
6. 在线截图显示 PPO KL/clip 已恢复到相对正常区间，因此当前瓶颈不是 PPO 更新幅度失控，而是策略目标和行为学习没有对齐。

## 建议

优先级从高到低：

1. 先补齐离线监控闭环：让 TensorBoard 转换脚本也包含 `approx_kl`、`clip_frac`、`grad_norm`、`aux_pos_loss`、`aux_dist_loss`，避免下次只能靠截图补充。
2. 不建议在当前 checkpoint 上继续长训。当前 52 分钟内没有任何成功样本，后段 reward 和行为质量没有好转。
3. 下一轮先短跑验收：只跑 30~60 分钟，验收 `success/post500/median_steps/reward/wall_or_still_after_move_rate/flash_success_rate` 是否同时改善。
4. 暂不优先调低 `lr` 或 `clip`。截图显示 `approx_kl` 和 `clip_frac` 已不再异常，继续压低更新幅度可能只会让学习更慢。
5. 奖励侧需要更直接地压制无效移动：提高撞墙、移动后原地、连续原地的惩罚权重，或把 `wall_or_still_after_move_rate` 作为硬验收指标。
6. 闪现需要单独处理：当前稀疏可用且成功率低，建议增加闪现后距离改善/脱离危险的正奖励，同时惩罚无效闪现；必要时把闪现动作拆成独立 head 或先做规则/辅助监督引导。
7. 资源目标需要更强信号：宝箱 18/426、buff 3/426 太稀疏，若资源对最终成功关键，应增加“接近资源/可见资源/获取资源”的分阶段 shaping。

## 验收标准建议

下一轮训练至少满足以下条件，才认为方向有效：

| 指标 | 最低验收 |
|---|---:|
| success_rate | > 0 |
| post500_survival_rate | > 0 |
| last100 median episode_steps | 高于 first100 |
| last100 total_reward mean | 高于 first100 |
| wall_or_still_after_move_rate | 低于本轮后 100 局的 0.230 |
| flash_success_rate | 高于本轮全量 0.061 |
| treasure_count / episode | 高于本轮 0.042 |

如果这些结果指标没有改善，即使 loss 继续下降，也应判定为无效训练。
