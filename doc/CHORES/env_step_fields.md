# env.step() 可读信息全字段清单（GorgeChase）

## 结论（本项目训练代码实际接口）

训练工作流里实际调用形式是：

```python
env_reward, env_obs = env.step(hero_actions)
```

也就是一次 `env.step()` 里，代码可以读到两块数据：

1. `env_reward`（当前帧得分信息）
2. `env_obs`（当前帧观测与终局状态）

---

## 字段总览（细化到基础类型 + example）

说明：

- `example` 是结合 `doc/md` 协议、默认配置和项目样本整理的代表性示例值。
- `[]` 表示数组元素类型。
- `?` 表示代码中出现但协议文档未显式列出的可选字段。

### A) `env_reward`

| 路径 | 基础类型 | example |
|---|---|---|
| `env_reward.frame_no` | `int` | `128` |
| `env_reward.env_id` | `string` | `"gorge_env_0001"` |
| `env_reward.reward` | `float` | `392.0` |

### B) `env_obs` 顶层

| 路径 | 基础类型 | example |
|---|---|---|
| `env_obs.env_id` | `string` | `"gorge_env_0001"` |
| `env_obs.frame_no` | `int` | `128` |
| `env_obs.terminated` | `bool` | `false` |
| `env_obs.truncated` | `bool` | `false` |

### C) `env_obs.observation`

| 路径 | 基础类型 | example |  |
|---|---|---|--|
| `env_obs.observation.step_no` | `int32` | `128` |  |
| `env_obs.observation.map_info` | `int32[][]` | `[[1,1,0],[1,1,1],[0,1,1]]`（真实是 `21x21`） | 1=可通行，0=障碍物 |
| `env_obs.observation.map_info[r][c]` | `int32` | `1` |  |
| `env_obs.observation.legal_act` | `bool[16]` | `[true,true,true,true,true,true,true,true,false,false,false,false,false,false,false,false]` |  |
| `env_obs.observation.legal_act[i]` | `bool` | `true` |  |
| `env_obs.observation.legal_action`（别名） | `bool[16] / number[16]` | `[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]` |  |
| `env_obs.observation.legal_action[i]` | `bool / number` | `1` |  |

### D) `env_obs.observation.frame_state`

| 路径 | 基础类型 | example |
|---|---|---|
| `env_obs.observation.frame_state.frame_no` | `int32` | `128` |

#### D.1 `heroes`（HeroState）

| 路径 | 基础类型 | example |  |
|---|---|---|--|
| `env_obs.observation.frame_state.heroes.hero_id` | `int32` | `1` |  |
| `env_obs.observation.frame_state.heroes.pos.x` | `int32` | `54` |  |
| `env_obs.observation.frame_state.heroes.pos.z` | `int32` | `76` |  |
| `env_obs.observation.frame_state.heroes.treasure_score` | `float` | `200.0` |  |
| `env_obs.observation.frame_state.heroes.step_score` | `float` | `192.0` |  |
| `env_obs.observation.frame_state.heroes.treasure_collected_count` | `int32` | `2` |  |
| `env_obs.observation.frame_state.heroes.flash_cooldown`（?） | `int/float` | `35` | 还剩多少步可以使用闪现，动态 |
| `env_obs.observation.frame_state.heroes.buff_remaining_time`（?） | `int/float` | `12` | buff还能持续多少秒，动态 |

#### D.2 `monsters[]`（MonsterState）

| 路径 | 基础类型 | example |  |
|---|---|---|--|
| `env_obs.observation.frame_state.monsters` | `array` | `[ {...} ]` |  |
| `env_obs.observation.frame_state.monsters[i].monster_id` | `int32` | `14` |  |
| `env_obs.observation.frame_state.monsters[i].pos.x` | `int32` | `19` |  |
| `env_obs.observation.frame_state.monsters[i].pos.z` | `int32` | `89` |  |
| `env_obs.observation.frame_state.monsters[i].start_pos.x` | `int32` | `96` |  |
| `env_obs.observation.frame_state.monsters[i].start_pos.z` | `int32` | `96` |  |
| `env_obs.observation.frame_state.monsters[i].hero_l2_distance` | `int32` | `0` | 与英雄的欧氏距离桶编号（0-5）。128×128 地图均匀划分：桶范围：0=[0,30), 1=[30,60), 2=[60,90), 3=[90,120), 4=[120,150), 5=[150,180] |
| `env_obs.observation.frame_state.monsters[i].hero_relative_direction` | `int32` | `6` | 怪物相对于英雄的方位（0-8）：0=重叠/无效，1=东，2=东北，3=北，4=西北，5=西，6=西南，7=南，8=东南 |
| `env_obs.observation.frame_state.monsters[i].speed` | `int32` | `1` |  |
| `env_obs.observation.frame_state.monsters[i].monster_interval` | `int32` | `300` |  |
| `env_obs.observation.frame_state.monsters[i].is_in_view`（?） | `int/bool` | `1`（不可见时常见 `0`） | 是否可以拾取 |

#### D.3 `organs[]`（宝箱/buff）

| 路径 | 基础类型 | example |  |
|---|---|---|--|
| `env_obs.observation.frame_state.organs` | `array` | `[ {...} ]` |  |
| `env_obs.observation.frame_state.organs[i].sub_type` | `int32` | `1`（宝箱）或 `2`（buff） |  |
| `env_obs.observation.frame_state.organs[i].config_id` | `int32` | `3` |  |
| `env_obs.observation.frame_state.organs[i].status` | `int32` | `1` |  |
| `env_obs.observation.frame_state.organs[i].pos.x` | `int32` | `61` |  |
| `env_obs.observation.frame_state.organs[i].pos.z` | `int32` | `83` |  |
| `env_obs.observation.frame_state.organs[i].hero_l2_distance` | `int32` | `2` | 与英雄的欧氏距离桶编号（0-5）。128×128 地图均匀划分：桶范围：0=[0,30), 1=[30,60), 2=[60,90), 3=[90,120), 4=[120,150), 5=[150,180 |
| `env_obs.observation.frame_state.organs[i].hero_relative_direction` | `int32` | `8` |  |

### E) `env_obs.observation.env_info`

| 路径 | 基础类型 | example |
|---|---|---|
| `env_obs.observation.env_info.total_score` | `float` | `392.0` |
| `env_obs.observation.env_info.step_no` | `int32` | `128` |
| `env_obs.observation.env_info.step_score` | `float` | `192.0` |
| `env_obs.observation.env_info.pos.x` | `int32` | `54` |
| `env_obs.observation.env_info.pos.z` | `int32` | `76` |
| `env_obs.observation.env_info.treasure_score` | `float` | `200.0` |
| `env_obs.observation.env_info.treasure_id` | `int32[]` | `[1,4,7,9,10,12,15,18]` |
| `env_obs.observation.env_info.treasure_id[i]` | `int32` | `7` |
| `env_obs.observation.env_info.monster_interval` | `int32` | `300` |
| `env_obs.observation.env_info.total_map` | `int32` | `10` |
| `env_obs.observation.env_info.map_random` | `int32` | `0` |
| `env_obs.observation.env_info.max_step` | `int32` | `1000` |
| `env_obs.observation.env_info.finished_steps` | `int32` | `128` |
| `env_obs.observation.env_info.flash_count` | `int32` | `2` |
| `env_obs.observation.env_info.flash_cooldown` | `int32` | `100` |
| `env_obs.observation.env_info.total_buff` | `int32` | `2` |
| `env_obs.observation.env_info.collected_buff` | `int32` | `1` |
| `env_obs.observation.env_info.buff_refresh_time` | `int32` | `200` |
| `env_obs.observation.env_info.total_treasure` | `int32` | `10` |
| `env_obs.observation.env_info.treasures_collected` | `int32` | `2` |
| `env_obs.observation.env_info.monster_speed` | `int32` | `1` |

### F) `env_obs.extra_info`

| 路径 | 基础类型 | example |
|---|---|---|
| `env_obs.extra_info.map_id` | `int32` | `3` |
| `env_obs.extra_info.result_code` | `int32` | `0` |
| `env_obs.extra_info.result_message` | `string` | `"success"` |
| `env_obs.extra_info.frame_state.frame_no` | `int32` | `128` |
| `env_obs.extra_info.frame_state.heroes.hero_id` | `int32` | `1` |
| `env_obs.extra_info.frame_state.heroes.pos.x` | `int32` | `54` |
| `env_obs.extra_info.frame_state.heroes.pos.z` | `int32` | `76` |
| `env_obs.extra_info.frame_state.monsters[i].monster_id` | `int32` | `14` |
| `env_obs.extra_info.frame_state.monsters[i].pos.x` | `int32` | `19` |
| `env_obs.extra_info.frame_state.monsters[i].pos.z` | `int32` | `89` |
| `env_obs.extra_info.frame_state.organs[i].sub_type` | `int32` | `1` |
| `env_obs.extra_info.frame_state.organs[i].status` | `int32` | `1` |
| `env_obs.extra_info.frame_state.organs[i].pos.x` | `int32` | `61` |
| `env_obs.extra_info.frame_state.organs[i].pos.z` | `int32` | `83` |

---

## 异常返回（需要注意）

环境通信异常时，`env_obs` 可能只有错误信息（没有 `observation`）：

```json
{
  "extra_info": {
    "result_code": -6,
    "result_message": "failed to get env_info during step, error: ..."
  }
}
```

---

## 附：文档中的底层 RL 标准接口（非本项目训练代码直接使用）

`doc/md/taa_rl_fw__rl_env.md` 里还定义了标准形式：

```python
frame_no, _obs, score, terminated, truncated, _state = env.step(act, stop_game=False)
```

本项目代码通过环境封装层转换为 `env_reward, env_obs` 两段结构后再使用。

