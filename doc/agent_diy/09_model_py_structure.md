# 09 `model.py` 模型结构拆解（Code-Aligned）

本页严格对齐 `code/agent_diy/model/model.py` 当前实现，目标是让你在 3-5 分钟内回答三个问题：

1. `obs(12198)` 是如何被拆分并送入各分支的？
2. 当前帧分支与时序分支分别做了什么？
3. actor / critic 的最终输出来自哪些中间特征？

## 1. 整体结构（一图总览）

```mermaid
flowchart TB
    OBS["obs: [B,12198]"] --> SPLIT["_split_obs"]

    SPLIT --> L0["local_obs: [B,9,21,21]"]
    SPLIT --> G0["global_obs: [B,6,32,32]"]
    SPLIT --> S0["scalar_obs: [B,48]"]
    SPLIT --> T0["temporal_tokens: [B,21,96]"]
    SPLIT --> M0["temporal_mask: [B,21]"]

    L0 --> L1["Local stem + 4xResBlock + GAP -> [B,64]"]
    G0 --> G1["Global stem + 4xResBlock + GAP -> [B,64]"]
    S0 --> S1["Scalar MLP (48->256->128) -> [B,128]"]
    L1 --> Ca["Cat local feat, global feat, scalar feat. 64+64+128=256"]
    G1 --> Ca
    S1 --> Ca
    Ca --> C1["Current fusion MLP (256->512->256->128)"]
    C1 --> CF["current_feat: [B,128]"]

    T0 --> TP["Linear(96->128)"]
    M0 --> TM["mask"]
    TP --> MB["MambaBlock x2 + LayerNorm"]
    TM --> MB
    MB --> MP["masked last-token pooling"]
    MP --> TF["temporal_feat: [B,128]"]

    CF --> F0["concat -> [B,256]"]
    TF --> F0
    F0 --> F1["ResidualFusionBlock(256)"]
    F1 --> F2["LayerNorm + Linear(256->128) + SiLU"]
    F2 --> ACT["actor_head: [B,16] logits"]
    F2 --> VAL["critic_head: [B,1] value"]
```

## 2. 输入切分契约（`_split_obs`）

`Config` 对应关系：

| 片段 | 维度 | 计算方式 |
|---|---:|---|
| `base_obs` | `10161` | `FEATURE_LEN` |
| `temporal_flat` | `2016` | `TEMPORAL_SEQ_LEN(21) * TEMPORAL_TOKEN_DIM(96)` |
| `temporal_mask` | `21` | `TEMPORAL_MASK_DIM` |
| 总计 | `12198` | `10161 + 2016 + 21` |

`base_obs` 再拆为：

| 子块 | 形状 | 来源 |
|---|---|---|
| `local` | `[B,9,21,21]` | `LOCAL_CHANNELS`, `LOCAL_MAP_SIZE` |
| `global_map` | `[B,6,32,32]` | `GLOBAL_CHANNELS`, `GLOBAL_MAP_SIZE` |
| `scalar` | `[B,48]` | `SCALAR_DIM` |

## 3. 当前帧分支（空间瞬时决策）

```mermaid
flowchart LR
    L["local [B,9,21,21]"] --> LS["ConvGNReLU(9->64)"]
    LS --> LB["ResBlock(d=1)->ResBlock(d=2)->ResBlock(d=3)->ResBlock(d=1)"]
    LB --> LG["GAP + flatten -> [B,64]"]

    G["global [B,6,32,32]"] --> GS["ConvGNReLU(6->64)"]
    GS --> GB["ResBlock(s=2)->ResBlock(s=2)->ResBlock(d=2)->ResBlock(d=3)"]
    GB --> GG["GAP + flatten -> [B,64]"]

    S["scalar [B,48]"] --> SM["MLP: 48->256->128"]

    LG --> CAT["concat [B,256]"]
    GG --> CAT
    SM --> CAT
    CAT --> FM["current_fusion_mlp: 256->512->256->128"]
    FM --> OUT["current_feat [B,128]"]
```

直觉理解：
- `local` 分支更关注近场细节（走位/碰撞风险）。
- `global` 分支更关注远场布局（路径与追逐关系）。
- `scalar` 分支承载非图像状态量（CD、计数、规范化统计量等）。

## 4. 时序分支（Mamba）

```mermaid
flowchart LR
    X["tokens [B,21,96]"] --> P["Linear(96->128)"]
    M["mask [B,21]"] --> MX["x * mask"]
    P --> MX
    MX --> B1["MambaBlock #1"]
    B1 --> B2["MambaBlock #2"]
    B2 --> LN["LayerNorm(128)"]
    LN --> POOL["masked_last_token_pool"]
    POOL --> TO["temporal_feat [B,128]"]
```

`MambaBlock` 内部可视化（简化版）：

```mermaid
flowchart TB
    X["x: [B,T,D]"] --> N["LayerNorm"]
    N --> IP["in_proj -> split(u, gate)"]
    IP --> U["u -> SiLU -> depthwise Conv1d"]
    N --> D["delta_proj -> softplus(delta)"]
    N --> B["b_proj -> tanh(b_term)"]
    N --> C["c_proj -> c_term"]
    U --> SCAN["for t in [0..T-1]: state update"]
    D --> SCAN
    B --> SCAN
    C --> SCAN
    SCAN --> Y["y_t concat -> out_proj"]
    IP --> G["sigmoid(gate)"]
    Y --> MUL["y * gate"]
    G --> MUL
    MUL --> MASK["optional mask"]
    MASK --> RES["residual add"]
```

关键点：
- 这是纯 PyTorch 的轻量状态空间实现，时间维是显式 `for t` 扫描。
- `mask` 同时作用于输入和 block 输出，避免填充帧污染有效时序表征。
- 池化采用“最后一个有效 token”，对应策略决策最相关的最近历史状态。

## 5. 融合与输出头

融合路径：

1. `fused = concat(current_feat, temporal_feat)` 得到 `[B,256]`
2. 过 `ResidualFusionBlock(256)` 保留主干稳定性
3. 过 `fusion_output: LayerNorm -> Linear(256,128) -> SiLU`
4. 分别进入：
- `actor_head: Linear(128,16)`
- `critic_head: Linear(128,1)`

这意味着 actor 与 critic 共享完整 backbone，并直接基于融合特征分别输出策略与价值。

## 6. 代码锚点（便于对照）

- 初始化与模块装配：`Model.__init__`
- 输入切分：`Model._split_obs`
- 时序编码：`Model._encode_temporal`
- 有效 token 池化：`Model._masked_last_token_pool`
- 主前向：`Model.forward`

源码：`code/agent_diy/model/model.py`
