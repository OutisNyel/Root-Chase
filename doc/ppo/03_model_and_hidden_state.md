# 03 模型与真隐状态

模型路径：`code/agent_ppo/model/model.py`。

2026-04-25 update: learner training uses `Model.forward_sequence(obs_seq, seq_mask)` to batch CNN/scalar/action-prior computation over `[B*T]` frames and only loop the hidden-state Mamba cell over `T=48`. Actor inference still uses the single-step `Model.forward(obs)` path.

## Lightweight Two-stage Fusion

The model input contract is now `global 6x32x32`. `Preprocessor` still keeps 128x128 global memory internally, then compresses it to 32x32 before concatenating obs. CNN width is raised to `CONV_CHANNEL = 64`.

Fusion is split into two stages:

```text
local_feat  = local_encoder(local_obs)
global_feat = global_encoder(global_obs)
feat_view   = visual_fusion(concat(local_feat, global_feat))
scalar_feat = scalar_mlp(scalar_obs)
x_mamba     = state_fusion(concat(feat_view, scalar_feat))
z_t, h_next = mamba(x_mamba, h_in)
```

## 模型总图

```mermaid
flowchart TB
    OBS["obs 9647"] --> SPLIT["_split_obs"]
    SPLIT --> L["local [B,7,21,21]"]
    SPLIT --> G["global [B,6,32,32]"]
    SPLIT --> S["scalar [B,288]"]
    SPLIT --> H["h_in [B,128]"]

    L --> LCNN["local CNN + GAP -> [B,64]"]
    G --> GCNN["global CNN + GAP -> [B,64]"]
    S --> SMLP["scalar MLP -> [B,96]"]

    LCNN --> VCAT["visual concat [B,128]"]
    GCNN --> VCAT
    VCAT --> VFUS["visual_fusion -> feat_view [B,64]"]
    VFUS --> SCAT["state concat [B,160]"]
    SMLP --> SCAT
    SCAT --> FUS["state_fusion -> x [B,128]"]

    FUS --> MAMBA["HiddenStateMambaCell"]
    H --> MAMBA
    MAMBA --> Z["z_t [B,128]"]
    MAMBA --> HN["h_next [B,128]"]

    Z --> MV["head_mv 8"]
    Z --> FL["head_flash 8"]
    MV --> LOGITS["concat logits 16"]
    FL --> LOGITS
    Z --> VAL["critic value 1"]
    Z --> POS["head_pos_monster 2"]
    Z --> DIST["head_dist_monster 6"]
```

## CNN 分支

```mermaid
flowchart LR
    L["local 7ch"] --> LS["ConvGNReLU"]
    LS --> LR1["ResBlock d=1"]
    LR1 --> LR2["ResBlock d=2"]
    LR2 --> LR3["ResBlock d=1"]
    LR3 --> LGAP["AdaptiveAvgPool -> 64"]

    G["global 6ch, 32x32"] --> GS["ConvGNReLU stride=1"]
    GS --> GR1["ResBlock stride=2"]
    GR1 --> GR2["ResBlock stride=2"]
    GR2 --> GR3["ResBlock stride=2"]
    GR3 --> GR5["ResBlock d=2"]
    GR5 --> GGAP["AdaptiveAvgPool -> 64"]
```

全局图输入是压缩后的 `32x32`，三次 stride=2 后进入 dilation block，再做 GAP。

## 真隐状态 Mamba Cell

实现目标：

```text
h_t = A(x_t) * h_{t-1} + B(x_t) * x_t
z_t = C(x_t) * h_t
```

代码中用门控近似：

```mermaid
flowchart TB
    X["x_t"] --> LN["LayerNorm"]
    LN --> A["a_gate = sigmoid(a_proj(x))"]
    LN --> B["b_gate = tanh(b_proj(x))"]
    LN --> U["x_term = silu(input_proj(x))"]
    H0["h_in"] --> MIX["h_next = a*h_in + (1-a)*(b*x_term)"]
    A --> MIX
    B --> MIX
    U --> MIX
    LN --> C["c_gate = sigmoid(c_proj(x))"]
    MIX --> ZM["c_gate * h_next"]
    C --> ZM
    ZM --> OUT["out_proj + residual x"]
    OUT --> Z["z_t"]
```

## Hidden State 传递

```mermaid
sequenceDiagram
    participant P as Preprocessor
    participant A as Agent
    participant M as Model

    A->>P: feature_process(env_obs, last_action, h_t)
    P-->>A: obs 包含 h_t
    A->>M: model(obs)
    M-->>A: logits, value, aux, h_next
    A->>A: self.hidden_state = h_next.detach()
```

训练时每个 `SampleData` 是一个非重叠 48 步窗口，携带 `seq_id/seq_pos/seq_mask/seq_len`。learner 会按窗口顺序做最多 `MAMBA_TBPTT_LEN = 48` 步 unroll；窗口首帧使用采样时固化在 `obs` 里的 `h_in`，后续帧使用当前模型产生的 `h_next` 继续反传，padding 帧不参与 loss。

## 输出头

| Head | 输出 | 用途 |
|---|---:|---|
| `head_mv` | 8 | 移动方向 logits |
| `head_flash` | 8 | 闪现方向 logits |
| `critic_head` | 1 | PPO value |
| `head_pos_monster` | 2 | 最近可见怪物归一化 `(x,z)` |
| `head_dist_monster` | 6 | 怪物距离桶分类 |

`head_mv` 和 `head_flash` 拼接成 16 维 logits，再由 legal action mask 做 masked softmax。
