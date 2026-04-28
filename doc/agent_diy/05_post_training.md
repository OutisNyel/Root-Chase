# 05 Post-Training (Fine-Tune)

This page describes how to continue training from a pretrained checkpoint in `agent_diy`.

## 1. Framework preload entry

The framework preload switch is still controlled by `code/conf/configure_app.toml`:

- `preload_model = true`
- `preload_model_dir = "{agent_name}/ckpt"` (or your own model directory)
- `preload_model_id = 1000` (example)

When enabled, the framework calls business `agent.load_model(...)` and then continues training from that model.

## 2. DIY post-train knobs

`code/agent_diy/conf/conf.py` now provides post-training controls:

- `POST_TRAIN_ENABLE`: turn on fine-tune behavior
- `POST_TRAIN_MODEL_FILE`: optional explicit checkpoint file path; overrides `path/id` when set
- `POST_TRAIN_ALLOW_PARTIAL_LOAD`: allow loading only matching keys/shapes
- `POST_TRAIN_STRICT_LOAD`: strict load mode when partial load is disabled
- `POST_TRAIN_LOAD_OPTIMIZER`: load optimizer state from checkpoint if present
- `POST_TRAIN_FORCE_LR`: override optimizer LR after loading
- `POST_TRAIN_FREEZE_PREFIXES`: parameter name prefixes to freeze (for staged fine-tune)
- `POST_TRAIN_FREEZE_STEPS`: how many train steps to keep frozen

All switches default to disabled or no-op, so historical DIY behavior is unchanged.

## 3. Checkpoint format compatibility

`agent_diy.agent.Agent.load_model` supports:

- pure PyTorch `state_dict`
- dict with `model_state_dict`
- dict with `state_dict`
- dict with `model` or `network`

If `POST_TRAIN_LOAD_OPTIMIZER = true`, it also tries:

- `optimizer_state_dict`
- `optimizer`

## 4. Custom loader hook

`agent_diy.agent.Agent` now includes:

```python
def load_custom_model(self, model_file_path):
    return torch.load(model_file_path, map_location=self.device)
```

If you need a custom format (for example encrypted or packed checkpoints), implement your own logic in this method.

## 5. Typical fine-tune recipe

1. Put pretrained model under `agent_diy/ckpt/` or set `POST_TRAIN_MODEL_FILE`.
2. Set `preload_model = true` in `configure_app.toml`.
3. Enable `POST_TRAIN_ENABLE = true`.
4. Set smaller LR with `POST_TRAIN_FORCE_LR` (example `1e-4`).
5. Optionally freeze backbone prefixes for early stabilization:
   - `POST_TRAIN_FREEZE_PREFIXES = ("local_stem", "local_blocks", "global_stem", "global_blocks")`
   - `POST_TRAIN_FREEZE_STEPS = 200`

After freeze steps are reached, parameters are automatically unfrozen and full-model training continues.
