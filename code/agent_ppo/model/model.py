#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Neural network model for hidden-state PPO-Mamba Gorge Chase."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent_ppo.conf.conf import Config


def _make_linear(in_features, out_features, gain=1.0):
    layer = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


def _make_conv(in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
    padding = dilation if kernel_size == 3 else kernel_size // 2
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=False,
    )
    nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
    return conv


class ConvGNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            _make_conv(in_channels, out_channels, 3, stride=stride, dilation=dilation),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, channels, stride=1, dilation=1):
        super().__init__()
        self.conv1 = _make_conv(channels, channels, 3, stride=stride, dilation=dilation)
        self.gn1 = nn.GroupNorm(8, channels)
        self.conv2 = _make_conv(channels, channels, 3, stride=1, dilation=dilation)
        self.gn2 = nn.GroupNorm(8, channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1:
            self.downsample = nn.Sequential(
                _make_conv(
                    channels, channels, kernel_size=1, stride=stride, dilation=1
                ),
                nn.GroupNorm(8, channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(out + identity)


class HiddenStateMambaCell(nn.Module):
    """Single-step state-space cell matching h_t = A(x)h_{t-1} + B(x)x."""

    def __init__(self, model_dim):
        super().__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.input_proj = _make_linear(model_dim, model_dim)
        self.a_proj = _make_linear(model_dim, model_dim)
        self.b_proj = _make_linear(model_dim, model_dim)
        self.c_proj = _make_linear(model_dim, model_dim)
        self.out_proj = _make_linear(model_dim, model_dim)

        nn.init.constant_(self.a_proj.bias, -1.0)

    def forward(self, x, h_in):
        x_norm = self.norm(x)
        x_term = F.silu(self.input_proj(x_norm))
        a_gate = torch.sigmoid(self.a_proj(x_norm))
        b_gate = torch.tanh(self.b_proj(x_norm))
        h_next = a_gate * h_in + (1.0 - a_gate) * (b_gate * x_term)
        c_gate = torch.sigmoid(self.c_proj(x_norm))
        z = self.out_proj(c_gate * h_next) + x
        return z, h_next


class Model(nn.Module):
    """Local/global CNN + scalar MLP + true hidden-state Mamba + PPO heads."""

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_ppo_hidden_mamba"
        self.device = device

        c = Config.CONV_CHANNEL

        self.local_stem = ConvGNReLU(Config.LOCAL_CHANNELS, c)
        self.local_blocks = nn.Sequential(
            ResBlock(c, stride=1, dilation=1),
            ResBlock(c, stride=1, dilation=2),
            ResBlock(c, stride=1, dilation=1),
        )

        self.global_stem = ConvGNReLU(Config.GLOBAL_CHANNELS, c, stride=1)
        self.global_blocks = nn.Sequential(
            ResBlock(c, stride=2, dilation=1),
            ResBlock(c, stride=2, dilation=1),
            ResBlock(c, stride=2, dilation=1),
            ResBlock(c, stride=1, dilation=2),
            ResBlock(c, stride=1, dilation=1),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        scalar_dims = [Config.SCALAR_DIM] + Config.SCALAR_MLP_DIMS
        scalar_layers = []
        for in_dim, out_dim in zip(scalar_dims[:-1], scalar_dims[1:]):
            scalar_layers.extend([_make_linear(in_dim, out_dim), nn.ReLU(inplace=True)])
        self.scalar_mlp = nn.Sequential(*scalar_layers)

        visual_dims = [c + c] + Config.VISUAL_FUSION_DIMS
        visual_layers = []
        for in_dim, out_dim in zip(visual_dims[:-1], visual_dims[1:]):
            visual_layers.extend([_make_linear(in_dim, out_dim), nn.ReLU(inplace=True)])
        self.visual_fusion = nn.Sequential(*visual_layers)

        state_input_dim = Config.VISUAL_FUSION_DIMS[-1] + Config.SCALAR_MLP_DIMS[-1]
        state_dims = [state_input_dim] + Config.VIEW_FUSION_DIMS
        state_layers = []
        for in_dim, out_dim in zip(state_dims[:-1], state_dims[1:]):
            state_layers.extend([_make_linear(in_dim, out_dim), nn.ReLU(inplace=True)])
        self.state_fusion = nn.Sequential(*state_layers)

        self.mamba = HiddenStateMambaCell(Config.MAMBA_HIDDEN_DIM)
        self.post_mamba = nn.Sequential(
            nn.LayerNorm(Config.MAMBA_HIDDEN_DIM),
            _make_linear(Config.MAMBA_HIDDEN_DIM, Config.MAMBA_HIDDEN_DIM),
            nn.SiLU(inplace=True),
        )

        self.head_mv = _make_linear(
            Config.MAMBA_HIDDEN_DIM, Config.MOVE_ACTION_NUM, gain=0.01
        )
        self.head_flash = _make_linear(
            Config.MAMBA_HIDDEN_DIM, Config.FLASH_ACTION_NUM, gain=0.01
        )
        self.critic_head = _make_linear(
            Config.MAMBA_HIDDEN_DIM, Config.VALUE_NUM, gain=0.01
        )
        self.head_pos_monster = _make_linear(Config.MAMBA_HIDDEN_DIM, 2)
        self.head_dist_monster = _make_linear(Config.MAMBA_HIDDEN_DIM, 6)
        move_dirs = torch.tensor(
            [
                [0.0, 1.0],
                [-1.0, 1.0],
                [-1.0, 0.0],
                [-1.0, -1.0],
                [0.0, -1.0],
                [1.0, -1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        move_dirs = move_dirs / move_dirs.norm(dim=1, keepdim=True).clamp_min(1.0)
        self.register_buffer("move_prior_dirs", move_dirs, persistent=False)

        self.local_flat_dim = Config.LOCAL_FLAT_DIM
        self.global_flat_dim = Config.GLOBAL_FLAT_DIM
        self.scalar_dim = Config.SCALAR_DIM
        self.hidden_dim = Config.MAMBA_HIDDEN_DIM

    def _split_obs(self, obs):
        local_end = self.local_flat_dim
        global_end = local_end + self.global_flat_dim
        scalar_end = global_end + self.scalar_dim
        hidden_end = scalar_end + self.hidden_dim

        local = obs[:, :local_end].reshape(
            -1,
            Config.LOCAL_CHANNELS,
            Config.LOCAL_MAP_SIZE,
            Config.LOCAL_MAP_SIZE,
        )
        global_map = obs[:, local_end:global_end].reshape(
            -1,
            Config.GLOBAL_CHANNELS,
            Config.GLOBAL_MAP_SIZE,
            Config.GLOBAL_MAP_SIZE,
        )
        scalar = obs[:, global_end:scalar_end]
        h_in = obs[:, scalar_end:hidden_end]
        return local, global_map, scalar, h_in

    def forward(self, obs, inference=False):
        del inference
        x, h_in, move_prior, flash_prior = self._encode_obs(obs)
        z, h_next = self.mamba(x, h_in)
        z = self.post_mamba(z)
        logits, value, aux = self._heads(z, move_prior, flash_prior)
        return logits, value, aux, h_next

    def forward_sequence(self, obs_seq, seq_mask):
        """Batch visual encoding for [B, T, obs], then unroll hidden-state Mamba."""
        if obs_seq.dim() != 3:
            raise ValueError(f"obs_seq must be [B, T, D], got {tuple(obs_seq.shape)}")
        if seq_mask.dim() == 2:
            seq_mask = seq_mask.unsqueeze(-1)
        if seq_mask.dim() != 3 or seq_mask.shape[:2] != obs_seq.shape[:2]:
            raise ValueError(
                f"seq_mask must match obs_seq [B, T], got {tuple(seq_mask.shape)}"
            )

        batch_size, sequence_len, obs_dim = obs_seq.shape
        flat_obs = obs_seq.reshape(batch_size * sequence_len, obs_dim)
        x_flat, h_flat, move_prior_flat, flash_prior_flat = self._encode_obs(flat_obs)

        x = x_flat.view(batch_size, sequence_len, Config.MAMBA_HIDDEN_DIM)
        h_from_obs = h_flat.view(batch_size, sequence_len, Config.MAMBA_HIDDEN_DIM)
        move_prior = move_prior_flat.view(
            batch_size, sequence_len, Config.MOVE_ACTION_NUM
        )
        flash_prior = flash_prior_flat.view(
            batch_size, sequence_len, Config.FLASH_ACTION_NUM
        )

        h = h_from_obs[:, 0, :]
        active = (seq_mask[:, :, 0] > 0.5).to(dtype=x.dtype)
        logits_out = []
        value_out = []
        aux_pos_out = []
        aux_dist_out = []

        for step_idx in range(sequence_len):
            z_t, h_next = self.mamba(x[:, step_idx, :], h)
            z_t = self.post_mamba(z_t)
            logits_t, value_t, aux_t = self._heads(
                z_t,
                move_prior[:, step_idx, :],
                flash_prior[:, step_idx, :],
            )
            logits_out.append(logits_t)
            value_out.append(value_t)
            aux_pos_out.append(aux_t["monster_pos"])
            aux_dist_out.append(aux_t["monster_dist"])

            active_t = active[:, step_idx : step_idx + 1]
            h = h_next * active_t + h * (1.0 - active_t)

        return (
            torch.stack(logits_out, dim=1),
            torch.stack(value_out, dim=1),
            {
                "monster_pos": torch.stack(aux_pos_out, dim=1),
                "monster_dist": torch.stack(aux_dist_out, dim=1),
            },
        )

    def _encode_obs(self, obs):
        local_obs, global_obs, scalar_obs, h_in = self._split_obs(obs)

        local_feat = self.local_stem(local_obs)
        local_feat = self.local_blocks(local_feat)
        local_feat = self.gap(local_feat).flatten(1)

        global_feat = self.global_stem(global_obs)
        global_feat = self.global_blocks(global_feat)
        global_feat = self.gap(global_feat).flatten(1)

        feat_view = self.visual_fusion(torch.cat([local_feat, global_feat], dim=1))
        scalar_feat = self.scalar_mlp(scalar_obs)
        x = self.state_fusion(torch.cat([feat_view, scalar_feat], dim=1))
        move_prior, flash_prior = self._action_prior(local_obs)
        return x, h_in, move_prior, flash_prior

    def _heads(self, z, move_prior, flash_prior):
        logits = torch.cat(
            [
                self.head_mv(z) + move_prior,
                self.head_flash(z) + flash_prior,
            ],
            dim=1,
        )
        value = self.critic_head(z)
        aux = {
            "monster_pos": torch.sigmoid(self.head_pos_monster(z)),
            "monster_dist": self.head_dist_monster(z),
        }
        return logits, value, aux

    def _action_prior(self, local_obs):
        batch_size = local_obs.shape[0]
        move_prior = local_obs.new_zeros((batch_size, Config.MOVE_ACTION_NUM))
        flash_prior = local_obs.new_full(
            (batch_size, Config.FLASH_ACTION_NUM),
            float(Config.ACTION_PRIOR_FLASH_BASE_LOGIT),
        )

        center = Config.LOCAL_MAP_SIZE // 2
        passable = local_obs[:, 0]
        for idx, (dr, dc) in enumerate(
            ((0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1))
        ):
            target_r = center + dr
            target_c = center + dc
            blocked = (passable[:, target_r, target_c] <= 0.5).float()
            move_prior[:, idx] += Config.ACTION_PRIOR_WALL_LOGIT * blocked

        monster = local_obs[:, 3]
        monster_mass = monster.flatten(1).sum(dim=1)
        has_monster = (monster_mass > 0.0).float()
        if has_monster.sum() <= 0:
            return move_prior, flash_prior

        coords = torch.arange(
            Config.LOCAL_MAP_SIZE, device=local_obs.device, dtype=local_obs.dtype
        )
        row_offsets = (coords - float(center)).view(1, -1, 1)
        col_offsets = (coords - float(center)).view(1, 1, -1)
        denom = monster_mass.clamp_min(1.0)
        row_delta = (monster * row_offsets).flatten(1).sum(dim=1) / denom
        col_delta = (monster * col_offsets).flatten(1).sum(dim=1) / denom
        monster_dist = torch.sqrt(row_delta.square() + col_delta.square()).clamp_min(
            1.0e-6
        )
        away = torch.stack([-row_delta / monster_dist, -col_delta / monster_dist], dim=1)
        escape_score = torch.matmul(away, self.move_prior_dirs.t())

        move_weight = (
            (Config.ACTION_PRIOR_MONSTER_ESCAPE_RADIUS - monster_dist)
            / Config.ACTION_PRIOR_MONSTER_ESCAPE_RADIUS
        ).clamp(0.0, 1.0)
        move_weight = move_weight * has_monster
        move_prior += (
            Config.ACTION_PRIOR_MONSTER_ESCAPE_LOGIT
            * move_weight.unsqueeze(1)
            * escape_score
        )

        flash_weight = (
            (Config.ACTION_PRIOR_FLASH_DANGER_RADIUS - monster_dist)
            / Config.ACTION_PRIOR_FLASH_DANGER_RADIUS
        ).clamp(0.0, 1.0)
        flash_weight = flash_weight * has_monster
        flash_prior += (
            Config.ACTION_PRIOR_FLASH_ESCAPE_LOGIT
            * flash_weight.unsqueeze(1)
            * escape_score
        )
        return move_prior, flash_prior

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
