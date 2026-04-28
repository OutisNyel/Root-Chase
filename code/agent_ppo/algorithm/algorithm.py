#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""PPO loss with hidden-state Mamba inputs and auxiliary monster heads."""

import os
import time

import torch
import torch.nn.functional as F

from agent_ppo.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.parameters = [
            p for pg in self.optimizer.param_groups for p in pg["params"]
        ]
        self.logger = logger
        self.monitor = monitor

        self.label_size = Config.ACTION_NUM
        self.value_num = Config.VALUE_NUM
        self.var_beta = Config.BETA_START
        self.vf_coef = Config.VF_COEF
        self.clip_param = Config.CLIP_PARAM

        self.last_report_monitor_time = 0
        self.train_step = 0

    def learn(self, list_sample_data):
        obs = self._sequence_field(
            list_sample_data, "obs", Config.DIM_OF_OBSERVATION
        )
        legal_action = self._sequence_field(
            list_sample_data, "legal_action", Config.ACTION_NUM
        )
        act = self._sequence_field(list_sample_data, "act", 1)
        old_prob = self._sequence_field(
            list_sample_data, "prob", Config.ACTION_NUM
        )
        reward = self._sequence_field(
            list_sample_data, "reward", Config.VALUE_NUM
        )
        advantage = self._sequence_field(
            list_sample_data, "advantage", Config.VALUE_NUM
        )
        old_value = self._sequence_field(
            list_sample_data, "value", Config.VALUE_NUM
        )
        reward_sum = self._sequence_field(
            list_sample_data, "reward_sum", Config.VALUE_NUM
        )
        monster_pos_target = self._sequence_field(
            list_sample_data, "monster_pos_target", 2
        )
        monster_pos_mask = self._sequence_field(
            list_sample_data, "monster_pos_mask", 1
        )
        monster_dist_target = self._sequence_field(
            list_sample_data, "monster_dist_target", 1
        )
        monster_dist_mask = self._sequence_field(
            list_sample_data, "monster_dist_mask", 1
        )
        seq_mask = self._sequence_field(
            list_sample_data, "seq_mask", 1, default=[1.0]
        )

        batch_stats = self._sanitize_cpu_batch(
            obs=obs,
            legal_action=legal_action,
            act=act,
            old_prob=old_prob,
            reward=reward,
            advantage=advantage,
            old_value=old_value,
            reward_sum=reward_sum,
            monster_pos_target=monster_pos_target,
            monster_pos_mask=monster_pos_mask,
            monster_dist_target=monster_dist_target,
            monster_dist_mask=monster_dist_mask,
            seq_mask=seq_mask,
        )

        obs = obs.to(self.device)
        legal_action = legal_action.to(self.device)
        act = act.to(self.device)
        old_prob = old_prob.to(self.device)
        reward = reward.to(self.device)
        advantage = advantage.to(self.device)
        old_value = old_value.to(self.device)
        reward_sum = reward_sum.to(self.device)
        monster_pos_target = monster_pos_target.to(self.device)
        monster_pos_mask = monster_pos_mask.to(self.device)
        monster_dist_target = monster_dist_target.to(self.device)
        monster_dist_mask = monster_dist_mask.to(self.device)
        seq_mask = seq_mask.to(self.device)
        self._cuda_sync("data_to_device")

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        logits, value_pred, aux_pred, sequence_info, train_index = (
            self._forward_sequence_batch(obs, seq_mask)
        )
        self._cuda_sync("forward_sequence")

        legal_action = self._active_flatten(legal_action, train_index)
        act = self._active_flatten(act, train_index)
        old_prob = self._active_flatten(old_prob, train_index)
        reward = self._active_flatten(reward, train_index)
        advantage = self._active_flatten(advantage, train_index)
        old_value = self._active_flatten(old_value, train_index)
        reward_sum = self._active_flatten(reward_sum, train_index)
        monster_pos_target = self._active_flatten(monster_pos_target, train_index)
        monster_pos_mask = self._active_flatten(monster_pos_mask, train_index)
        monster_dist_target = self._active_flatten(
            monster_dist_target, train_index
        ).view(-1)
        monster_dist_mask = self._active_flatten(monster_dist_mask, train_index).view(
            -1
        )
        advantage_raw_mean = advantage.mean()
        advantage_raw_std = advantage.std(unbiased=False)
        if Config.ADVANTAGE_NORMALIZE and advantage.numel() > 1:
            advantage = (advantage - advantage_raw_mean) / advantage_raw_std.clamp_min(
                Config.ADVANTAGE_NORM_EPS
            )

        total_loss, info = self._compute_loss(
            logits=logits,
            value_pred=value_pred,
            aux_pred=aux_pred,
            legal_action=legal_action,
            old_action=act,
            old_prob=old_prob,
            advantage=advantage,
            old_value=old_value,
            reward_sum=reward_sum,
            monster_pos_target=monster_pos_target,
            monster_pos_mask=monster_pos_mask,
            monster_dist_target=monster_dist_target,
            monster_dist_mask=monster_dist_mask,
        )
        self._cuda_sync("compute_loss")

        kl_limit = float(Config.TARGET_KL) * float(Config.KL_EARLY_STOP_MULTIPLIER)
        kl_early_stop = bool(info["approx_kl"].detach().item() > kl_limit)
        if kl_early_stop:
            # approx_kl is measured before optimizer.step(). If it is already too high,
            # this batch is stale or actor/learner distributions are mismatched.
            if self.logger:
                self.logger.warning(
                    "[train] skip optimizer_step: approx_kl=%.6f > %.6f"
                    % (info["approx_kl"].detach().item(), kl_limit)
                )
            grad_norm_value = 0.0
        else:
            total_loss.backward()
            self._cuda_sync("backward")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters, Config.GRAD_CLIP_RANGE
            )
            self.optimizer.step()
            self._cuda_sync("optimizer_step")
            grad_norm_value = float(
                grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm
            )
        self.train_step += 1

        results = {
            "total_loss": round(total_loss.item(), 4),
            "value_loss": round(info["value_loss"].item(), 4),
            "policy_loss": round(info["policy_loss"].item(), 4),
            "entropy_loss": round(info["entropy_loss"].item(), 4),
            "aux_pos_loss": round(info["aux_pos_loss"].item(), 4),
            "aux_dist_loss": round(info["aux_dist_loss"].item(), 4),
            "approx_kl": round(info["approx_kl"].item(), 6),
            "clip_frac": round(info["clip_frac"].item(), 6),
            "grad_norm": round(grad_norm_value, 6),
            "kl_early_stop": int(kl_early_stop),
            "advantage_mean": round(advantage_raw_mean.item(), 6),
            "advantage_std": round(advantage_raw_std.item(), 6),
            "reward": round(reward.mean().item(), 4),
            "sequence_segments": sequence_info["segments"],
            "sequence_mean_len": round(sequence_info["mean_len"], 4),
            "sequence_max_len": sequence_info["max_len"],
            "sequence_active_max_len": sequence_info["active_max_len"],
            "bad_action_count": batch_stats.get("bad_action_count", 0),
            "bad_legal_mask_count": batch_stats.get("bad_legal_mask_count", 0),
            "bad_old_prob_count": batch_stats.get("bad_old_prob_count", 0),
            "bad_finite_count": batch_stats.get("bad_finite_count", 0),
        }

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            if self.logger:
                self.logger.info(
                    f"[train] total_loss:{results['total_loss']} "
                    f"policy_loss:{results['policy_loss']} "
                    f"value_loss:{results['value_loss']} "
                    f"entropy:{results['entropy_loss']} "
                    f"aux_pos:{results['aux_pos_loss']} "
                    f"aux_dist:{results['aux_dist_loss']} "
                    f"kl:{results['approx_kl']} "
                    f"clip_frac:{results['clip_frac']} "
                    f"grad_norm:{results['grad_norm']} "
                    f"kl_early_stop:{results['kl_early_stop']} "
                    f"adv_mean:{results['advantage_mean']} "
                    f"adv_std:{results['advantage_std']} "
                    f"seq_mean_len:{results['sequence_mean_len']} "
                    f"seq_max_len:{results['sequence_max_len']} "
                    f"seq_active_max_len:{results['sequence_active_max_len']} "
                    f"bad_action:{results['bad_action_count']} "
                    f"bad_legal:{results['bad_legal_mask_count']} "
                    f"bad_prob:{results['bad_old_prob_count']} "
                    f"bad_finite:{results['bad_finite_count']}"
                )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now
        return results

    def _sanitize_cpu_batch(
        self,
        obs,
        legal_action,
        act,
        old_prob,
        reward,
        advantage,
        old_value,
        reward_sum,
        monster_pos_target,
        monster_pos_mask,
        monster_dist_target,
        monster_dist_mask,
        seq_mask,
    ):
        stats = {
            "bad_action_count": 0,
            "bad_legal_mask_count": 0,
            "bad_old_prob_count": 0,
            "bad_finite_count": 0,
        }
        if not Config.LEARNER_VALIDATE_BATCH:
            return stats

        expected = (
            ("obs", obs, Config.DIM_OF_OBSERVATION),
            ("legal_action", legal_action, Config.ACTION_NUM),
            ("act", act, 1),
            ("old_prob", old_prob, Config.ACTION_NUM),
            ("reward", reward, Config.VALUE_NUM),
            ("advantage", advantage, Config.VALUE_NUM),
            ("old_value", old_value, Config.VALUE_NUM),
            ("reward_sum", reward_sum, Config.VALUE_NUM),
            ("monster_pos_target", monster_pos_target, 2),
            ("monster_pos_mask", monster_pos_mask, 1),
            ("monster_dist_target", monster_dist_target, 1),
            ("monster_dist_mask", monster_dist_mask, 1),
            ("seq_mask", seq_mask, 1),
        )
        for name, tensor, inner_dim in expected:
            if tensor.dim() != 3 or tensor.shape[-1] != inner_dim:
                raise ValueError(
                    f"{name} must be [B,T,{inner_dim}], got {tuple(tensor.shape)}"
                )
            if name != "act":
                count = self._replace_nonfinite_cpu(name, tensor)
                stats["bad_finite_count"] += count

        seq_mask.copy_((seq_mask > 0.5).float())
        active = seq_mask[:, :, 0] > 0.5

        action_value = act[:, :, 0]
        rounded_action = action_value.round()
        valid_action = (
            torch.isfinite(action_value)
            & ((rounded_action - action_value).abs() <= 1.0e-4)
            & (rounded_action >= 0)
            & (rounded_action < Config.ACTION_NUM)
        )
        bad_action = active & ~valid_action
        bad_action_count = int(bad_action.sum().item())
        if bad_action_count:
            stats["bad_action_count"] = bad_action_count
            self._handle_batch_issue(
                "invalid action ids found; dropping those active timesteps "
                f"count={bad_action_count} min={float(action_value.min().item()):.4f} "
                f"max={float(action_value.max().item()):.4f}"
            )
            seq_active = seq_mask[:, :, 0]
            seq_active[bad_action] = 0.0

        safe_action = torch.where(
            torch.isfinite(rounded_action), rounded_action, torch.zeros_like(rounded_action)
        ).clamp(0, Config.ACTION_NUM - 1)
        act[:, :, 0].copy_(safe_action)

        active = seq_mask[:, :, 0] > 0.5
        if int(active.sum().item()) <= 0:
            raise ValueError("empty sequence batch after input validation")

        legal_action.copy_((legal_action > 0.5).float())
        bad_legal = active & (legal_action.sum(dim=-1) <= 0.0)
        bad_legal_count = int(bad_legal.sum().item())
        if bad_legal_count:
            stats["bad_legal_mask_count"] = bad_legal_count
            self._handle_batch_issue(
                "all-zero legal action masks found; replacing with all-legal "
                f"count={bad_legal_count}"
            )
            legal_action[bad_legal] = 1.0

        old_prob.clamp_(min=0.0)
        prob_sum = old_prob.sum(dim=-1, keepdim=True)
        bad_prob = active & (prob_sum[:, :, 0] <= 0.0)
        bad_prob_count = int(bad_prob.sum().item())
        if bad_prob_count:
            stats["bad_old_prob_count"] = bad_prob_count
            self._handle_batch_issue(
                "invalid old action probabilities found; replacing with uniform "
                f"count={bad_prob_count}"
            )
            old_prob[bad_prob] = 1.0 / Config.ACTION_NUM
            prob_sum = old_prob.sum(dim=-1, keepdim=True)

        good_prob = active & ~bad_prob
        if int(good_prob.sum().item()) > 0:
            old_prob[good_prob] = old_prob[good_prob] / prob_sum[good_prob].clamp_min(
                1.0e-9
            )
        return stats

    def _replace_nonfinite_cpu(self, name, tensor):
        bad = ~torch.isfinite(tensor)
        count = int(bad.sum().item())
        if count <= 0:
            return 0
        self._handle_batch_issue(
            f"non-finite values in {name}; replacing with zero count={count}"
        )
        tensor.copy_(torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0))
        return count

    def _handle_batch_issue(self, message):
        if Config.LEARNER_STRICT_BATCH_VALIDATION:
            raise ValueError(message)
        if self.logger:
            self.logger.warning(f"[batch_validate] {message}")

    def _cuda_sync(self, stage):
        if not Config.LEARNER_CUDA_SYNC_DEBUG:
            return
        if self.device is None:
            return
        if torch.device(self.device).type == "cuda":
            try:
                torch.cuda.synchronize(torch.device(self.device))
            except RuntimeError as exc:
                raise RuntimeError(f"CUDA failure after {stage}") from exc

    def _stack_field(self, list_sample_data, field_name, default=None):
        values = []
        for sample in list_sample_data:
            value = getattr(sample, field_name, None)
            if value is None:
                value = default
            if torch.is_tensor(value):
                tensor = value.float()
            else:
                tensor = torch.as_tensor(value, dtype=torch.float32)
            values.append(tensor)
        return torch.stack(values)

    def _sequence_field(self, list_sample_data, field_name, inner_dim, default=None):
        tensor = self._stack_field(list_sample_data, field_name, default=default)
        if tensor.dim() == 1:
            tensor = tensor.view(tensor.shape[0], 1)
        batch_size = tensor.shape[0]
        flat_dim = int(tensor.numel() / max(batch_size, 1))
        tensor = tensor.view(batch_size, flat_dim)
        window_len = int(Config.MAMBA_TBPTT_LEN)
        if flat_dim == inner_dim * window_len:
            return tensor.view(batch_size, window_len, inner_dim)
        if flat_dim == inner_dim:
            return tensor.view(batch_size, 1, inner_dim)
        if inner_dim > 0 and flat_dim % inner_dim == 0:
            return tensor.view(batch_size, flat_dim // inner_dim, inner_dim)
        raise ValueError(
            f"field {field_name} dim {flat_dim} is not divisible by {inner_dim}"
        )

    def _active_flatten(self, sequence_tensor, train_index):
        flat = sequence_tensor.reshape(-1, sequence_tensor.shape[-1])
        return flat.index_select(0, train_index)

    def _forward_sequence_batch(self, obs, seq_mask):
        batch_size, sequence_len, _ = obs.shape
        active_mask = seq_mask[:, :, 0] > 0.5
        train_index = active_mask.reshape(-1).nonzero(as_tuple=False).view(-1)
        if train_index.numel() <= 0:
            raise ValueError("empty sequence batch: all seq_mask values are zero")

        logits_seq, value_seq, aux_seq = self.model.forward_sequence(obs, seq_mask)
        logits = self._active_flatten(logits_seq, train_index)
        value = self._active_flatten(value_seq, train_index)
        aux = {
            "monster_pos": self._active_flatten(aux_seq["monster_pos"], train_index),
            "monster_dist": self._active_flatten(aux_seq["monster_dist"], train_index),
        }
        active_lengths = active_mask.sum(dim=1)
        segment_lengths = [
            int(x)
            for x in active_lengths.detach().cpu().tolist()
            if int(x) > 0
        ]
        sequence_info = {
            "segments": len(segment_lengths),
            "mean_len": float(sum(segment_lengths)) / max(1, len(segment_lengths)),
            "max_len": int(sequence_len),
            "active_max_len": int(max(segment_lengths) if segment_lengths else 0),
        }
        return logits, value, aux, sequence_info, train_index

    def _compute_loss(
        self,
        logits,
        value_pred,
        aux_pred,
        legal_action,
        old_action,
        old_prob,
        advantage,
        old_value,
        reward_sum,
        monster_pos_target,
        monster_pos_mask,
        monster_dist_target,
        monster_dist_mask,
    ):
        self._validate_loss_inputs(logits, value_pred, aux_pred, old_action, old_prob)
        prob_dist = self._masked_softmax(logits, legal_action)

        action_index = old_action[:, 0].round().long()
        one_hot = F.one_hot(action_index, self.label_size).float()
        new_prob = (one_hot * prob_dist).sum(1, keepdim=True).clamp(1e-9)
        old_action_prob = (one_hot * old_prob).sum(1, keepdim=True).clamp(1e-9)
        ratio = new_prob / old_action_prob
        approx_kl = (torch.log(old_action_prob) - torch.log(new_prob)).mean()
        clip_frac = (torch.abs(ratio - 1.0) > self.clip_param).float().mean()
        policy_loss1 = -ratio * advantage
        policy_loss2 = (
            -ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * advantage
        )
        policy_loss = torch.maximum(policy_loss1, policy_loss2).mean()

        value_clip = old_value + (value_pred - old_value).clamp(
            -Config.VALUE_CLIP_PARAM, Config.VALUE_CLIP_PARAM
        )
        value_loss = (
            0.5
            * torch.maximum(
                torch.square(reward_sum - value_pred),
                torch.square(reward_sum - value_clip),
            ).mean()
        )

        entropy = (-prob_dist * torch.log(prob_dist.clamp(1e-9, 1))).sum(1).mean()
        entropy_loss = -entropy
        aux_pos_loss = self._masked_position_loss(
            pred_pos=aux_pred["monster_pos"],
            target_pos=monster_pos_target,
            mask=monster_pos_mask,
        )
        aux_dist_loss = self._masked_cross_entropy(
            logits=aux_pred["monster_dist"],
            target=monster_dist_target,
            mask=monster_dist_mask,
        )

        total_loss = (
            self.vf_coef * value_loss
            + policy_loss
            + self.var_beta * entropy_loss
            + Config.AUX_MONSTER_POS_COEF * aux_pos_loss
            + Config.AUX_MONSTER_DIST_COEF * aux_dist_loss
        )
        if Config.LEARNER_VALIDATE_MODEL_OUTPUTS and not torch.isfinite(total_loss):
            raise FloatingPointError("non-finite PPO total_loss")

        return total_loss, {
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
            "aux_pos_loss": aux_pos_loss,
            "aux_dist_loss": aux_dist_loss,
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
        }

    def _validate_loss_inputs(self, logits, value_pred, aux_pred, old_action, old_prob):
        if not Config.LEARNER_VALIDATE_MODEL_OUTPUTS:
            return
        for name, tensor in (
            ("logits", logits),
            ("value_pred", value_pred),
            ("old_prob", old_prob),
            ("aux_monster_pos", aux_pred["monster_pos"]),
            ("aux_monster_dist", aux_pred["monster_dist"]),
        ):
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"non-finite tensor before PPO loss: {name}")

        action_value = old_action[:, 0]
        rounded_action = action_value.round()
        valid_action = (
            torch.isfinite(action_value)
            & ((rounded_action - action_value).abs() <= 1.0e-4)
            & (rounded_action >= 0)
            & (rounded_action < self.label_size)
        )
        if not valid_action.all():
            bad = action_value[~valid_action]
            raise ValueError(
                "invalid action ids before one_hot: "
                f"count={bad.numel()} min={float(bad.min().item()):.4f} "
                f"max={float(bad.max().item()):.4f}"
            )

    def _masked_position_loss(self, pred_pos, target_pos, mask):
        if mask.sum() <= 0:
            return pred_pos.sum() * 0.0
        loss = F.smooth_l1_loss(pred_pos, target_pos, reduction="none").sum(
            dim=1, keepdim=True
        )
        # Apply extra weight when monster is visible / 怪物可见时增加额外权重
        weighted_mask = mask * Config.AUX_LOSS_VISIBLE_MONSTER_WEIGHT
        return (loss * weighted_mask).sum() / mask.sum().clamp_min(1.0)

    def _masked_cross_entropy(self, logits, target, mask):
        if mask.sum() <= 0:
            return logits.sum() * 0.0
        target = target.long().clamp(0, logits.shape[-1] - 1)
        loss = F.cross_entropy(logits, target, reduction="none")
        # Apply extra weight when monster is visible / 怪物可见时增加额外权重
        weighted_mask = mask * Config.AUX_LOSS_VISIBLE_MONSTER_WEIGHT
        return (loss * weighted_mask).sum() / mask.sum().clamp_min(1.0)

    def _masked_softmax(self, logits, legal_action):
        legal = (legal_action > 0.5).float()
        empty = legal.sum(dim=1, keepdim=True) <= 0.0
        legal = torch.where(empty, torch.ones_like(legal), legal)
        masked_logits = logits.masked_fill(legal <= 0.0, -1.0e9)
        return F.softmax(masked_logits, dim=1)
