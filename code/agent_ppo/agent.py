#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Agent class for hidden-state PPO-Mamba Gorge Chase."""

import os

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.model.model import Model
from kaiwudrl.interface.agent import BaseAgent


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = device
        self.model = Model(device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=Config.INIT_LEARNING_RATE_START,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.algorithm = Algorithm(
            self.model, self.optimizer, self.device, logger, monitor
        )
        self.preprocessor = Preprocessor()
        self.hidden_state = np.zeros(Config.MAMBA_HIDDEN_DIM, dtype=np.float32)
        self.last_action = -1
        self.last_metrics = {}
        self.logger = logger
        self.monitor = monitor
        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs=None):
        del env_obs
        self.preprocessor.reset()
        self.hidden_state = np.zeros(Config.MAMBA_HIDDEN_DIM, dtype=np.float32)
        self.last_action = -1
        self.last_metrics = {}

    def observation_process(self, env_obs):
        feature, legal_action, reward, aux_info = self.preprocessor.feature_process(
            env_obs,
            self.last_action,
            self.hidden_state,
        )
        obs_data = ObsData(feature=list(feature), legal_action=legal_action)
        remain_info = {
            "reward": reward,
            "aux": aux_info,
            "metrics": aux_info.get("metrics", {}),
        }
        self.last_metrics = dict(remain_info["metrics"])
        return obs_data, remain_info

    def predict(self, list_obs_data):
        feature = list_obs_data[0].feature
        legal_action = list_obs_data[0].legal_action

        _, value, prob = self._run_model(feature, legal_action)

        # Keep rollout `old_prob` on the same policy basis the learner can
        # recompute: model logits plus legal-action mask. Runtime guards must
        # not rewrite training probabilities unless learner loss mirrors them.
        sample_prob = prob

        action = self._legal_sample(sample_prob, use_max=False)
        d_action = self._select_deterministic_action(prob, legal_action)

        return [
            ActData(
                action=[action],
                d_action=[d_action],
                prob=list(sample_prob),
                value=value,
            )
        ]

    def exploit(self, env_obs):
        obs_data, _ = self.observation_process(env_obs)
        act_data = self.predict([obs_data])
        return self.action_process(act_data[0], is_stochastic=False)

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        if path is None:
            if self.logger:
                self.logger.warning("save_model path is None, skip business save")
            return
        model_file_path = self._resolve_model_path(path, id)
        model_dir = os.path.dirname(model_file_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        state_dict_cpu = {
            k: v.clone().cpu() for k, v in self.model.state_dict().items()
        }
        torch.save(state_dict_cpu, model_file_path)
        if self.logger:
            self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        if path is None:
            return
        model_file_path = self._resolve_model_path(
            path, id, allow_unique_fallback=True
        )
        if not os.path.isfile(model_file_path):
            if self.logger:
                self.logger.warning(
                    f"model file not found, skip load: {model_file_path}"
                )
            return

        state_dict = torch.load(model_file_path, map_location=self.device)
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as exc:
            loaded = self._load_compatible_state_dict(state_dict)
            if self.logger:
                self.logger.warning(
                    f"partial model load from {model_file_path}, compatible_keys={loaded}, err={exc}"
                )
        if self.logger:
            self.logger.info(f"load model {model_file_path} successfully")

    def action_process(self, act_data, is_stochastic=True):
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return int(action[0])

    def _run_model(self, feature, legal_action):
        self.model.set_eval_mode()
        obs_tensor = torch.tensor(np.array([feature]), dtype=torch.float32).to(
            self.device
        )

        with torch.no_grad():
            logits, value, _, h_next = self.model(obs_tensor, inference=True)

        self.hidden_state = h_next.detach().cpu().numpy()[0].astype(np.float32)
        logits_np = logits.cpu().numpy()[0]
        value_np = value.cpu().numpy()[0]

        legal_action_np = np.array(legal_action, dtype=np.float32)
        prob = self._legal_soft_max(logits_np, legal_action_np)

        return logits_np, value_np, prob

    def _legal_soft_max(self, input_hidden, legal_action):
        legal_action = np.asarray(legal_action, dtype=np.float32)
        if legal_action.shape[0] != Config.ACTION_NUM or np.sum(legal_action) <= 0:
            legal_action = np.ones(Config.ACTION_NUM, dtype=np.float32)
        legal_action = (legal_action > 0.5).astype(np.float32)

        logits = np.asarray(input_hidden, dtype=np.float64)
        masked_logits = np.where(legal_action > 0.0, logits, -np.inf)
        max_logit = np.max(masked_logits)
        if not np.isfinite(max_logit):
            return legal_action / np.sum(legal_action)

        exp_logits = np.exp(np.clip(masked_logits - max_logit, -60.0, 60.0)) * legal_action
        denom = np.sum(exp_logits)
        if not np.isfinite(denom) or denom <= 0.0:
            return legal_action / np.sum(legal_action)
        return (exp_logits / denom).astype(np.float32)

    def _legal_sample(self, probs, use_max=False):
        probs = self._normalize_probs(probs)
        if use_max:
            return int(np.argmax(probs))
        return int(np.random.choice(len(probs), p=probs))

    def _select_deterministic_action(self, probs, legal_action):
        probs = self._normalize_probs(probs)
        planned = self._resource_override_action(legal_action)
        if (
            planned is not None
            and 0 <= planned < probs.shape[0]
            and probs[planned] > 0.0
        ):
            return int(planned)
        return int(np.argmax(probs))

    def _apply_runtime_action_guard(self, probs, legal_action):
        probs = self._normalize_probs(probs)
        legal = np.asarray(legal_action, dtype=np.float64)
        if legal.shape[0] != Config.ACTION_NUM or np.sum(legal) <= 0.0:
            legal = np.ones(Config.ACTION_NUM, dtype=np.float64)
        legal = (legal > 0.5).astype(np.float64)

        guarded = probs * legal
        if Config.FLASH_GUARD_ENABLED and not self._flash_allowed_now():
            no_flash = legal.copy()
            no_flash[Config.MOVE_ACTION_NUM :] = 0.0
            if np.sum(no_flash) > 0.0:
                guarded *= no_flash

        if Config.MONSTER_GUARD_ENABLED:
            monster_guard = self._monster_move_guard_mask(legal)
            if monster_guard is not None and np.sum(monster_guard * legal) > 0.0:
                guarded *= monster_guard

        total = np.sum(guarded)
        if not np.isfinite(total) or total <= 0.0:
            guarded = legal
            total = np.sum(guarded)
        return guarded / total

    def _apply_training_sample_filter(self, probs, legal_action=None):
        base = self._normalize_probs(probs).astype(np.float64)
        filtered = base.copy()

        if legal_action is not None and Config.RESOURCE_OVERRIDE_TRAIN_LOGIT > 0.0:
            planned = self._resource_override_action(legal_action)
            if (
                planned is not None
                and 0 <= planned < filtered.shape[0]
                and filtered[int(planned)] > 0.0
            ):
                active = filtered > 0.0
                logits = np.full_like(filtered, -np.inf, dtype=np.float64)
                logits[active] = np.log(np.clip(filtered[active], 1.0e-12, 1.0))
                logits[int(planned)] += float(Config.RESOURCE_OVERRIDE_TRAIN_LOGIT)
                max_logit = np.max(logits[active])
                adjusted = np.zeros_like(filtered)
                adjusted[active] = np.exp(logits[active] - max_logit)
                total = np.sum(adjusted)
                if np.isfinite(total) and total > 0.0:
                    filtered = adjusted / total

        positive_indices = np.flatnonzero(filtered > 0.0)
        if positive_indices.size <= 0:
            return base.astype(np.float32)

        top_k = int(max(0, Config.TRAIN_SAMPLE_TOP_K))
        if 0 < top_k < positive_indices.size:
            top_indices = positive_indices[
                np.argsort(filtered[positive_indices])[-top_k:]
            ]
            top_filtered = np.zeros_like(filtered)
            top_filtered[top_indices] = filtered[top_indices]
            filtered = top_filtered

        temperature = float(Config.TRAIN_SAMPLE_TEMPERATURE)
        if np.isfinite(temperature) and temperature > 0.0 and temperature != 1.0:
            active = filtered > 0.0
            logits = np.log(np.clip(filtered[active], 1.0e-12, 1.0)) / temperature
            logits -= np.max(logits)
            adjusted = np.zeros_like(filtered)
            adjusted[active] = np.exp(logits)
            filtered = adjusted

        total = np.sum(filtered)
        if not np.isfinite(total) or total <= 0.0:
            return base.astype(np.float32)
        return (filtered / total).astype(np.float32)

    def _flash_allowed_now(self):
        metrics = self.last_metrics or {}
        has_visible = float(metrics.get("has_visible_monster", 0.0)) > 0.5
        if not has_visible:
            return False
        min_bfs = float(metrics.get("min_monster_bfs", Config.BFS_SENTINEL))
        return min_bfs <= Config.FLASH_GUARD_DANGER_BFS

    def _monster_move_guard_mask(self, legal_action):
        metrics = self.last_metrics or {}
        has_visible = float(metrics.get("has_visible_monster", 0.0)) > 0.5
        if not has_visible:
            return None
        min_bfs = float(metrics.get("min_monster_bfs", Config.BFS_SENTINEL))
        if min_bfs > Config.MONSTER_GUARD_DANGER_BFS:
            return None

        hero_pos = getattr(self.preprocessor, "last_hero_pos", None)
        monsters = self._visible_monster_positions()
        if hero_pos is None or not monsters:
            return None

        hero_x, hero_z = hero_pos
        current_dist = self._min_l2_to_monsters(hero_x, hero_z, monsters)
        move_scores = []
        mask = np.ones(Config.ACTION_NUM, dtype=np.float64)
        for action_id in range(Config.MOVE_ACTION_NUM):
            if legal_action[action_id] <= 0.5:
                continue
            nx, nz = self._candidate_move_position(action_id, hero_x, hero_z)
            next_dist = self._min_l2_to_monsters(nx, nz, monsters)
            move_scores.append((action_id, next_dist))
            if next_dist + Config.MONSTER_GUARD_L2_EPS < current_dist:
                mask[action_id] = 0.0

        if not move_scores:
            return None

        if any(mask[action_id] > 0.0 for action_id, _ in move_scores):
            return mask

        best_dist = max(next_dist for _, next_dist in move_scores)
        for action_id, next_dist in move_scores:
            if next_dist + Config.MONSTER_GUARD_L2_EPS < best_dist:
                mask[action_id] = 0.0
            else:
                mask[action_id] = 1.0
        return mask

    def _visible_monster_positions(self):
        positions = []
        for monster in getattr(self.preprocessor, "last_visible_monsters", []) or []:
            if "_x" in monster and "_z" in monster:
                positions.append((float(monster["_x"]), float(monster["_z"])))
        return positions

    def _min_l2_to_monsters(self, x, z, monsters):
        return min(float(np.hypot(float(x) - mx, float(z) - mz)) for mx, mz in monsters)

    def _candidate_move_position(self, action_id, hero_x, hero_z):
        directions = (
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
            (0, -1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
        dz, dx = directions[int(action_id)]
        nz = int(np.clip(hero_z + dz, 0, Config.MAP_SIZE - 1))
        nx = int(np.clip(hero_x + dx, 0, Config.MAP_SIZE - 1))
        return nx, nz

    def _resource_override_action(self, legal_action):
        if not Config.RESOURCE_OVERRIDE_ENABLED or not self._safe_for_resource_override():
            return None

        legal = np.asarray(legal_action, dtype=np.float32)
        if legal.shape[0] != Config.ACTION_NUM:
            return None
        target = self._nearest_platform_target()
        if target is None:
            return None
        return self._best_move_towards(target, legal)

    def _safe_for_resource_override(self):
        metrics = self.last_metrics or {}
        has_visible = float(metrics.get("has_visible_monster", 0.0)) > 0.5
        if not has_visible:
            return True
        min_bfs = float(metrics.get("min_monster_bfs", Config.BFS_SENTINEL))
        return min_bfs >= Config.RESOURCE_OVERRIDE_SAFE_BFS

    def _nearest_platform_target(self):
        hero_pos = getattr(self.preprocessor, "last_hero_pos", None)
        if hero_pos is None:
            return None
        hero_x, hero_z = hero_pos
        frontier = self._frontier_target_map()

        if self._exploration_behind_schedule():
            target = self._nearest_cell(frontier, hero_x, hero_z, min_bfs=1.0)
            if target is not None:
                return target

        target = self._nearest_cell(
            getattr(self.preprocessor, "global_treasure_alive", None), hero_x, hero_z
        )
        if target is not None:
            return target

        try:
            buff_map = self.preprocessor._build_buff_availability_map()
        except Exception:
            buff_map = getattr(self.preprocessor, "global_buff_alive", None)
        target = self._nearest_cell(buff_map, hero_x, hero_z)
        if target is not None:
            return target

        return self._nearest_cell(frontier, hero_x, hero_z, min_bfs=1.0)

    def _exploration_behind_schedule(self):
        gap = float(getattr(self.preprocessor, "last_global_explore_gap", 0.0))
        return gap <= Config.RESOURCE_OVERRIDE_EXPLORE_GAP

    def _frontier_target_map(self):
        explored = getattr(self.preprocessor, "global_explored", None)
        passable = getattr(self.preprocessor, "global_passable", None)
        if explored is None or passable is None:
            return None
        return (1.0 - explored) * passable

    def _nearest_cell(self, target_map, hero_x, hero_z, min_bfs=0.0):
        if target_map is None:
            return None
        arr = np.asarray(target_map)
        if arr.shape != (Config.MAP_SIZE, Config.MAP_SIZE):
            return None
        target_mask = arr > 0.5
        hero_bfs = getattr(self.preprocessor, "last_hero_bfs_raw", None)
        if hero_bfs is not None:
            bfs = np.asarray(hero_bfs)
            if bfs.shape == arr.shape:
                reachable = (
                    target_mask
                    & np.isfinite(bfs)
                    & (bfs < Config.BFS_SENTINEL)
                    & (bfs >= float(min_bfs))
                )
                coords = np.argwhere(reachable)
                if coords.size > 0:
                    values = bfs[reachable]
                    idx = int(np.argmin(values))
                    z, x = coords[idx]
                    return int(x), int(z)

        coords = np.argwhere(target_mask)
        if coords.size <= 0:
            return None
        dz = coords[:, 0].astype(np.float32) - float(hero_z)
        dx = coords[:, 1].astype(np.float32) - float(hero_x)
        idx = int(np.argmin(dx * dx + dz * dz))
        return int(coords[idx, 1]), int(coords[idx, 0])

    def _best_move_towards(self, target, legal_action):
        hero_pos = getattr(self.preprocessor, "last_hero_pos", None)
        if hero_pos is None:
            return None
        hero_x, hero_z = hero_pos
        target_x, target_z = target
        passable = getattr(self.preprocessor, "global_passable", None)
        target_bfs = None
        if passable is not None:
            try:
                target_bfs = self.preprocessor._bfs_distance_map(
                    sources=[(int(target_z), int(target_x))],
                    passable=passable,
                    max_depth=Config.GLOBAL_BFS_THRESHOLD,
                )
            except Exception:
                target_bfs = None
        directions = (
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
            (0, -1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
        best_action = None
        best_score = None
        for action_id, (dz, dx) in enumerate(directions):
            if legal_action[action_id] <= 0.5:
                continue
            nz = int(np.clip(hero_z + dz, 0, Config.MAP_SIZE - 1))
            nx = int(np.clip(hero_x + dx, 0, Config.MAP_SIZE - 1))
            if passable is not None and passable[nz, nx] < 0.5:
                continue
            if target_bfs is not None and target_bfs[nz, nx] < Config.BFS_SENTINEL:
                score = float(target_bfs[nz, nx])
            else:
                score = (nx - target_x) * (nx - target_x) + (nz - target_z) * (
                    nz - target_z
                )
            if best_score is None or score < best_score:
                best_score = score
                best_action = action_id
        return best_action

    def _normalize_probs(self, probs):
        probs = np.asarray(probs, dtype=np.float64)
        probs = np.where(np.isfinite(probs) & (probs > 0.0), probs, 0.0)
        total = np.sum(probs)
        if probs.ndim != 1 or probs.shape[0] != Config.ACTION_NUM or total <= 0.0:
            return np.full(Config.ACTION_NUM, 1.0 / Config.ACTION_NUM, dtype=np.float64)
        return probs / total

    def _resolve_model_path(self, path, id, allow_unique_fallback=False):
        if isinstance(path, str) and path.endswith(".pkl"):
            return path
        if allow_unique_fallback and self._is_auto_model_id(id):
            return self._resolve_unique_model_path(str(path))
        return os.path.join(str(path), f"model.ckpt-{str(id)}.pkl")

    def _is_auto_model_id(self, id):
        return str(id).strip().lower() in ("", "0", "auto", "unique")

    def _resolve_unique_model_path(self, model_dir):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"model preload dir not found: {model_dir}")
        names = sorted(
            name
            for name in os.listdir(model_dir)
            if name.startswith("model.ckpt-") and name.endswith(".pkl")
        )
        if len(names) != 1:
            raise RuntimeError(
                f"expected exactly one preload pkl in {model_dir}, found {len(names)}: {names}"
            )
        return os.path.join(model_dir, names[0])

    def _load_compatible_state_dict(self, source_state_dict):
        if not isinstance(source_state_dict, dict):
            return 0
        target_state_dict = self.model.state_dict()
        filtered = {}
        for key, value in source_state_dict.items():
            if key in target_state_dict and tuple(
                target_state_dict[key].shape
            ) == tuple(value.shape):
                filtered[key] = value
        if filtered:
            self.model.load_state_dict(filtered, strict=False)
        return len(filtered)
