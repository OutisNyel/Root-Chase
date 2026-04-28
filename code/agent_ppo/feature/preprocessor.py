#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Feature preprocessing, global memory, BFS features, and reward shaping."""

from collections import deque
import time

import numpy as np

from agent_ppo.conf.conf import Config

_DIRECTIONS_8 = (
    (0, 1),  # east
    (-1, 1),  # north-east
    (-1, 0),  # north
    (-1, -1),  # north-west
    (0, -1),  # west
    (1, -1),  # south-west
    (1, 0),  # south
    (1, 1),  # south-east
)


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return int(default)


def _norm(v, v_max, v_min=0.0):
    if v_max <= v_min:
        return 0.0
    v = float(np.clip(_safe_float(v, 0.0), v_min, v_max))
    return (v - v_min) / (v_max - v_min)


def _direction_one_hot(direction_id):
    one_hot = np.zeros(8, dtype=np.float32)
    idx = _safe_int(direction_id, 0) - 1
    if 0 <= idx < 8:
        one_hot[idx] = 1.0
    return one_hot


def _relative_direction_id(src_x, src_z, dst_x, dst_z):
    dx = int(np.sign(dst_x - src_x))
    dz = int(np.sign(dst_z - src_z))
    if dx == 0 and dz == 0:
        return 0
    if dx > 0 and dz == 0:
        return 1
    if dx > 0 and dz < 0:
        return 2
    if dx == 0 and dz < 0:
        return 3
    if dx < 0 and dz < 0:
        return 4
    if dx < 0 and dz == 0:
        return 5
    if dx < 0 and dz > 0:
        return 6
    if dx == 0 and dz > 0:
        return 7
    return 8


def _l2_bucket(dx, dz):
    dist = float(np.sqrt(dx * dx + dz * dz))
    return int(np.clip(dist // 30.0, 0, 5))


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 1000
        self.buff_refresh_steps = Config.BUFF_COOLDOWN_DEFAULT

        self.global_passable = np.ones(
            (Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.float32
        )
        self.global_explored = np.zeros(
            (Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.float32
        )
        self.global_treasure_alive = np.zeros(
            (Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.float32
        )
        self.global_buff_alive = np.zeros(
            (Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.float32
        )
        self.global_buff_known = np.zeros(
            (Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.float32
        )
        self.global_buff_cd_remaining_norm = np.zeros(
            (Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.float32
        )

        self.known_treasure_pos = set()
        self.known_buff_pos = set()
        self.last_total_score = 0.0
        self.last_treasures_collected = 0
        self.last_collected_buff = 0
        self.last_hero_pos = None
        self.last_visible_monsters = []
        self.last_hero_bfs_raw = None
        self.last_global_explore_ratio = 0.0
        self.last_global_explore_target_ratio = 0.0
        self.last_global_explore_gap = 0.0
        self.last_danger_pos = None
        self.post_escape_until_step = 0
        self.last_danger_distance = None
        self.last_open_neighbor_count = None
        self.last_min_monster_bfs = None
        self.last_nearest_treasure_bfs = None
        self.last_nearest_buff_bfs = None
        self.last_nearest_unexplored_bfs = None
        self.last_explore_direction_features = np.zeros(6, dtype=np.float32)
        self.consecutive_still_count = 0
        self.position_history = deque(maxlen=Config.ANTI_LOOP_WINDOW_STEPS + 1)
        self.last_legal_action = None
        self.last_feature_timing = {}
        self.survival_milestones_paid = set()
        self.last_reward_debug = {
            "survival_bonus": 0.0,
            "safety_bonus": 0.0,
            "monster_bfs_reward": 0.0,
            "explore_approach_bonus": 0.0,
            "explore_vector_reward": 0.0,
            "global_explore_reward": 0.0,
            "resource_approach_bonus": 0.0,
            "post_escape_scout_reward": 0.0,
            "trail_reward": 0.0,
            "route_safety_reward": 0.0,
            "flash_reward": 0.0,
            "penalty_near_monster": 0.0,
            "penalty_still": 0.0,
            "penalty_caught": 0.0,
        }

    def feature_process(self, env_obs, last_action, hidden_state=None):
        perf_counter = time.perf_counter
        total_t0 = perf_counter()
        timing = {}

        section_t0 = total_t0
        observation = env_obs.get("observation", {})
        frame_state = observation.get("frame_state", {})
        env_info = observation.get("env_info", {})
        map_info = observation.get("map_info", None)
        legal_act_raw = observation.get(
            "legal_action", observation.get("legal_act", None)
        )

        self.step_no = _safe_int(
            observation.get("step_no", env_info.get("step_no", self.step_no)),
            self.step_no,
        )
        self.max_step = max(
            1, _safe_int(env_info.get("max_step", self.max_step), self.max_step)
        )
        self.buff_refresh_steps = max(
            1.0,
            _safe_float(
                env_info.get("buff_refresh_time", self.buff_refresh_steps),
                self.buff_refresh_steps,
            ),
        )

        hero = self._normalize_hero(frame_state.get("heroes", {}))
        hero_pos = hero.get("pos", {}) if isinstance(hero, dict) else {}
        if not isinstance(hero_pos, dict):
            hero_pos = {}
        env_pos = env_info.get("pos", {})
        if not isinstance(env_pos, dict):
            env_pos = {}
        hero_x = int(
            np.clip(
                _safe_int(hero_pos.get("x", env_pos.get("x", 0)), 0),
                0,
                Config.MAP_SIZE - 1,
            )
        )
        hero_z = int(
            np.clip(
                _safe_int(hero_pos.get("z", env_pos.get("z", 0)), 0),
                0,
                Config.MAP_SIZE - 1,
            )
        )

        monsters = self._normalize_entity_list(frame_state.get("monsters", []))
        organs = self._normalize_entity_list(frame_state.get("organs", []))
        timing["parse"] = (perf_counter() - section_t0) * 1000.0

        section_t0 = perf_counter()
        prev_explored = self.global_explored.copy()
        self._decay_dynamic_maps()
        new_explored_cells = self._stitch_local_map(map_info, hero_x, hero_z)
        delta_treasure, delta_buff = self._update_organs_and_counters(
            organs, env_info, hero_x, hero_z
        )
        total_score = _safe_float(
            env_info.get("total_score", self.last_total_score), self.last_total_score
        )
        score_delta = max(0.0, total_score - self.last_total_score)
        timing["map_update"] = (perf_counter() - section_t0) * 1000.0

        section_t0 = perf_counter()
        hero_bfs_raw = self._bfs_distance_map(
            sources=[(hero_z, hero_x)],
            passable=self.global_passable,
            max_depth=Config.GLOBAL_BFS_THRESHOLD,
        )
        timing["bfs"] = (perf_counter() - section_t0) * 1000.0

        section_t0 = perf_counter()
        visible_monsters, target_monster, min_monster_bfs = (
            self._select_visible_monsters(
                monsters=monsters,
                hero_x=hero_x,
                hero_z=hero_z,
                hero_bfs_raw=hero_bfs_raw,
            )
        )
        self.last_visible_monsters = [dict(monster) for monster in visible_monsters]
        self.last_hero_bfs_raw = hero_bfs_raw

        legal_action = self._build_legal_action(legal_act_raw)
        buff_availability = self._build_buff_availability_map()
        nearest_treasure_bfs = self._nearest_bfs_from_positions(
            hero_bfs_raw, self.global_treasure_alive
        )
        nearest_buff_bfs = self._nearest_bfs_from_positions(
            hero_bfs_raw, buff_availability
        )
        nearest_unexplored_bfs = self._nearest_bfs_from_positions(
            hero_bfs_raw, (1.0 - self.global_explored) * self.global_passable
        )
        explore_direction_features = self._explore_direction_features(
            hero_x=hero_x,
            hero_z=hero_z,
            hero_bfs_raw=hero_bfs_raw,
            min_monster_bfs=min_monster_bfs,
        )
        timing["targets"] = (perf_counter() - section_t0) * 1000.0

        section_t0 = perf_counter()
        local_tensor = self._build_local_tensor(
            hero_x=hero_x,
            hero_z=hero_z,
            hero_bfs_raw=hero_bfs_raw,
            visible_monsters=visible_monsters,
            buff_availability=buff_availability,
            prev_explored=prev_explored,
        )
        global_tensor = self._build_global_tensor(
            hero_x=hero_x,
            hero_z=hero_z,
            hero_bfs_raw=hero_bfs_raw,
            buff_availability=buff_availability,
        )
        scalar_vec = self._build_scalar_vector(
            hero=hero,
            env_info=env_info,
            monsters=monsters,
            organs=organs,
            visible_monsters=visible_monsters,
            hero_x=hero_x,
            hero_z=hero_z,
            hero_bfs_raw=hero_bfs_raw,
            min_monster_bfs=min_monster_bfs,
            local_tensor=local_tensor,
            buff_availability=buff_availability,
            explore_direction_features=explore_direction_features,
        )
        aux_info = self._build_aux_targets(target_monster)
        timing["tensor"] = (perf_counter() - section_t0) * 1000.0

        section_t0 = perf_counter()
        h = np.zeros(Config.MAMBA_HIDDEN_DIM, dtype=np.float32)
        if hidden_state is not None:
            hidden_np = np.asarray(hidden_state, dtype=np.float32).reshape(-1)
            copy_len = min(Config.MAMBA_HIDDEN_DIM, hidden_np.shape[0])
            h[:copy_len] = hidden_np[:copy_len]

        feature = np.concatenate(
            [
                local_tensor.reshape(-1),
                global_tensor.reshape(-1),
                scalar_vec.reshape(-1),
                h,
            ]
        ).astype(np.float32)
        timing["concat"] = (perf_counter() - section_t0) * 1000.0

        section_t0 = perf_counter()
        reward = self._compute_reward(
            env_obs=env_obs,
            delta_treasure=delta_treasure,
            delta_buff=delta_buff,
            new_explored_cells=new_explored_cells,
            score_delta=score_delta,
            min_monster_bfs=min_monster_bfs,
            nearest_treasure_bfs=nearest_treasure_bfs,
            nearest_buff_bfs=nearest_buff_bfs,
            nearest_unexplored_bfs=nearest_unexplored_bfs,
            last_action=last_action,
            hero_x=hero_x,
            hero_z=hero_z,
            hero_info=hero,
        )
        timing["reward"] = (perf_counter() - section_t0) * 1000.0

        section_t0 = perf_counter()
        aux_info["metrics"] = self._build_monitor_metrics(
            env_obs=env_obs,
            delta_treasure=delta_treasure,
            delta_buff=delta_buff,
            new_explored_cells=new_explored_cells,
            score_delta=score_delta,
            min_monster_bfs=min_monster_bfs,
            visible_monsters=visible_monsters,
            legal_action=legal_action,
            last_action=last_action,
            hero_x=hero_x,
            hero_z=hero_z,
        )
        timing["metrics"] = (perf_counter() - section_t0) * 1000.0

        if min_monster_bfs < Config.BFS_SENTINEL:
            self.last_min_monster_bfs = float(min_monster_bfs)
        self.position_history.append((hero_x, hero_z))
        self.last_hero_pos = (hero_x, hero_z)
        self.last_legal_action = list(legal_action)
        self.last_explore_direction_features = np.asarray(
            explore_direction_features, dtype=np.float32
        )
        self.last_total_score = total_score
        timing["total"] = (perf_counter() - total_t0) * 1000.0
        timing_metrics = self._build_timing_metrics(timing)
        aux_info["metrics"].update(timing_metrics)
        self.last_feature_timing = dict(timing_metrics)

        return feature, legal_action, [reward], aux_info

    def _build_timing_metrics(self, timing):
        metrics = {}
        for key, value in timing.items():
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                metrics[f"feature_ms_{key}"] = v
        return metrics

    def _normalize_hero(self, raw_hero):
        if isinstance(raw_hero, list):
            return raw_hero[0] if raw_hero and isinstance(raw_hero[0], dict) else {}
        return raw_hero if isinstance(raw_hero, dict) else {}

    def _normalize_entity_list(self, raw_entities):
        if isinstance(raw_entities, list):
            return [x for x in raw_entities if isinstance(x, dict)]
        if isinstance(raw_entities, dict):
            return [x for x in raw_entities.values() if isinstance(x, dict)]
        return []

    def _decay_dynamic_maps(self):
        cooldown_decay = 1.0 / max(1.0, self.buff_refresh_steps)
        self.global_buff_cd_remaining_norm = np.clip(
            self.global_buff_cd_remaining_norm - cooldown_decay,
            0.0,
            1.0,
        )

    def _stitch_local_map(self, map_info, hero_x, hero_z):
        if map_info is None:
            return 0.0

        arr = np.asarray(map_info)
        if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] <= 0:
            return 0.0

        center_r = arr.shape[0] // 2
        center_c = arr.shape[1] // 2
        new_passable_cells = 0

        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                gr = hero_z + (r - center_r)
                gc = hero_x + (c - center_c)
                if gr < 0 or gc < 0 or gr >= Config.MAP_SIZE or gc >= Config.MAP_SIZE:
                    continue

                passable = 1.0 if _safe_float(arr[r, c], 0.0) != 0.0 else 0.0
                if self.global_explored[gr, gc] < 0.5 and passable > 0.5:
                    new_passable_cells += 1
                self.global_explored[gr, gc] = 1.0
                self.global_passable[gr, gc] = passable

        return float(new_passable_cells)

    def _update_organs_and_counters(self, organs, env_info, hero_x, hero_z):
        observed_treasure_alive = set()
        observed_buff_alive = set()

        for organ in organs:
            stype = _safe_int(organ.get("sub_type", 0), 0)
            status = _safe_int(organ.get("status", 0), 0)
            pos = organ.get("pos", {})
            if not isinstance(pos, dict):
                continue
            ox = int(np.clip(_safe_int(pos.get("x", 0), 0), 0, Config.MAP_SIZE - 1))
            oz = int(np.clip(_safe_int(pos.get("z", 0), 0), 0, Config.MAP_SIZE - 1))
            key = (oz, ox)

            if stype == 1:
                self.known_treasure_pos.add(key)
                if status == 1:
                    observed_treasure_alive.add(key)
                    self.global_treasure_alive[oz, ox] = 1.0
                else:
                    self.global_treasure_alive[oz, ox] = 0.0
            elif stype == 2:
                self.known_buff_pos.add(key)
                self.global_buff_known[oz, ox] = 1.0
                if status == 1:
                    observed_buff_alive.add(key)
                    self.global_buff_alive[oz, ox] = 1.0
                    self.global_buff_cd_remaining_norm[oz, ox] = 0.0
                else:
                    self.global_buff_alive[oz, ox] = 0.0

        treasures_collected = _safe_int(
            env_info.get("treasures_collected", self.last_treasures_collected),
            self.last_treasures_collected,
        )
        collected_buff = _safe_int(
            env_info.get("collected_buff", self.last_collected_buff),
            self.last_collected_buff,
        )
        delta_treasure = max(0, treasures_collected - self.last_treasures_collected)
        delta_buff = max(0, collected_buff - self.last_collected_buff)

        if delta_treasure > 0:
            self._mark_recent_treasures_collected(
                hero_x, hero_z, delta_treasure, observed_treasure_alive
            )
        if delta_buff > 0:
            self._mark_recent_buffs_collected(
                hero_x, hero_z, delta_buff, observed_buff_alive
            )

        self.last_treasures_collected = treasures_collected
        self.last_collected_buff = collected_buff
        return float(delta_treasure), float(delta_buff)

    def _mark_recent_treasures_collected(self, hero_x, hero_z, count, observed_alive):
        candidates = []
        for rr, cc in self.known_treasure_pos:
            if (rr, cc) in observed_alive:
                continue
            if self.global_treasure_alive[rr, cc] > 0.5:
                candidates.append((abs(rr - hero_z) + abs(cc - hero_x), rr, cc))
        candidates.sort(key=lambda x: x[0])
        for _, rr, cc in candidates[:count]:
            self.global_treasure_alive[rr, cc] = 0.0

    def _mark_recent_buffs_collected(self, hero_x, hero_z, count, observed_alive):
        candidates = []
        for rr, cc in self.known_buff_pos:
            if (rr, cc) in observed_alive:
                continue
            if self.global_buff_alive[rr, cc] > 0.5:
                candidates.append((abs(rr - hero_z) + abs(cc - hero_x), rr, cc))
        candidates.sort(key=lambda x: x[0])
        for _, rr, cc in candidates[:count]:
            self.global_buff_alive[rr, cc] = 0.0
            self.global_buff_cd_remaining_norm[rr, cc] = 1.0
            self.global_buff_known[rr, cc] = 1.0

    def _build_legal_action(self, legal_act_raw):
        legal_action = [1] * Config.ACTION_NUM

        if isinstance(legal_act_raw, np.ndarray):
            legal_act_raw = legal_act_raw.tolist()

        if isinstance(legal_act_raw, (list, tuple)) and legal_act_raw:
            raw = list(legal_act_raw)
            first = raw[0]
            if isinstance(first, (bool, np.bool_)):
                for j in range(min(Config.ACTION_NUM, len(raw))):
                    legal_action[j] = int(bool(raw[j]))
            else:
                mask_slice = raw[: Config.ACTION_NUM]
                numeric_mask = len(mask_slice) == Config.ACTION_NUM and all(
                    isinstance(a, (bool, np.bool_, int, np.integer, float, np.floating))
                    for a in mask_slice
                )
                if numeric_mask and all(
                    abs(_safe_float(a, 0.0) - round(_safe_float(a, 0.0))) < 1e-6
                    for a in mask_slice
                ):
                    if all(_safe_float(a, 0.0) in (0.0, 1.0) for a in mask_slice):
                        legal_action = [
                            1 if _safe_float(a, 0.0) > 0.5 else 0 for a in mask_slice
                        ]
                    else:
                        valid_set = {
                            int(a) for a in raw if 0 <= int(a) < Config.ACTION_NUM
                        }
                        legal_action = [
                            1 if j in valid_set else 0 for j in range(Config.ACTION_NUM)
                        ]
                else:
                    valid_set = {
                        int(a)
                        for a in raw
                        if isinstance(a, (int, np.integer, float, np.floating))
                        and 0 <= int(a) < Config.ACTION_NUM
                    }
                    legal_action = [
                        1 if j in valid_set else 0 for j in range(Config.ACTION_NUM)
                    ]

        if sum(legal_action) == 0:
            legal_action = [1] * Config.ACTION_NUM
        return legal_action

    def _bfs_distance_map(self, sources, passable, max_depth):
        dist = np.full(
            (Config.MAP_SIZE, Config.MAP_SIZE), Config.BFS_SENTINEL, dtype=np.float32
        )
        queue = deque()
        for rr, cc in sources:
            if 0 <= rr < Config.MAP_SIZE and 0 <= cc < Config.MAP_SIZE:
                dist[rr, cc] = 0.0
                queue.append((rr, cc))

        while queue:
            rr, cc = queue.popleft()
            next_dist = dist[rr, cc] + 1.0
            if next_dist > max_depth:
                continue
            for dr, dc in _DIRECTIONS_8:
                nr = rr + dr
                nc = cc + dc
                if nr < 0 or nc < 0 or nr >= Config.MAP_SIZE or nc >= Config.MAP_SIZE:
                    continue
                if passable[nr, nc] < 0.5 and dist[nr, nc] > 0.0:
                    continue
                if next_dist < dist[nr, nc]:
                    dist[nr, nc] = next_dist
                    queue.append((nr, nc))
        return dist

    def _normalize_bfs(self, bfs_raw, threshold):
        return (np.minimum(bfs_raw, threshold) / max(1.0, threshold)).astype(np.float32)

    def _select_visible_monsters(self, monsters, hero_x, hero_z, hero_bfs_raw):
        visible = []
        for monster in monsters:
            pos = monster.get("pos", {})
            if not isinstance(pos, dict):
                continue
            mx = int(
                np.clip(_safe_int(pos.get("x", hero_x), hero_x), 0, Config.MAP_SIZE - 1)
            )
            mz = int(
                np.clip(_safe_int(pos.get("z", hero_z), hero_z), 0, Config.MAP_SIZE - 1)
            )
            is_visible = _safe_float(monster.get("is_in_view", 1), 1.0) > 0.5
            if not is_visible:
                continue
            bfs_dist = float(hero_bfs_raw[mz, mx])
            l2_dist = float(np.sqrt((hero_x - mx) ** 2 + (hero_z - mz) ** 2))
            enriched = dict(monster)
            enriched["_x"] = mx
            enriched["_z"] = mz
            enriched["_bfs"] = bfs_dist
            enriched["_l2"] = l2_dist
            visible.append(enriched)

        visible.sort(key=lambda m: (m["_bfs"], m["_l2"]))
        target = visible[0] if visible else None
        min_bfs = float(target["_bfs"]) if target is not None else Config.BFS_SENTINEL
        return visible, target, min_bfs

    def _build_buff_availability_map(self):
        availability = np.zeros((Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.float32)
        known = self.global_buff_known > 0.5
        availability[known] = 1.0 - self.global_buff_cd_remaining_norm[known]
        availability[self.global_buff_alive > 0.5] = 1.0
        return np.clip(availability, 0.0, 1.0).astype(np.float32)

    def _build_local_tensor(
        self,
        hero_x,
        hero_z,
        hero_bfs_raw,
        visible_monsters,
        buff_availability,
        prev_explored,
    ):
        passable = self._crop_map(
            self.global_passable, hero_x, hero_z, Config.LOCAL_MAP_SIZE, fill_value=0.0
        )
        bfs = self._crop_map(
            self._normalize_bfs(hero_bfs_raw, Config.LOCAL_BFS_THRESHOLD),
            hero_x,
            hero_z,
            Config.LOCAL_MAP_SIZE,
            fill_value=1.0,
        )
        hero_layer = np.zeros(
            (Config.LOCAL_MAP_SIZE, Config.LOCAL_MAP_SIZE), dtype=np.float32
        )
        hero_layer[Config.LOCAL_MAP_SIZE // 2, Config.LOCAL_MAP_SIZE // 2] = 1.0

        monster_layer_global = np.zeros(
            (Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.float32
        )
        for monster in visible_monsters:
            monster_layer_global[monster["_z"], monster["_x"]] = 1.0
        monster_layer = self._crop_map(
            monster_layer_global, hero_x, hero_z, Config.LOCAL_MAP_SIZE, fill_value=0.0
        )

        treasure_layer = self._crop_map(
            self.global_treasure_alive,
            hero_x,
            hero_z,
            Config.LOCAL_MAP_SIZE,
            fill_value=0.0,
        )
        buff_layer = self._crop_map(
            buff_availability, hero_x, hero_z, Config.LOCAL_MAP_SIZE, fill_value=0.0
        )
        unexplored_passable = (1.0 - prev_explored) * self.global_passable
        explore_layer = self._crop_map(
            unexplored_passable, hero_x, hero_z, Config.LOCAL_MAP_SIZE, fill_value=0.0
        )

        return np.stack(
            [
                passable,
                bfs,
                hero_layer,
                monster_layer,
                treasure_layer,
                buff_layer,
                explore_layer,
            ],
            axis=0,
        ).astype(np.float32)

    def _build_global_tensor(self, hero_x, hero_z, hero_bfs_raw, buff_availability):
        hero_layer = np.zeros((Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.float32)
        hero_layer[hero_z, hero_x] = 1.0
        unexplored_passable = (1.0 - self.global_explored) * self.global_passable

        full_tensor = np.stack(
            [
                self.global_passable,
                self._normalize_bfs(hero_bfs_raw, Config.GLOBAL_BFS_THRESHOLD),
                hero_layer,
                self.global_treasure_alive,
                buff_availability,
                unexplored_passable,
            ],
            axis=0,
        ).astype(np.float32)
        return self._compress_global_tensor(full_tensor)

    def _compress_global_tensor(self, full_tensor):
        if Config.GLOBAL_MAP_SIZE == Config.MAP_SIZE:
            return full_tensor.astype(np.float32)
        if Config.MAP_SIZE % Config.GLOBAL_MAP_SIZE != 0:
            row_idx = np.linspace(
                0, Config.MAP_SIZE - 1, Config.GLOBAL_MAP_SIZE
            ).astype(np.int32)
            col_idx = np.linspace(
                0, Config.MAP_SIZE - 1, Config.GLOBAL_MAP_SIZE
            ).astype(np.int32)
            return full_tensor[:, row_idx][:, :, col_idx].astype(np.float32)

        block = Config.MAP_SIZE // Config.GLOBAL_MAP_SIZE
        reshaped = full_tensor.reshape(
            Config.GLOBAL_CHANNELS,
            Config.GLOBAL_MAP_SIZE,
            block,
            Config.GLOBAL_MAP_SIZE,
            block,
        )
        compressed = np.empty(
            (
                Config.GLOBAL_CHANNELS,
                Config.GLOBAL_MAP_SIZE,
                Config.GLOBAL_MAP_SIZE,
            ),
            dtype=np.float32,
        )
        compressed[0] = reshaped[0].mean(axis=(1, 3))
        compressed[1] = reshaped[1].min(axis=(1, 3))
        compressed[2:] = reshaped[2:].max(axis=(2, 4))
        return compressed.astype(np.float32)

    def _build_scalar_vector(
        self,
        hero,
        env_info,
        monsters,
        organs,
        visible_monsters,
        hero_x,
        hero_z,
        hero_bfs_raw,
        min_monster_bfs,
        local_tensor,
        buff_availability,
        explore_direction_features,
    ):
        treasures = self._filter_organs(
            organs, sub_type=1, hero_x=hero_x, hero_z=hero_z
        )
        buffs = self._filter_organs(organs, sub_type=2, hero_x=hero_x, hero_z=hero_z)

        nearest_treasure_bfs = self._nearest_bfs_from_positions(
            hero_bfs_raw, self.global_treasure_alive
        )
        nearest_buff_bfs = self._nearest_bfs_from_positions(
            hero_bfs_raw, buff_availability
        )
        flash_cd_norm = _norm(
            hero.get("flash_cooldown", env_info.get("flash_cooldown", 0)),
            Config.MAX_FLASH_CD,
        )
        hero_buff_norm = _norm(
            hero.get("buff_remaining_time", 0), Config.MAX_BUFF_DURATION
        )
        hero_speed = (
            2.0 if _safe_float(hero.get("buff_remaining_time", 0), 0.0) > 0.0 else 1.0
        )
        global_explore_progress = self._global_explore_progress_feature()

        base = [
            float(
                np.sum(local_tensor[0] > 0.5)
                / (Config.LOCAL_MAP_SIZE * Config.LOCAL_MAP_SIZE)
            ),
            self._norm_bfs_scalar(min_monster_bfs),
            self._norm_bfs_scalar(nearest_treasure_bfs),
            self._norm_bfs_scalar(nearest_buff_bfs),
            _norm(len(visible_monsters), Config.MAX_MONSTERS),
            float(
                np.mean(
                    self.global_buff_cd_remaining_norm[self.global_buff_known > 0.5]
                )
            )
            if np.any(self.global_buff_known > 0.5)
            else 0.0,
            float(
                np.sum(local_tensor[6] > 0.5)
                / (Config.LOCAL_MAP_SIZE * Config.LOCAL_MAP_SIZE)
            ),
            float(
                np.sum((1.0 - self.global_explored) * self.global_passable)
                / (Config.MAP_SIZE * Config.MAP_SIZE)
            ),
            _norm(self.step_no, self.max_step),
            _norm(hero_x, Config.MAP_SIZE - 1),
            _norm(hero_z, Config.MAP_SIZE - 1),
            flash_cd_norm,
            hero_buff_norm,
            _norm(hero_speed, Config.MAX_HERO_SPEED),
            _norm(len(monsters), Config.MAX_MONSTERS),
            _norm(
                len([o for o in treasures if _safe_int(o.get("status", 1), 1) == 1]),
                Config.MAX_TREASURES,
            ),
            _norm(
                len([o for o in buffs if _safe_int(o.get("status", 1), 1) == 1]),
                Config.MAX_BUFFS,
            ),
            global_explore_progress,
        ]

        scalar_parts = [np.asarray(base, dtype=np.float32)]

        sorted_monsters = sorted(
            monsters, key=lambda m: self._entity_l2(m, hero_x, hero_z)
        )
        for i in range(Config.MAX_MONSTERS):
            scalar_parts.append(
                self._monster_scalar(
                    sorted_monsters[i] if i < len(sorted_monsters) else None,
                    hero_x,
                    hero_z,
                    hero_bfs_raw,
                )
            )

        for i in range(Config.MAX_TREASURES):
            scalar_parts.append(
                self._organ_scalar(
                    treasures[i] if i < len(treasures) else None,
                    hero_x,
                    hero_z,
                    hero_bfs_raw,
                    include_cd=False,
                )
            )

        for i in range(Config.MAX_BUFFS):
            scalar_parts.append(
                self._organ_scalar(
                    buffs[i] if i < len(buffs) else None,
                    hero_x,
                    hero_z,
                    hero_bfs_raw,
                    include_cd=True,
                )
            )

        scalar_parts.append(np.asarray(explore_direction_features, dtype=np.float32))

        scalar = np.concatenate(scalar_parts).astype(np.float32)
        if scalar.shape[0] < Config.SCALAR_DIM:
            scalar = np.concatenate(
                [
                    scalar,
                    np.zeros(Config.SCALAR_DIM - scalar.shape[0], dtype=np.float32),
                ]
            )
        elif scalar.shape[0] > Config.SCALAR_DIM:
            scalar = scalar[: Config.SCALAR_DIM]
        return scalar.astype(np.float32)

    def _filter_organs(self, organs, sub_type, hero_x, hero_z):
        filtered = [o for o in organs if _safe_int(o.get("sub_type", 0), 0) == sub_type]
        filtered.sort(key=lambda o: self._entity_l2(o, hero_x, hero_z))
        return filtered

    def _entity_l2(self, entity, hero_x, hero_z):
        if not isinstance(entity, dict):
            return float("inf")
        pos = entity.get("pos", {})
        if not isinstance(pos, dict):
            return float("inf")
        return float(
            (hero_x - _safe_int(pos.get("x", hero_x), hero_x)) ** 2
            + (hero_z - _safe_int(pos.get("z", hero_z), hero_z)) ** 2
        )

    def _nearest_bfs_from_positions(self, hero_bfs_raw, mask):
        masked_values = hero_bfs_raw[mask > 0.5]
        if masked_values.size == 0:
            return Config.BFS_SENTINEL
        return float(np.min(masked_values))

    def _norm_bfs_scalar(self, value):
        return _norm(
            min(value, Config.GLOBAL_BFS_THRESHOLD), Config.GLOBAL_BFS_THRESHOLD
        )

    def _explore_direction_features(
        self, hero_x, hero_z, hero_bfs_raw, min_monster_bfs
    ):
        safety = self._explore_safety_factor(min_monster_bfs)
        no_frontier = self._scale_explore_features(
            [0.0, 0.0, 0.0, 1.0, 0.0, safety]
        )

        reachable = (
            np.isfinite(hero_bfs_raw)
            & (hero_bfs_raw < Config.BFS_SENTINEL)
            & (hero_bfs_raw <= Config.GLOBAL_BFS_THRESHOLD)
        )
        unexplored = (self.global_passable > 0.5) & (self.global_explored < 0.5)
        frontier = unexplored & self._adjacent_to_explored_mask()
        active_frontier = frontier & reachable
        if not np.any(active_frontier):
            active_frontier = unexplored & reachable
        frontier_count = int(np.count_nonzero(active_frontier))
        if frontier_count <= 0:
            return no_frontier

        zs, xs = np.where(active_frontier)
        dist = hero_bfs_raw[zs, xs].astype(np.float32)
        dx = xs.astype(np.float32) - float(hero_x)
        dz = zs.astype(np.float32) - float(hero_z)
        l2 = np.sqrt(dx * dx + dz * dz)
        mask = (dist > 0.0) & (l2 > 1.0e-6) & np.isfinite(dist)
        if not np.any(mask):
            features = np.asarray(
                [0.0, 0.0, 0.0, 1.0, 0.0, safety], dtype=np.float32
            )
            features[4] = _norm(
                frontier_count, Config.EXPLORE_VECTOR_FRONTIER_COUNT_NORM
            )
            return self._scale_explore_features(features)

        dist = dist[mask]
        dx = dx[mask]
        dz = dz[mask]
        l2 = l2[mask]
        weights = 1.0 / (dist + 1.0)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1.0e-6:
            return no_frontier

        raw_x = float(np.sum(weights * (dx / l2)) / weight_sum)
        raw_z = float(np.sum(weights * (dz / l2)) / weight_sum)
        coherence = float(np.clip(np.sqrt(raw_x * raw_x + raw_z * raw_z), 0.0, 1.0))
        if coherence > 1.0e-6:
            unit_x = raw_x / coherence
            unit_z = raw_z / coherence
        else:
            unit_x = 0.0
            unit_z = 0.0

        strength = coherence * safety
        nearest_bfs = float(np.min(dist))
        return self._scale_explore_features(
            [
                float(np.clip(unit_x * strength, -1.0, 1.0)),
                float(np.clip(unit_z * strength, -1.0, 1.0)),
                float(np.clip(strength, 0.0, 1.0)),
                self._norm_bfs_scalar(nearest_bfs),
                _norm(frontier_count, Config.EXPLORE_VECTOR_FRONTIER_COUNT_NORM),
                safety,
            ]
        )

    def _scale_explore_features(self, features):
        scale = float(np.clip(Config.EXPLORE_VECTOR_FEATURE_SCALE, 0.0, 1.0))
        return (np.asarray(features, dtype=np.float32) * scale).astype(np.float32)

    def _explore_vector_reward(self, hero_x, hero_z, move_used, has_prev_pos):
        if not move_used or not has_prev_pos or self.last_hero_pos is None:
            return 0.0
        scale = float(Config.EXPLORE_VECTOR_FEATURE_SCALE)
        if not np.isfinite(scale) or scale <= 1.0e-6:
            return 0.0
        prev_features = np.asarray(
            self.last_explore_direction_features, dtype=np.float32
        ).reshape(-1)
        if prev_features.shape[0] < 3:
            return 0.0

        raw_strength = float(prev_features[2] / scale)
        if raw_strength < Config.EXPLORE_VECTOR_REWARD_MIN_STRENGTH:
            return 0.0

        last_x, last_z = self.last_hero_pos
        move_x = float(hero_x - last_x)
        move_z = float(hero_z - last_z)
        move_norm = float(np.sqrt(move_x * move_x + move_z * move_z))
        if move_norm <= 1.0e-6:
            return 0.0

        move_x /= move_norm
        move_z /= move_norm
        explore_x = float(prev_features[0] / scale)
        explore_z = float(prev_features[1] / scale)
        alignment = float(
            np.clip(move_x * explore_x + move_z * explore_z, -1.0, 1.0)
        )
        if alignment > 0.0:
            return float(Config.REWARD_EXPLORE_VECTOR_ALIGN * alignment)
        if alignment < 0.0:
            return float(Config.PENALTY_EXPLORE_VECTOR_OPPOSE * (-alignment))
        return 0.0

    def _adjacent_to_explored_mask(self):
        explored = self.global_explored > 0.5
        adjacent = np.zeros_like(explored, dtype=bool)
        height, width = explored.shape
        for dz, dx in _DIRECTIONS_8:
            src_z0 = max(0, -dz)
            src_z1 = height - max(0, dz)
            src_x0 = max(0, -dx)
            src_x1 = width - max(0, dx)
            dst_z0 = max(0, dz)
            dst_z1 = dst_z0 + (src_z1 - src_z0)
            dst_x0 = max(0, dx)
            dst_x1 = dst_x0 + (src_x1 - src_x0)
            if src_z1 > src_z0 and src_x1 > src_x0:
                adjacent[dst_z0:dst_z1, dst_x0:dst_x1] |= explored[
                    src_z0:src_z1, src_x0:src_x1
                ]
        return adjacent

    def _explore_safety_factor(self, min_monster_bfs):
        if (
            not np.isfinite(min_monster_bfs)
            or min_monster_bfs >= Config.BFS_SENTINEL
        ):
            return 1.0
        danger_bfs = float(Config.EXPLORE_VECTOR_DANGER_BFS)
        safe_bfs = max(float(Config.EXPLORE_VECTOR_SAFE_BFS), danger_bfs + 1.0e-6)
        danger_scale = float(np.clip(Config.EXPLORE_VECTOR_DANGER_SCALE, 0.0, 1.0))
        monster_bfs = float(min_monster_bfs)
        if monster_bfs <= danger_bfs:
            return danger_scale
        if monster_bfs >= safe_bfs:
            return 1.0
        t = (monster_bfs - danger_bfs) / (safe_bfs - danger_bfs)
        return float(danger_scale + (1.0 - danger_scale) * t)

    def _monster_scalar(self, monster, hero_x, hero_z, hero_bfs_raw):
        if monster is None:
            return np.zeros(23, dtype=np.float32)

        pos = monster.get("pos", {})
        start_pos = monster.get("start_pos", {})
        if not isinstance(pos, dict):
            pos = {}
        if not isinstance(start_pos, dict):
            start_pos = {}
        mx = int(
            np.clip(_safe_int(pos.get("x", hero_x), hero_x), 0, Config.MAP_SIZE - 1)
        )
        mz = int(
            np.clip(_safe_int(pos.get("z", hero_z), hero_z), 0, Config.MAP_SIZE - 1)
        )
        sx = int(np.clip(_safe_int(start_pos.get("x", mx), mx), 0, Config.MAP_SIZE - 1))
        sz = int(np.clip(_safe_int(start_pos.get("z", mz), mz), 0, Config.MAP_SIZE - 1))
        visible = 1.0 if _safe_float(monster.get("is_in_view", 1), 1.0) > 0.5 else 0.0
        bucket = _safe_int(
            monster.get("hero_l2_distance", _l2_bucket(hero_x - mx, hero_z - mz)), 0
        )
        direction_id = monster.get(
            "hero_relative_direction", _relative_direction_id(hero_x, hero_z, mx, mz)
        )
        l2_one_hot = np.zeros(6, dtype=np.float32)
        l2_one_hot[int(np.clip(bucket, 0, 5))] = 1.0

        return np.concatenate(
            [
                np.asarray(
                    [
                        1.0,
                        visible,
                        _norm(mx, Config.MAP_SIZE - 1),
                        _norm(mz, Config.MAP_SIZE - 1),
                        _norm(sx, Config.MAP_SIZE - 1),
                        _norm(sz, Config.MAP_SIZE - 1),
                    ],
                    dtype=np.float32,
                ),
                l2_one_hot,
                _direction_one_hot(direction_id),
                np.asarray(
                    [
                        _norm(monster.get("speed", 0), Config.MAX_MONSTER_SPEED),
                        _norm(
                            monster.get("monster_interval", 0),
                            Config.MAX_MONSTER_INTERVAL,
                        ),
                        self._norm_bfs_scalar(float(hero_bfs_raw[mz, mx])),
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)

    def _organ_scalar(self, organ, hero_x, hero_z, hero_bfs_raw, include_cd):
        size = 19 if include_cd else 18
        if organ is None:
            return np.zeros(size, dtype=np.float32)

        pos = organ.get("pos", {})
        if not isinstance(pos, dict):
            pos = {}
        ox = int(
            np.clip(_safe_int(pos.get("x", hero_x), hero_x), 0, Config.MAP_SIZE - 1)
        )
        oz = int(
            np.clip(_safe_int(pos.get("z", hero_z), hero_z), 0, Config.MAP_SIZE - 1)
        )
        status = 1.0 if _safe_int(organ.get("status", 1), 1) == 1 else 0.0
        bucket = _safe_int(
            organ.get("hero_l2_distance", _l2_bucket(hero_x - ox, hero_z - oz)), 0
        )
        direction_id = organ.get(
            "hero_relative_direction", _relative_direction_id(hero_x, hero_z, ox, oz)
        )
        l2_one_hot = np.zeros(6, dtype=np.float32)
        l2_one_hot[int(np.clip(bucket, 0, 5))] = 1.0

        parts = [
            np.asarray(
                [
                    1.0,
                    status,
                    _norm(ox, Config.MAP_SIZE - 1),
                    _norm(oz, Config.MAP_SIZE - 1),
                ],
                dtype=np.float32,
            ),
            l2_one_hot,
            _direction_one_hot(direction_id),
            np.asarray(
                [self._norm_bfs_scalar(float(hero_bfs_raw[oz, ox]))], dtype=np.float32
            ),
        ]
        if include_cd:
            parts.append(
                np.asarray(
                    [1.0 - float(self.global_buff_cd_remaining_norm[oz, ox])],
                    dtype=np.float32,
                )
            )
        return np.concatenate(parts).astype(np.float32)

    def _build_aux_targets(self, target_monster):
        if target_monster is None:
            return {
                "monster_pos_target": np.zeros(2, dtype=np.float32),
                "monster_pos_mask": np.zeros(1, dtype=np.float32),
                "monster_dist_target": np.zeros(1, dtype=np.float32),
                "monster_dist_mask": np.zeros(1, dtype=np.float32),
            }

        mx = int(target_monster["_x"])
        mz = int(target_monster["_z"])
        bucket = _safe_int(target_monster.get("hero_l2_distance", 0), 0)

        return {
            "monster_pos_target": np.asarray(
                [_norm(mx, Config.MAP_SIZE - 1), _norm(mz, Config.MAP_SIZE - 1)],
                dtype=np.float32,
            ),
            "monster_pos_mask": np.ones(1, dtype=np.float32),
            "monster_dist_target": np.asarray(
                [float(np.clip(bucket, 0, 5))], dtype=np.float32
            ),
            "monster_dist_mask": np.ones(1, dtype=np.float32),
        }

    def _build_monitor_metrics(
        self,
        env_obs,
        delta_treasure,
        delta_buff,
        new_explored_cells,
        score_delta,
        min_monster_bfs,
        visible_monsters,
        legal_action,
        last_action,
        hero_x,
        hero_z,
    ):
        last_action_id = _safe_int(last_action, -1)
        has_action = 0 <= last_action_id < Config.ACTION_NUM
        move_used = 0 <= last_action_id < Config.MOVE_ACTION_NUM
        flash_used = Config.MOVE_ACTION_NUM <= last_action_id < Config.ACTION_NUM
        has_visible_monster = min_monster_bfs < Config.BFS_SENTINEL
        still_step = self.last_hero_pos is not None and self.last_hero_pos == (
            hero_x,
            hero_z,
        )
        position_changed = self.last_hero_pos is not None and self.last_hero_pos != (
            hero_x,
            hero_z,
        )
        wall_or_still_after_move = move_used and still_step

        prev_legal = self.last_legal_action
        if (
            isinstance(prev_legal, (list, tuple))
            and has_action
            and len(prev_legal) >= Config.ACTION_NUM
        ):
            chosen_action_is_legal = float(prev_legal[last_action_id] > 0)
        else:
            chosen_action_is_legal = 1.0 if not has_action else 0.0

        flash_bfs_delta = 0.0
        flash_success = False
        if flash_used and self.last_min_monster_bfs is not None and has_visible_monster:
            flash_bfs_delta = float(min_monster_bfs - self.last_min_monster_bfs)
            flash_success = flash_bfs_delta > 0.0

        reward_debug = getattr(self, "last_reward_debug", {})
        metrics = {
            "explore_cells": float(new_explored_cells),
            "global_explore_ratio": float(
                reward_debug.get("global_explore_ratio", 0.0)
            ),
            "global_explore_target_ratio": float(
                reward_debug.get("global_explore_target_ratio", 0.0)
            ),
            "global_explore_gap": float(
                reward_debug.get("global_explore_gap", 0.0)
            ),
            "score_delta": float(score_delta),
            "treasure_count": float(delta_treasure),
            "buff_count": float(delta_buff),
            "move_used": float(move_used),
            "flash_used": float(flash_used),
            "flash_success": float(flash_success),
            "flash_bfs_delta": float(flash_bfs_delta),
            "still_step": float(still_step),
            "position_changed": float(position_changed),
            "wall_or_still_after_move": float(wall_or_still_after_move),
            "chosen_action_is_legal": float(chosen_action_is_legal),
            "executed_action_count": float(has_action),
            "legal_action_count": float(sum(legal_action)),
            "legal_move_count": float(sum(legal_action[: Config.MOVE_ACTION_NUM])),
            "legal_flash_count": float(sum(legal_action[Config.MOVE_ACTION_NUM :])),
            "caught": float(bool(env_obs.get("terminated", False))),
            "post500_alive": float(self.step_no >= 500),
            "visible_monster_count": float(len(visible_monsters)),
            "has_visible_monster": float(has_visible_monster),
            "min_monster_bfs": float(min(min_monster_bfs, Config.GLOBAL_BFS_THRESHOLD))
            if has_visible_monster
            else 0.0,
        }
        for key in (
            "step_reward",
            "score_reward",
            "treasure_reward",
            "explore_cell_reward",
            "buff_reward",
            "survival_bonus",
            "safety_bonus",
            "monster_bfs_reward",
            "explore_approach_bonus",
            "explore_vector_reward",
            "global_explore_reward",
            "resource_approach_bonus",
            "post_escape_scout_reward",
            "trail_reward",
            "route_safety_reward",
            "flash_reward",
            "anti_loop_reward",
            "penalty_near_monster",
            "penalty_still",
            "penalty_caught",
            "reward_component_count",
            "reward_component_count_sq",
            "reward_positive_component_count",
            "reward_negative_component_count",
        ):
            metrics[key] = float(reward_debug.get(key, 0.0))
        return metrics

    def _compute_reward(
        self,
        env_obs,
        delta_treasure,
        delta_buff,
        new_explored_cells,
        score_delta,
        min_monster_bfs,
        nearest_treasure_bfs,
        nearest_buff_bfs,
        nearest_unexplored_bfs,
        last_action,
        hero_x,
        hero_z,
        hero_info,
    ):
        reward_debug = {
            "step_reward": 0.0,
            "score_reward": 0.0,
            "treasure_reward": 0.0,
            "explore_cell_reward": 0.0,
            "buff_reward": 0.0,
            "survival_bonus": 0.0,
            "safety_bonus": 0.0,
            "monster_bfs_reward": 0.0,
            "explore_approach_bonus": 0.0,
            "explore_vector_reward": 0.0,
            "global_explore_reward": 0.0,
            "global_explore_ratio": 0.0,
            "global_explore_target_ratio": 0.0,
            "global_explore_gap": 0.0,
            "resource_approach_bonus": 0.0,
            "post_escape_scout_reward": 0.0,
            "trail_reward": 0.0,
            "route_safety_reward": 0.0,
            "flash_reward": 0.0,
            "anti_loop_reward": 0.0,
            "penalty_near_monster": 0.0,
            "penalty_still": 0.0,
            "penalty_caught": 0.0,
        }

        reward = Config.REWARD_STEP
        reward_debug["step_reward"] = float(Config.REWARD_STEP)
        reward += Config.REWARD_SURVIVE_ALIVE_PER_STEP
        reward_debug["survival_bonus"] += float(Config.REWARD_SURVIVE_ALIVE_PER_STEP)
        score_reward = Config.REWARD_SCORE_DELTA * float(score_delta)
        treasure_reward = Config.REWARD_TREASURE * float(delta_treasure)
        explore_cell_reward = Config.REWARD_EXPLORE_PER_CELL * float(
            new_explored_cells
        )
        reward += score_reward
        reward += treasure_reward
        reward += explore_cell_reward
        reward_debug["score_reward"] = float(score_reward)
        reward_debug["treasure_reward"] = float(treasure_reward)
        reward_debug["explore_cell_reward"] = float(explore_cell_reward)

        (
            global_explore_reward,
            global_explore_ratio,
            global_explore_target,
            global_explore_gap,
        ) = self._global_explore_progress_reward()
        reward += global_explore_reward
        reward_debug["global_explore_reward"] = float(global_explore_reward)
        reward_debug["global_explore_ratio"] = float(global_explore_ratio)
        reward_debug["global_explore_target_ratio"] = float(global_explore_target)
        reward_debug["global_explore_gap"] = float(global_explore_gap)

        # Buff maintenance reward / Buff 维持奖励
        if _safe_float(hero_info.get("buff_remaining_time", 0.0)) > 0.0:
            reward += Config.REWARD_BUFF_MAINTAIN
            reward_debug["buff_reward"] += float(Config.REWARD_BUFF_MAINTAIN)

        has_prev_pos = self.last_hero_pos is not None
        still_step = has_prev_pos and self.last_hero_pos == (hero_x, hero_z)
        last_action_id = _safe_int(last_action, -1)
        move_used = 0 <= last_action_id < Config.MOVE_ACTION_NUM
        flash_used = Config.MOVE_ACTION_NUM <= last_action_id < Config.ACTION_NUM

        if still_step:
            self.consecutive_still_count += 1
        else:
            self.consecutive_still_count = 0

        if delta_buff > 0:
            buff_reward = (
                Config.REWARD_BUFF_AFTER_500
                if self.step_no >= 500
                else Config.REWARD_BUFF_BEFORE_500
            ) * float(delta_buff)
            reward += buff_reward
            reward_debug["buff_reward"] += float(buff_reward)

        monster_bfs_delta = None
        if (
            self.last_min_monster_bfs is not None
            and min_monster_bfs < Config.BFS_SENTINEL
        ):
            monster_bfs_delta = float(min_monster_bfs - self.last_min_monster_bfs)

        if min_monster_bfs >= Config.BFS_SENTINEL:
            reward += Config.REWARD_NO_VISIBLE_MONSTER
            reward_debug["safety_bonus"] += float(Config.REWARD_NO_VISIBLE_MONSTER)
        elif min_monster_bfs >= Config.REWARD_SAFE_MONSTER_BFS_THRESHOLD:
            safe_bfs = min(min_monster_bfs, Config.REWARD_SAFE_MONSTER_BFS_CAP)
            safety_bonus = (
                safe_bfs - Config.REWARD_SAFE_MONSTER_BFS_THRESHOLD
            ) * Config.REWARD_SAFE_MONSTER_BFS_COEF
            reward += safety_bonus
            reward_debug["safety_bonus"] += float(safety_bonus)

        monster_pressure_scale = self._monster_pressure_scale(min_monster_bfs)
        if move_used and monster_bfs_delta is not None and monster_pressure_scale > 0.0:
            clipped_delta = float(
                np.clip(
                    monster_bfs_delta,
                    -Config.REWARD_MONSTER_BFS_DELTA_CLIP,
                    Config.REWARD_MONSTER_BFS_DELTA_CLIP,
                )
            )
            if clipped_delta > 0.0:
                bfs_reward = (
                    Config.REWARD_MONSTER_BFS_DELTA_POS
                    * clipped_delta
                    * monster_pressure_scale
                )
                reward += bfs_reward
                reward_debug["monster_bfs_reward"] += float(bfs_reward)
            elif clipped_delta < 0.0:
                bfs_reward = (
                    Config.REWARD_MONSTER_BFS_DELTA_NEG
                    * clipped_delta
                    * monster_pressure_scale
                )
                reward += bfs_reward
                reward_debug["monster_bfs_reward"] += float(bfs_reward)

        if (
            move_used
            and self.last_min_monster_bfs is not None
            and self.last_min_monster_bfs <= Config.REWARD_DANGER_ESCAPE_THRESHOLD
            and min_monster_bfs > Config.REWARD_DANGER_ESCAPE_THRESHOLD
        ):
            reward += Config.REWARD_DANGER_ESCAPE_BONUS
            reward_debug["safety_bonus"] += float(Config.REWARD_DANGER_ESCAPE_BONUS)

        if flash_used:
            danger_before_flash = (
                self.last_min_monster_bfs is not None
                and self.last_min_monster_bfs
                <= Config.REWARD_FLASH_DANGER_BFS_THRESHOLD
            )
            if danger_before_flash:
                if min_monster_bfs >= Config.BFS_SENTINEL:
                    flash_reward = Config.REWARD_FLASH_DANGER_SUCCESS
                elif (
                    monster_bfs_delta is not None
                    and monster_bfs_delta >= Config.REWARD_FLASH_SUCCESS_DELTA
                ):
                    flash_reward = Config.REWARD_FLASH_DANGER_SUCCESS
                else:
                    flash_reward = Config.PENALTY_FLASH_DANGER_FAIL
            else:
                flash_reward = Config.PENALTY_FLASH_SAFE
            reward += flash_reward
            reward_debug["flash_reward"] += float(flash_reward)

            # 检查是否在 CD 期间尝试闪现
            if (
                self.last_legal_action is not None
                and self.last_legal_action[last_action_id] == 0
            ):
                reward += Config.PENALTY_FLASH_ON_CD
                reward_debug["flash_reward"] += float(Config.PENALTY_FLASH_ON_CD)

        for milestone_step, milestone_reward in Config.REWARD_SURVIVAL_MILESTONES:
            if (
                self.step_no >= milestone_step
                and milestone_step not in self.survival_milestones_paid
            ):
                reward += milestone_reward
                reward_debug["survival_bonus"] += float(milestone_reward)
                self.survival_milestones_paid.add(milestone_step)

        if still_step:
            still_penalty = Config.PENALTY_STILL
            if move_used:
                still_penalty += Config.PENALTY_MOVE_STILL
            still_penalty += max(
                Config.PENALTY_CONSECUTIVE_STILL_MAX,
                Config.PENALTY_CONSECUTIVE_STILL_STEP
                * float(self.consecutive_still_count),
            )
            reward += still_penalty
            reward_debug["penalty_still"] = float(still_penalty)

        anti_loop_dist = self._distance_from_history(hero_x, hero_z)
        if anti_loop_dist is not None:
            if anti_loop_dist < Config.ANTI_LOOP_CLOSE_DIST:
                reward += Config.PENALTY_ANTI_LOOP_CLOSE
                reward_debug["anti_loop_reward"] += float(
                    Config.PENALTY_ANTI_LOOP_CLOSE
                )
            elif anti_loop_dist < Config.ANTI_LOOP_NEAR_DIST:
                reward += Config.PENALTY_ANTI_LOOP_NEAR
                reward_debug["anti_loop_reward"] += float(
                    Config.PENALTY_ANTI_LOOP_NEAR
                )
            elif move_used and anti_loop_dist >= Config.ANTI_LOOP_PROGRESS_DIST:
                reward += Config.REWARD_ANTI_LOOP_PROGRESS
                reward_debug["anti_loop_reward"] += float(
                    Config.REWARD_ANTI_LOOP_PROGRESS
                )

        # Frontier approach reward / 探索前沿引导奖励
        post_escape_reward = self._post_escape_scout_reward(
            hero_x, hero_z, min_monster_bfs, move_used
        )
        reward += post_escape_reward
        reward_debug["post_escape_scout_reward"] += float(post_escape_reward)

        trail_reward = self._trail_progress_reward(hero_x, hero_z, move_used)
        reward += trail_reward
        reward_debug["trail_reward"] += float(trail_reward)

        route_safety_reward = self._route_safety_reward(
            hero_x, hero_z, min_monster_bfs
        )
        reward += route_safety_reward
        reward_debug["route_safety_reward"] += float(route_safety_reward)

        explore_vector_reward = self._explore_vector_reward(
            hero_x=hero_x,
            hero_z=hero_z,
            move_used=move_used,
            has_prev_pos=has_prev_pos,
        )
        reward += explore_vector_reward
        reward_debug["explore_vector_reward"] += float(explore_vector_reward)

        if (
            self.last_nearest_unexplored_bfs is not None
            and self.last_nearest_unexplored_bfs < Config.BFS_SENTINEL
            and nearest_unexplored_bfs < Config.BFS_SENTINEL
        ):
            explore_delta = float(
                self.last_nearest_unexplored_bfs - nearest_unexplored_bfs
            )
            if explore_delta > 0.0:
                explore_bonus = Config.REWARD_EXPLORE_APPROACH * explore_delta
                reward += explore_bonus
                reward_debug["explore_approach_bonus"] += float(explore_bonus)

        # Resource approach rewards with decay / 资源接近奖励及衰减
        resource_reward_factor = 1.0
        if min_monster_bfs >= Config.REWARD_RESOURCE_SAFE_BFS_HIGH:
            resource_reward_factor = 1.0
        elif min_monster_bfs <= Config.REWARD_RESOURCE_SAFE_BFS_LOW:
            resource_reward_factor = Config.REWARD_RESOURCE_DANGER_FACTOR
        else:
            # Linear decay from SAFE_HIGH to SAFE_LOW / 线性衰减
            resource_reward_factor = Config.REWARD_RESOURCE_DANGER_FACTOR + (
                1.0 - Config.REWARD_RESOURCE_DANGER_FACTOR
            ) * (min_monster_bfs - Config.REWARD_RESOURCE_SAFE_BFS_LOW) / (
                Config.REWARD_RESOURCE_SAFE_BFS_HIGH
                - Config.REWARD_RESOURCE_SAFE_BFS_LOW
            )

        if (
            self.last_nearest_treasure_bfs is not None
            and self.last_nearest_treasure_bfs < Config.BFS_SENTINEL
            and nearest_treasure_bfs < Config.BFS_SENTINEL
        ):
            treasure_delta = float(
                self.last_nearest_treasure_bfs - nearest_treasure_bfs
            )
            if treasure_delta > 0.0:
                resource_bonus = (
                    Config.REWARD_TREASURE_APPROACH
                    * treasure_delta
                    * resource_reward_factor
                )
                reward += resource_bonus
                reward_debug["resource_approach_bonus"] += float(resource_bonus)
        if (
            self.last_nearest_buff_bfs is not None
            and self.last_nearest_buff_bfs < Config.BFS_SENTINEL
            and nearest_buff_bfs < Config.BFS_SENTINEL
        ):
            buff_delta = float(self.last_nearest_buff_bfs - nearest_buff_bfs)
            if buff_delta > 0.0:
                resource_bonus = (
                    Config.REWARD_BUFF_APPROACH * buff_delta * resource_reward_factor
                )
                reward += resource_bonus
                reward_debug["resource_approach_bonus"] += float(resource_bonus)

        if min_monster_bfs < Config.PENALTY_NEAR_MONSTER_THRESHOLD:
            near_monster_penalty = -(
                Config.PENALTY_NEAR_MONSTER_THRESHOLD - min_monster_bfs
            ) * Config.PENALTY_NEAR_MONSTER_SLOPE
            reward += near_monster_penalty
            reward_debug["penalty_near_monster"] = float(near_monster_penalty)

        if bool(env_obs.get("terminated", False)):
            reward += Config.PENALTY_CAUGHT
            reward_debug["penalty_caught"] = float(Config.PENALTY_CAUGHT)

        self.last_nearest_treasure_bfs = float(nearest_treasure_bfs)
        self.last_nearest_buff_bfs = float(nearest_buff_bfs)
        self.last_nearest_unexplored_bfs = float(nearest_unexplored_bfs)

        if not np.isfinite(reward):
            reward = 0.0
        reward_debug.update(self._reward_density_stats(reward_debug))
        self.last_reward_debug = reward_debug
        return float(np.clip(reward, Config.REWARD_CLIP_MIN, Config.REWARD_CLIP_MAX))

    def _monster_pressure_scale(self, min_monster_bfs):
        if min_monster_bfs >= Config.BFS_SENTINEL:
            return 0.0
        full = float(Config.REWARD_MONSTER_PRESSURE_FULL_BFS)
        zero = max(float(Config.REWARD_MONSTER_PRESSURE_ZERO_BFS), full + 1.0e-6)
        if min_monster_bfs <= full:
            return 1.0
        if min_monster_bfs >= zero:
            return 0.0
        return float((zero - min_monster_bfs) / (zero - full))

    def _post_escape_scout_reward(self, hero_x, hero_z, min_monster_bfs, move_used):
        if min_monster_bfs <= Config.POST_ESCAPE_DANGER_BFS:
            self.last_danger_pos = (hero_x, hero_z)
            self.post_escape_until_step = int(
                self.step_no + Config.POST_ESCAPE_SCOUT_STEPS
            )
            self.last_danger_distance = 0.0
            return 0.0

        if (
            not move_used
            or self.last_danger_pos is None
            or self.step_no > self.post_escape_until_step
        ):
            return 0.0

        danger_x, danger_z = self.last_danger_pos
        dist = float(np.hypot(hero_x - danger_x, hero_z - danger_z))
        if self.last_danger_distance is None:
            self.last_danger_distance = dist
            return 0.0

        delta = float(
            np.clip(
                dist - self.last_danger_distance,
                -Config.POST_ESCAPE_PROGRESS_CLIP,
                Config.POST_ESCAPE_PROGRESS_CLIP,
            )
        )
        self.last_danger_distance = dist
        if delta > 0.0:
            return float(Config.REWARD_POST_ESCAPE_PROGRESS * delta)
        if delta < 0.0:
            return float(Config.PENALTY_POST_ESCAPE_REGRESS * abs(delta))
        return 0.0

    def _trail_progress_reward(self, hero_x, hero_z, move_used):
        if not move_used or len(self.position_history) < 4:
            return 0.0

        history = list(self.position_history)[-Config.TRAIL_PROGRESS_WINDOW :]
        positions = history + [(hero_x, hero_z)]
        unique_ratio = float(len(set(positions)) / max(len(positions), 1))
        oldest_x, oldest_z = history[0]
        displacement = float(np.hypot(hero_x - oldest_x, hero_z - oldest_z))

        if displacement >= Config.TRAIL_PROGRESS_DIST:
            return float(Config.REWARD_TRAIL_PROGRESS)
        if (
            displacement <= Config.TRAIL_STALL_DIST
            and unique_ratio <= Config.TRAIL_LOW_UNIQUE_RATIO
        ):
            return float(Config.PENALTY_TRAIL_STALL)
        return 0.0

    def _route_safety_reward(self, hero_x, hero_z, min_monster_bfs):
        open_count = self._open_neighbor_count(hero_x, hero_z)
        reward = 0.0

        if self.last_open_neighbor_count is not None:
            delta = int(open_count - self.last_open_neighbor_count)
            if delta > 0:
                reward += float(Config.REWARD_OPEN_MOVE_DELTA * delta)
            elif (
                delta < 0
                and min_monster_bfs < Config.BFS_SENTINEL
                and min_monster_bfs <= Config.OPEN_SPACE_DANGER_BFS
            ):
                reward += float(Config.PENALTY_OPEN_MOVE_LOSS * abs(delta))

        if (
            min_monster_bfs < Config.BFS_SENTINEL
            and min_monster_bfs <= Config.OPEN_SPACE_DANGER_BFS
            and open_count <= Config.OPEN_SPACE_LOW_NEIGHBORS
        ):
            reward += float(Config.PENALTY_LOW_OPEN_SPACE)

        self.last_open_neighbor_count = int(open_count)
        return reward

    def _open_neighbor_count(self, hero_x, hero_z):
        count = 0
        for dz, dx in _DIRECTIONS_8:
            nz = int(np.clip(hero_z + dz, 0, Config.MAP_SIZE - 1))
            nx = int(np.clip(hero_x + dx, 0, Config.MAP_SIZE - 1))
            if self.global_passable[nz, nx] > 0.5:
                count += 1
        return count

    def _global_explore_progress_reward(self):
        total_cells = float(Config.MAP_SIZE * Config.MAP_SIZE)
        explored_cells = float(np.sum(self.global_explored > 0.5))
        explored_ratio = float(
            np.clip(explored_cells / max(total_cells, 1.0), 0.0, 1.0)
        )
        target_steps = max(float(Config.GLOBAL_EXPLORE_TARGET_STEPS), 1.0)
        target_ratio = float(np.clip(float(self.step_no) / target_steps, 0.0, 1.0))
        gap = float(explored_ratio - target_ratio)

        if gap >= 0.0:
            reward = Config.REWARD_GLOBAL_EXPLORE_ON_TRACK + (
                min(gap, Config.GLOBAL_EXPLORE_GAP_CLIP)
                * Config.REWARD_GLOBAL_EXPLORE_AHEAD_COEF
            )
        else:
            reward = max(
                Config.PENALTY_GLOBAL_EXPLORE_LAG_MAX,
                gap * Config.PENALTY_GLOBAL_EXPLORE_LAG_COEF,
            )

        self.last_global_explore_ratio = float(explored_ratio)
        self.last_global_explore_target_ratio = float(target_ratio)
        self.last_global_explore_gap = float(gap)
        return float(reward), explored_ratio, target_ratio, gap

    def _global_explore_progress_feature(self):
        total_cells = float(Config.MAP_SIZE * Config.MAP_SIZE)
        explored_cells = float(np.sum(self.global_explored > 0.5))
        explored_ratio = float(
            np.clip(explored_cells / max(total_cells, 1.0), 0.0, 1.0)
        )
        target_steps = max(float(Config.GLOBAL_EXPLORE_TARGET_STEPS), 1.0)
        target_ratio = float(np.clip(float(self.step_no) / target_steps, 0.0, 1.0))
        gap = float(
            np.clip(
                explored_ratio - target_ratio,
                -Config.GLOBAL_EXPLORE_GAP_CLIP,
                Config.GLOBAL_EXPLORE_GAP_CLIP,
            )
        )
        return float(
            (gap + Config.GLOBAL_EXPLORE_GAP_CLIP)
            / (2.0 * Config.GLOBAL_EXPLORE_GAP_CLIP)
        )

    def _reward_density_stats(self, reward_debug):
        values = []
        for key, value in reward_debug.items():
            if key.startswith("reward_component_"):
                continue
            if key in (
                "global_explore_ratio",
                "global_explore_target_ratio",
                "global_explore_gap",
            ):
                continue
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(v) and abs(v) > Config.REWARD_DENSITY_EPS:
                values.append(v)

        positive_count = sum(1 for v in values if v > 0.0)
        negative_count = sum(1 for v in values if v < 0.0)
        total_count = positive_count + negative_count
        return {
            "reward_component_count": float(total_count),
            "reward_component_count_sq": float(total_count * total_count),
            "reward_positive_component_count": float(positive_count),
            "reward_negative_component_count": float(negative_count),
        }

    def _distance_from_history(self, hero_x, hero_z):
        if len(self.position_history) < Config.ANTI_LOOP_WINDOW_STEPS:
            return None
        past_x, past_z = self.position_history[-Config.ANTI_LOOP_WINDOW_STEPS]
        dx = float(hero_x - past_x)
        dz = float(hero_z - past_z)
        return float(np.sqrt(dx * dx + dz * dz))

    def _crop_map(self, src, center_x, center_z, size, fill_value=0.0):
        half = size // 2
        out = np.full((size, size), fill_value, dtype=np.float32)
        r0 = center_z - half
        c0 = center_x - half
        for rr in range(size):
            for cc in range(size):
                gr = r0 + rr
                gc = c0 + cc
                if 0 <= gr < Config.MAP_SIZE and 0 <= gc < Config.MAP_SIZE:
                    out[rr, cc] = src[gr, gc]
        return out.astype(np.float32)
