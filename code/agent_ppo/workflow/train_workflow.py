#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Gorge Chase PPO.
峡谷追猎 PPO 训练工作流。
"""

import os
import time

import numpy as np

from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import SampleData, sample_process
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    # Read user config / 读取用户配置
    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error(
            "usr_conf is None, please check agent_ppo/conf/train_env_conf.toml"
        )
        return

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= 1800:
                agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    FEATURE_TIMING_KEYS = (
        "feature_ms_total",
        "feature_ms_parse",
        "feature_ms_map_update",
        "feature_ms_bfs",
        "feature_ms_targets",
        "feature_ms_tensor",
        "feature_ms_concat",
        "feature_ms_reward",
        "feature_ms_metrics",
    )

    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_get_training_metrics_time = 0
        self.sequence_id_base = (
            os.getpid() % Config.SEQUENCE_ID_PID_MOD
        ) * Config.SEQUENCE_ID_STRIDE

    def run_episodes(self):
        """Run a single episode and yield collected samples.

        执行单局对局并 yield 训练样本。
        """
        while True:
            # Periodically fetch training metrics / 定期获取训练指标
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics is {training_metrics}")

            # Reset env / 重置环境
            env_obs = self.env.reset(self.usr_conf)

            # Disaster recovery / 容灾处理
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            # Reset agent & load latest model / 重置 Agent 并加载最新模型
            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            # Initial observation / 初始观测处理
            obs_data, remain_info = self.agent.observation_process(env_obs)

            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0
            episode_metrics = self._new_episode_metrics()

            self.logger.info(f"Episode {self.episode_cnt} start")

            while not done:
                # Predict action / Agent 推理（随机采样）
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                act = self.agent.action_process(act_data)

                # Step env / 与环境交互
                env_reward, env_obs = self.env.step(act)

                # Disaster recovery / 容灾处理
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                step += 1
                done = terminated or truncated

                # Next observation / 处理下一步观测
                _obs_data, _remain_info = self.agent.observation_process(env_obs)

                # Step reward is computed from the post-step observation.
                reward = np.array(_remain_info.get("reward", [0.0]), dtype=np.float32)
                total_reward += float(reward[0])
                self._update_episode_metrics(
                    episode_metrics, _remain_info.get("metrics", {})
                )

                if done:
                    env_info = env_obs["observation"]["env_info"]
                    total_score = env_info.get("total_score", 0)
                    result_str = "FAIL" if terminated else "SUCCESS"
                    monitor_data = self._finalize_episode_metrics(
                        episode_metrics=episode_metrics,
                        total_reward=total_reward,
                        step=step,
                        terminated=terminated,
                    )

                    self.logger.info(
                        f"[GAMEOVER] episode:{self.episode_cnt} steps:{step} "
                        f"result:{result_str} sim_score:{total_score:.1f} "
                        f"total_reward:{total_reward:.3f} "
                        f"score_delta:{monitor_data['score_delta']:.1f} "
                        f"survival_bonus:{monitor_data['survival_bonus']:.3f} "
                        f"safety_bonus:{monitor_data['safety_bonus']:.3f} "
                        f"monster_bfs_reward:{monitor_data['monster_bfs_reward']:.3f} "
                        f"explore_approach_bonus:{monitor_data['explore_approach_bonus']:.3f} "
                        f"explore_vector_reward:{monitor_data['explore_vector_reward']:.3f} "
                        f"global_explore_reward:{monitor_data['global_explore_reward']:.3f} "
                        f"global_explore_ratio:{monitor_data['global_explore_ratio']:.4f} "
                        f"global_explore_target_ratio:{monitor_data['global_explore_target_ratio']:.4f} "
                        f"global_explore_gap:{monitor_data['global_explore_gap']:.4f} "
                        f"resource_approach_bonus:{monitor_data['resource_approach_bonus']:.3f} "
                        f"flash_reward:{monitor_data['flash_reward']:.3f} "
                        f"penalty_near_monster:{monitor_data['penalty_near_monster']:.3f} "
                        f"penalty_still:{monitor_data['penalty_still']:.3f} "
                        f"penalty_caught:{monitor_data['penalty_caught']:.3f} "
                        f"reward_component_mean:{monitor_data['reward_component_mean']:.4f} "
                        f"reward_component_std:{monitor_data['reward_component_std']:.4f} "
                        f"reward_component_var:{monitor_data['reward_component_var']:.4f} "
                        f"reward_positive_component_mean:{monitor_data['reward_positive_component_mean']:.4f} "
                        f"reward_negative_component_mean:{monitor_data['reward_negative_component_mean']:.4f} "
                        f"feature_ms_total_mean:{monitor_data['feature_ms_total_mean']:.4f} "
                        f"feature_ms_total_max:{monitor_data['feature_ms_total_max']:.4f} "
                        f"feature_ms_bfs_mean:{monitor_data['feature_ms_bfs_mean']:.4f} "
                        f"feature_ms_bfs_max:{monitor_data['feature_ms_bfs_max']:.4f} "
                        f"feature_ms_map_update_mean:{monitor_data['feature_ms_map_update_mean']:.4f} "
                        f"feature_ms_targets_mean:{monitor_data['feature_ms_targets_mean']:.4f} "
                        f"feature_ms_tensor_mean:{monitor_data['feature_ms_tensor_mean']:.4f} "
                        f"feature_ms_concat_mean:{monitor_data['feature_ms_concat_mean']:.4f} "
                        f"feature_ms_reward_mean:{monitor_data['feature_ms_reward_mean']:.4f} "
                        f"feature_ms_metrics_mean:{monitor_data['feature_ms_metrics_mean']:.4f} "
                        f"explore_cells:{monitor_data['explore_cells']:.1f} "
                        f"treasure_count:{monitor_data['treasure_count']:.1f} "
                        f"buff_count:{monitor_data['buff_count']:.1f} "
                        f"flash_rate:{monitor_data['flash_rate']:.4f} "
                        f"flash_success_rate:{monitor_data['flash_success_rate']:.4f} "
                        f"still_rate:{monitor_data['still_rate']:.4f} "
                        f"avg_min_monster_bfs:{monitor_data['avg_min_monster_bfs']:.4f} "
                        f"visible_monster_count:{monitor_data['visible_monster_count']:.4f} "
                        f"legal_action_count:{monitor_data['legal_action_count']:.4f} "
                        f"legal_move_count:{monitor_data['legal_move_count']:.4f} "
                        f"legal_flash_count:{monitor_data['legal_flash_count']:.4f} "
                        f"chosen_action_is_legal:{monitor_data['chosen_action_is_legal']:.4f} "
                        f"move_action_rate:{monitor_data['move_action_rate']:.4f} "
                        f"position_changed_rate:{monitor_data['position_changed_rate']:.4f} "
                        f"wall_or_still_after_move_rate:{monitor_data['wall_or_still_after_move_rate']:.4f} "
                        f"caught_rate:{monitor_data['caught_rate']:.4f} "
                        f"post500_survival_rate:{monitor_data['post500_survival_rate']:.4f}"
                    )

                aux_info = remain_info.get("aux", {})

                # Build sample frame from the state that produced act_data.
                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array([act_data.action[0]], dtype=np.float32),
                    reward=reward,
                    done=np.array([float(done)], dtype=np.float32),
                    reward_sum=np.zeros(1, dtype=np.float32),
                    value=np.array(act_data.value, dtype=np.float32).flatten()[:1],
                    next_value=np.zeros(1, dtype=np.float32),
                    advantage=np.zeros(1, dtype=np.float32),
                    prob=np.array(act_data.prob, dtype=np.float32),
                    monster_pos_target=np.array(
                        aux_info.get("monster_pos_target", np.zeros(2)),
                        dtype=np.float32,
                    ),
                    monster_pos_mask=np.array(
                        aux_info.get("monster_pos_mask", np.zeros(1)), dtype=np.float32
                    ),
                    monster_dist_target=np.array(
                        aux_info.get("monster_dist_target", np.zeros(1)),
                        dtype=np.float32,
                    ),
                    monster_dist_mask=np.array(
                        aux_info.get("monster_dist_mask", np.zeros(1)), dtype=np.float32
                    ),
                    seq_id=np.array(
                        [
                            float(
                                self.sequence_id_base
                                + (self.episode_cnt % Config.SEQUENCE_ID_STRIDE)
                            )
                        ],
                        dtype=np.float32,
                    ),
                    seq_pos=np.array([float(step - 1)], dtype=np.float32),
                )
                collector.append(frame)

                # Episode end / 对局结束
                if done:
                    # Monitor report / 监控上报
                    if self.monitor:
                        self.monitor.put_data({os.getpid(): monitor_data})

                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                # Update state / 状态更新
                obs_data = _obs_data
                remain_info = _remain_info

    def _new_episode_metrics(self):
        metrics = {
            "explore_cells": 0.0,
            "treasure_count": 0.0,
            "buff_count": 0.0,
            "score_delta": 0.0,
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
            "flash_reward": 0.0,
            "penalty_near_monster": 0.0,
            "penalty_still": 0.0,
            "penalty_caught": 0.0,
            "reward_component_count_sum": 0.0,
            "reward_component_count_sq_sum": 0.0,
            "reward_positive_component_count_sum": 0.0,
            "reward_negative_component_count_sum": 0.0,
            "flash_count": 0.0,
            "flash_success_count": 0.0,
            "move_count": 0.0,
            "still_count": 0.0,
            "position_changed_count": 0.0,
            "wall_or_still_after_move_count": 0.0,
            "chosen_action_legal_sum": 0.0,
            "executed_action_count": 0.0,
            "legal_action_count_sum": 0.0,
            "legal_move_count_sum": 0.0,
            "legal_flash_count_sum": 0.0,
            "visible_monster_count_sum": 0.0,
            "min_monster_bfs_sum": 0.0,
            "min_monster_bfs_samples": 0.0,
            "post500_alive": 0.0,
        }
        for key in self.FEATURE_TIMING_KEYS:
            metrics[f"{key}_sum"] = 0.0
            metrics[f"{key}_max"] = 0.0
        return metrics

    def _update_episode_metrics(self, episode_metrics, step_metrics):
        if not isinstance(step_metrics, dict):
            return

        episode_metrics["explore_cells"] += self._metric_float(
            step_metrics, "explore_cells"
        )
        episode_metrics["treasure_count"] += self._metric_float(
            step_metrics, "treasure_count"
        )
        episode_metrics["buff_count"] += self._metric_float(step_metrics, "buff_count")
        episode_metrics["score_delta"] += self._metric_float(
            step_metrics, "score_delta"
        )
        episode_metrics["survival_bonus"] += self._metric_float(
            step_metrics, "survival_bonus"
        )
        episode_metrics["safety_bonus"] += self._metric_float(
            step_metrics, "safety_bonus"
        )
        episode_metrics["monster_bfs_reward"] += self._metric_float(
            step_metrics, "monster_bfs_reward"
        )
        episode_metrics["explore_approach_bonus"] += self._metric_float(
            step_metrics, "explore_approach_bonus"
        )
        episode_metrics["explore_vector_reward"] += self._metric_float(
            step_metrics, "explore_vector_reward"
        )
        episode_metrics["global_explore_reward"] += self._metric_float(
            step_metrics, "global_explore_reward"
        )
        episode_metrics["global_explore_ratio"] = max(
            episode_metrics["global_explore_ratio"],
            self._metric_float(step_metrics, "global_explore_ratio"),
        )
        episode_metrics["global_explore_target_ratio"] = max(
            episode_metrics["global_explore_target_ratio"],
            self._metric_float(step_metrics, "global_explore_target_ratio"),
        )
        if "global_explore_gap" in step_metrics:
            episode_metrics["global_explore_gap"] = self._metric_float(
                step_metrics, "global_explore_gap"
            )
        episode_metrics["resource_approach_bonus"] += self._metric_float(
            step_metrics, "resource_approach_bonus"
        )
        episode_metrics["flash_reward"] += self._metric_float(
            step_metrics, "flash_reward"
        )
        episode_metrics["penalty_near_monster"] += self._metric_float(
            step_metrics, "penalty_near_monster"
        )
        episode_metrics["penalty_still"] += self._metric_float(
            step_metrics, "penalty_still"
        )
        episode_metrics["penalty_caught"] += self._metric_float(
            step_metrics, "penalty_caught"
        )
        episode_metrics["reward_component_count_sum"] += self._metric_float(
            step_metrics, "reward_component_count"
        )
        episode_metrics["reward_component_count_sq_sum"] += self._metric_float(
            step_metrics, "reward_component_count_sq"
        )
        episode_metrics["reward_positive_component_count_sum"] += self._metric_float(
            step_metrics, "reward_positive_component_count"
        )
        episode_metrics["reward_negative_component_count_sum"] += self._metric_float(
            step_metrics, "reward_negative_component_count"
        )
        episode_metrics["move_count"] += self._metric_float(step_metrics, "move_used")
        episode_metrics["flash_count"] += self._metric_float(step_metrics, "flash_used")
        episode_metrics["flash_success_count"] += self._metric_float(
            step_metrics, "flash_success"
        )
        episode_metrics["still_count"] += self._metric_float(step_metrics, "still_step")
        episode_metrics["position_changed_count"] += self._metric_float(
            step_metrics, "position_changed"
        )
        episode_metrics["wall_or_still_after_move_count"] += self._metric_float(
            step_metrics, "wall_or_still_after_move"
        )
        episode_metrics["chosen_action_legal_sum"] += self._metric_float(
            step_metrics, "chosen_action_is_legal"
        )
        episode_metrics["executed_action_count"] += self._metric_float(
            step_metrics, "executed_action_count"
        )
        episode_metrics["legal_action_count_sum"] += self._metric_float(
            step_metrics, "legal_action_count"
        )
        episode_metrics["legal_move_count_sum"] += self._metric_float(
            step_metrics, "legal_move_count"
        )
        episode_metrics["legal_flash_count_sum"] += self._metric_float(
            step_metrics, "legal_flash_count"
        )
        episode_metrics["visible_monster_count_sum"] += self._metric_float(
            step_metrics, "visible_monster_count"
        )
        episode_metrics["post500_alive"] = max(
            episode_metrics["post500_alive"],
            self._metric_float(step_metrics, "post500_alive"),
        )
        for key in self.FEATURE_TIMING_KEYS:
            value = self._metric_float(step_metrics, key)
            episode_metrics[f"{key}_sum"] += value
            episode_metrics[f"{key}_max"] = max(
                episode_metrics[f"{key}_max"],
                value,
            )

        if self._metric_float(step_metrics, "has_visible_monster") > 0.5:
            episode_metrics["min_monster_bfs_sum"] += self._metric_float(
                step_metrics, "min_monster_bfs"
            )
            episode_metrics["min_monster_bfs_samples"] += 1.0

    def _finalize_episode_metrics(
        self, episode_metrics, total_reward, step, terminated
    ):
        safe_steps = max(float(step), 1.0)
        move_count = episode_metrics["move_count"]
        flash_count = episode_metrics["flash_count"]
        monster_samples = max(episode_metrics["min_monster_bfs_samples"], 1.0)
        executed_actions = max(episode_metrics["executed_action_count"], 1.0)
        component_mean = episode_metrics["reward_component_count_sum"] / safe_steps
        component_second_moment = (
            episode_metrics["reward_component_count_sq_sum"] / safe_steps
        )
        component_var = max(
            0.0, component_second_moment - component_mean * component_mean
        )
        component_std = component_var ** 0.5

        monitor_data = {
            "reward": round(float(total_reward), 4),
            "episode_steps": int(step),
            "caught_rate": round(1.0 if terminated else 0.0, 6),
            "post500_survival_rate": round(
                float(episode_metrics["post500_alive"] > 0.5), 6
            ),
            "explore_cells": round(episode_metrics["explore_cells"], 4),
            "treasure_count": round(episode_metrics["treasure_count"], 4),
            "buff_count": round(episode_metrics["buff_count"], 4),
            "score_delta": round(episode_metrics["score_delta"], 4),
            "survival_bonus": round(episode_metrics["survival_bonus"], 4),
            "safety_bonus": round(episode_metrics["safety_bonus"], 4),
            "monster_bfs_reward": round(
                episode_metrics["monster_bfs_reward"], 4
            ),
            "explore_approach_bonus": round(
                episode_metrics["explore_approach_bonus"], 4
            ),
            "explore_vector_reward": round(
                episode_metrics["explore_vector_reward"], 4
            ),
            "global_explore_reward": round(
                episode_metrics["global_explore_reward"], 4
            ),
            "global_explore_ratio": round(
                episode_metrics["global_explore_ratio"], 6
            ),
            "global_explore_target_ratio": round(
                episode_metrics["global_explore_target_ratio"], 6
            ),
            "global_explore_gap": round(episode_metrics["global_explore_gap"], 6),
            "resource_approach_bonus": round(
                episode_metrics["resource_approach_bonus"], 4
            ),
            "flash_reward": round(episode_metrics["flash_reward"], 4),
            "penalty_near_monster": round(
                episode_metrics["penalty_near_monster"], 4
            ),
            "penalty_still": round(episode_metrics["penalty_still"], 4),
            "penalty_caught": round(episode_metrics["penalty_caught"], 4),
            "reward_component_mean": round(component_mean, 6),
            "reward_component_std": round(component_std, 6),
            "reward_component_var": round(component_var, 6),
            "reward_positive_component_mean": round(
                episode_metrics["reward_positive_component_count_sum"] / safe_steps,
                6,
            ),
            "reward_negative_component_mean": round(
                episode_metrics["reward_negative_component_count_sum"] / safe_steps,
                6,
            ),
            "move_action_rate": round(move_count / safe_steps, 6),
            "flash_rate": round(flash_count / safe_steps, 6),
            "flash_success_rate": round(
                episode_metrics["flash_success_count"] / max(flash_count, 1.0), 6
            ),
            "still_rate": round(episode_metrics["still_count"] / safe_steps, 6),
            "position_changed_rate": round(
                episode_metrics["position_changed_count"] / safe_steps, 6
            ),
            "wall_or_still_after_move_rate": round(
                episode_metrics["wall_or_still_after_move_count"]
                / max(move_count, 1.0),
                6,
            ),
            "chosen_action_is_legal": round(
                episode_metrics["chosen_action_legal_sum"] / executed_actions, 6
            ),
            "legal_action_count": round(
                episode_metrics["legal_action_count_sum"] / safe_steps, 6
            ),
            "legal_move_count": round(
                episode_metrics["legal_move_count_sum"] / safe_steps, 6
            ),
            "legal_flash_count": round(
                episode_metrics["legal_flash_count_sum"] / safe_steps, 6
            ),
            "avg_min_monster_bfs": round(
                episode_metrics["min_monster_bfs_sum"] / monster_samples, 6
            ),
            "visible_monster_count": round(
                episode_metrics["visible_monster_count_sum"] / safe_steps, 6
            ),
        }
        for key in self.FEATURE_TIMING_KEYS:
            monitor_data[f"{key}_mean"] = round(
                episode_metrics[f"{key}_sum"] / safe_steps,
                6,
            )
            monitor_data[f"{key}_max"] = round(episode_metrics[f"{key}_max"], 6)
        return monitor_data

    def _metric_float(self, metrics, key, default=0.0):
        try:
            value = float(metrics.get(key, default))
        except (TypeError, ValueError):
            return float(default)
        return value if np.isfinite(value) else float(default)
