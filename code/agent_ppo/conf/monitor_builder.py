#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Monitor panel configuration builder for Gorge Chase PPO."""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def _add_line_panel(monitor, name, name_en, metric_names, expr_fn=None):
    monitor.add_panel(name=name, name_en=name_en, type="line")
    for metric_name in metric_names:
        expr = expr_fn(metric_name) if expr_fn else f"avg({metric_name}{{}})"
        monitor.add_metric(metrics_name=metric_name, expr=expr)
    return monitor.end_panel()


def build_monitor():
    """Create custom monitor panel configurations for agent_ppo."""
    monitor = MonitorConfigBuilder()
    monitor.title("峡谷追猎")

    # Keep the internal group key as "algorithm" for platform compatibility.
    monitor.add_group(group_name="训练诊断", group_name_en="algorithm")
    _add_line_panel(
        monitor,
        name="PPO损失",
        name_en="ppo_losses",
        metric_names=("total_loss", "policy_loss", "value_loss", "entropy_loss"),
    )
    _add_line_panel(
        monitor,
        name="辅助损失",
        name_en="aux_losses",
        metric_names=("aux_pos_loss", "aux_dist_loss"),
    )
    _add_line_panel(
        monitor,
        name="策略KL",
        name_en="approx_kl",
        metric_names=("approx_kl",),
    )
    _add_line_panel(
        monitor,
        name="裁剪比例",
        name_en="clip_frac",
        metric_names=("clip_frac",),
    )
    _add_line_panel(
        monitor,
        name="梯度范数",
        name_en="grad_norm",
        metric_names=("grad_norm",),
    )
    _add_line_panel(
        monitor,
        name="Advantage",
        name_en="advantage_stats",
        metric_names=("advantage_mean", "advantage_std"),
    )
    _add_line_panel(
        monitor,
        name="序列长度",
        name_en="sequence_lengths",
        metric_names=("sequence_mean_len", "sequence_max_len", "sequence_active_max_len"),
    )
    monitor.end_group()

    monitor.add_group(group_name="对局结果", group_name_en="episode")
    _add_line_panel(
        monitor,
        name="回报步数",
        name_en="reward_steps",
        metric_names=("reward", "episode_steps"),
    )
    _add_line_panel(
        monitor,
        name="奖励分量",
        name_en="reward_debug",
        metric_names=(
            "score_delta",
            "survival_bonus",
            "safety_bonus",
            "monster_bfs_reward",
            "explore_approach_bonus",
            "explore_vector_reward",
            "global_explore_reward",
            "resource_approach_bonus",
            "flash_reward",
            "penalty_near_monster",
            "penalty_still",
            "penalty_caught",
        ),
    )
    _add_line_panel(
        monitor,
        name="结果比例",
        name_en="outcome_rates",
        metric_names=("caught_rate", "post500_survival_rate"),
    )
    _add_line_panel(
        monitor,
        name="Reward density mean",
        name_en="reward_density_mean",
        metric_names=(
            "reward_component_mean",
            "reward_positive_component_mean",
            "reward_negative_component_mean",
        ),
    )
    _add_line_panel(
        monitor,
        name="Reward density std",
        name_en="reward_density_std",
        metric_names=("reward_component_std",),
    )
    monitor.end_group()

    monitor.add_group(group_name="行为指标", group_name_en="behavior")
    _add_line_panel(
        monitor,
        name="探索资源",
        name_en="explore_resources",
        metric_names=("explore_cells", "treasure_count", "buff_count"),
    )
    _add_line_panel(
        monitor,
        name="全图探索进度",
        name_en="global_explore_progress",
        metric_names=(
            "global_explore_ratio",
            "global_explore_target_ratio",
            "global_explore_gap",
        ),
    )
    _add_line_panel(
        monitor,
        name="闪现频率",
        name_en="flash_rate",
        metric_names=("flash_rate",),
    )
    _add_line_panel(
        monitor,
        name="闪现成功",
        name_en="flash_success_rate",
        metric_names=("flash_success_rate",),
    )
    _add_line_panel(
        monitor,
        name="静止比例",
        name_en="still_rate",
        metric_names=("still_rate",),
    )
    _add_line_panel(
        monitor,
        name="怪物压力",
        name_en="monster_pressure",
        metric_names=("avg_min_monster_bfs", "visible_monster_count"),
    )
    _add_line_panel(
        monitor,
        name="合法动作数",
        name_en="legal_action_counts",
        metric_names=("legal_action_count", "legal_move_count", "legal_flash_count"),
    )
    _add_line_panel(
        monitor,
        name="动作合法率",
        name_en="chosen_action_is_legal",
        metric_names=("chosen_action_is_legal",),
    )
    _add_line_panel(
        monitor,
        name="移动频率",
        name_en="move_action_rate",
        metric_names=("move_action_rate",),
    )
    _add_line_panel(
        monitor,
        name="位移成功",
        name_en="position_changed_rate",
        metric_names=("position_changed_rate",),
    )
    _add_line_panel(
        monitor,
        name="移动卡住",
        name_en="wall_or_still_after_move_rate",
        metric_names=("wall_or_still_after_move_rate",),
    )
    monitor.end_group()

    monitor.add_group(group_name="Performance", group_name_en="performance")
    _add_line_panel(
        monitor,
        name="Feature timing mean",
        name_en="feature_timing_mean",
        metric_names=(
            "feature_ms_total_mean",
            "feature_ms_bfs_mean",
            "feature_ms_map_update_mean",
            "feature_ms_targets_mean",
            "feature_ms_tensor_mean",
            "feature_ms_reward_mean",
            "feature_ms_metrics_mean",
        ),
    )
    _add_line_panel(
        monitor,
        name="Feature timing max",
        name_en="feature_timing_max",
        metric_names=(
            "feature_ms_total_max",
            "feature_ms_bfs_max",
            "feature_ms_map_update_max",
            "feature_ms_targets_max",
            "feature_ms_tensor_max",
            "feature_ms_reward_max",
            "feature_ms_metrics_max",
        ),
    )
    monitor.end_group()

    return monitor.build()
