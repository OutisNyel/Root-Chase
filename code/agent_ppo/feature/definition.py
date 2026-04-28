#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Data definitions and GAE/window packing for Gorge Chase PPO-Mamba."""

import numpy as np

from common_python.utils.common_func import create_cls
from agent_ppo.conf.conf import Config


ObsData = create_cls("ObsData", feature=None, legal_action=None)
ActData = create_cls("ActData", action=None, d_action=None, prob=None, value=None)

SEQUENCE_LEN = Config.MAMBA_TBPTT_LEN

SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION * SEQUENCE_LEN,
    legal_action=Config.ACTION_NUM * SEQUENCE_LEN,
    act=SEQUENCE_LEN,
    reward=Config.VALUE_NUM * SEQUENCE_LEN,
    reward_sum=Config.VALUE_NUM * SEQUENCE_LEN,
    done=SEQUENCE_LEN,
    value=Config.VALUE_NUM * SEQUENCE_LEN,
    next_value=Config.VALUE_NUM * SEQUENCE_LEN,
    advantage=Config.VALUE_NUM * SEQUENCE_LEN,
    prob=Config.ACTION_NUM * SEQUENCE_LEN,
    monster_pos_target=2 * SEQUENCE_LEN,
    monster_pos_mask=SEQUENCE_LEN,
    monster_dist_target=SEQUENCE_LEN,
    monster_dist_mask=SEQUENCE_LEN,
    seq_id=1,
    seq_pos=SEQUENCE_LEN,
    seq_mask=SEQUENCE_LEN,
    seq_len=1,
)


def sample_process(list_sample_data):
    """Fill GAE fields, then pack one episode into fixed-length windows."""
    if not list_sample_data:
        return list_sample_data

    for i in range(len(list_sample_data) - 1):
        list_sample_data[i].next_value = list_sample_data[i + 1].value

    _calc_gae(list_sample_data)
    return _pack_sequence_windows(list_sample_data)


def _calc_gae(list_sample_data):
    """Compute Generalized Advantage Estimation."""
    gae = 0.0
    gamma = Config.GAMMA
    lamda = Config.LAMDA
    for sample in reversed(list_sample_data):
        delta = -sample.value + sample.reward + gamma * sample.next_value
        gae = gae * gamma * lamda + delta
        sample.advantage = gae
        sample.reward_sum = gae + sample.value


def _pack_sequence_windows(list_sample_data):
    windows = []
    for start in range(0, len(list_sample_data), SEQUENCE_LEN):
        chunk = list_sample_data[start : start + SEQUENCE_LEN]
        windows.append(_pack_one_window(chunk))
    return windows


def _pack_one_window(chunk):
    seq_len = len(chunk)
    seq_mask = np.zeros(SEQUENCE_LEN, dtype=np.float32)
    seq_mask[:seq_len] = 1.0

    return SampleData(
        obs=_stack_window_field(chunk, "obs", Config.DIM_OF_OBSERVATION),
        legal_action=_stack_window_field(chunk, "legal_action", Config.ACTION_NUM),
        act=_stack_window_field(chunk, "act", 1),
        reward=_stack_window_field(chunk, "reward", Config.VALUE_NUM),
        done=_stack_window_field(chunk, "done", 1),
        reward_sum=_stack_window_field(chunk, "reward_sum", Config.VALUE_NUM),
        value=_stack_window_field(chunk, "value", Config.VALUE_NUM),
        next_value=_stack_window_field(chunk, "next_value", Config.VALUE_NUM),
        advantage=_stack_window_field(chunk, "advantage", Config.VALUE_NUM),
        prob=_stack_window_field(chunk, "prob", Config.ACTION_NUM),
        monster_pos_target=_stack_window_field(chunk, "monster_pos_target", 2),
        monster_pos_mask=_stack_window_field(chunk, "monster_pos_mask", 1),
        monster_dist_target=_stack_window_field(chunk, "monster_dist_target", 1),
        monster_dist_mask=_stack_window_field(chunk, "monster_dist_mask", 1),
        seq_id=_fixed_array(getattr(chunk[0], "seq_id", [0.0]), 1),
        seq_pos=_stack_window_field(chunk, "seq_pos", 1),
        seq_mask=seq_mask,
        seq_len=np.array([float(seq_len)], dtype=np.float32),
    )


def _stack_window_field(chunk, field_name, field_dim):
    values = np.zeros((SEQUENCE_LEN, field_dim), dtype=np.float32)
    for idx, sample in enumerate(chunk):
        values[idx] = _fixed_array(getattr(sample, field_name, None), field_dim)
    return values.reshape(-1)


def _fixed_array(value, field_dim):
    if value is None:
        return np.zeros(field_dim, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    fixed = np.zeros(field_dim, dtype=np.float32)
    copy_len = min(field_dim, arr.shape[0])
    if copy_len > 0:
        fixed[:copy_len] = arr[:copy_len]
    return fixed
