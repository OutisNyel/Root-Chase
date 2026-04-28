#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Configuration for Gorge Chase PPO adv1.
"""


class Config:
    # Map sizes
    MAP_SIZE = 128
    LOCAL_MAP_SIZE = 21
    GLOBAL_MAP_SIZE = 32

    # Observation tensor channels
    LOCAL_CHANNELS = 10
    GLOBAL_CHANNELS = 7
    SCALAR_DIM = 48

    # Flattened observation layout
    LOCAL_FLAT_DIM = LOCAL_CHANNELS * LOCAL_MAP_SIZE * LOCAL_MAP_SIZE
    GLOBAL_FLAT_DIM = GLOBAL_CHANNELS * GLOBAL_MAP_SIZE * GLOBAL_MAP_SIZE
    FEATURE_SPLIT_SHAPE = [LOCAL_FLAT_DIM, GLOBAL_FLAT_DIM, SCALAR_DIM]
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Action space: 8 moves + 8 flash actions
    ACTION_NUM = 16

    # Critic output dimension
    VALUE_NUM = 1

    # Model capacity (large model preference)
    CONV_CHANNEL = 64
    SCALAR_MLP_DIMS = [256, 128]
    FUSION_MLP_DIMS = [512, 256, 128]

    # Preprocessor normalization constants
    MAX_MONSTER_SPEED = 5.0
    MAX_FLASH_CD = 2000.0
    MAX_BUFF_DURATION = 50.0
    MAX_MONSTER_INTERVAL = 2000.0
    MAX_BUFF_REFRESH = 500.0
    MAX_SCORE = 3000.0
    BUFF_COOLDOWN_DEFAULT = 200.0

    # Reward shaping coefficients
    REWARD_SURVIVE = 0.01
    REWARD_MONSTER_DIST = 0.15
    REWARD_TREASURE_GAIN = 1.2
    REWARD_BUFF_GAIN = 0.6
    REWARD_EXPLORE_GAIN = 0.25
    REWARD_FLASH_GOOD = 0.08
    REWARD_FLASH_BAD = -0.03
    REWARD_TERMINATED = -10.0
    REWARD_TRUNCATED = 10.0
    REWARD_CLIP_MIN = -20.0
    REWARD_CLIP_MAX = 20.0
    EXPLORE_GAIN_CLIP = 64.0
    DANGER_DIST_THRESHOLD_NORM = 0.2
    MONSTER_RISK_DECAY = 0.92

    # PPO hyperparameters
    GAMMA = 0.99
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0003
    BETA_START = 0.001
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5
