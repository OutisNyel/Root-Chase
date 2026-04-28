#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from kaiwudrl.common.utils.train_test_utils import run_train_test

# To run the train_test, modify algorithm_name to one value in algorithm_name_list.
algorithm_name_list = ["ppo", "ppo_adv1", "diy"]
algorithm_name = "ppo"


if __name__ == "__main__":
    run_train_test(
        algorithm_name=algorithm_name,
        algorithm_name_list=algorithm_name_list,
        env_vars={
            "replay_buffer_capacity": "10",
            "preload_ratio": "0.2",
            "train_batch_size": "2",
            "dump_model_freq": "1",
        },
    )
