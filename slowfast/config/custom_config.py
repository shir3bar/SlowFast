#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""


def add_custom_config(_C):
    # Add your own customized configs.
    _C.DATA.BRIGHTNESS_PROB = 0.3
    _C.DATA.BRIGHTNESS_RATIO = 0.2
    _C.DATA.BLUR_PROB = 0.2
    _C.DATA.VARIANCE_IMG = False
    _C.DATA.VAR_DIM = 1
