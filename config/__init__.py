#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置模块
========

提供项目所有的配置参数管理。

使用方法:
    from config import config, get_config, print_config

    # 访问配置
    print(config.model.base_model_path)
    print(config.training.learning_rate)
"""

from .config import config, get_config, print_config

__all__ = ['config', 'get_config', 'print_config']
