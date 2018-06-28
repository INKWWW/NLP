#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import sys


# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = '/opt/Flask/logs/tezh/comNameSim'
filename = os.path.join(BASE_DIR, 'comNameSim.log')
if not os.path.isdir(BASE_DIR):
    os.mkdir(BASE_DIR)

# 获取logger实例，如果参数为空则返回root logger
logger = logging.getLogger(__name__)
# 指定日志的最低输出级别
logger.setLevel(logging.INFO)

# 指定logger输出格式
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)s - %(message)s')
# 文件日志
file_handler = logging.FileHandler(filename)
file_handler.setLevel(logging.WARN)
file_handler.setFormatter(formatter)  # 可以通过setFormatter指定输出格式
# 控制台日志
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.formatter = formatter  # 也可以直接给formatter赋值
# 为logger添加的日志处理器
logger.addHandler(file_handler)
logger.addHandler(console_handler)
