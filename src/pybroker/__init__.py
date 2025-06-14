"""Global imports."""  # 全局导入

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

# 从缓存模块导入缓存管理函数
from pybroker.cache import (
    clear_caches,  # 清除所有缓存
    clear_data_source_cache,  # 清除数据源缓存
    clear_indicator_cache,  # 清除指标缓存
    clear_model_cache,  # 清除模型缓存
    disable_caches,  # 禁用所有缓存
    disable_data_source_cache,  # 禁用数据源缓存
    disable_indicator_cache,  # 禁用指标缓存
    disable_model_cache,  # 禁用模型缓存
    enable_caches,  # 启用所有缓存
    enable_data_source_cache,  # 启用数据源缓存
    enable_indicator_cache,  # 启用指标缓存
    enable_model_cache,  # 启用模型缓存
)
# 从公共模块导入基础数据结构和枚举
from pybroker.common import BarData, DataCol, Day, FeeMode, PriceType
# 从上下文模块导入执行上下文相关类
from pybroker.context import ExecContext, ExecSignal, PosSizeContext
# 从配置模块导入策略配置
from pybroker.config import StrategyConfig
# 从数据模块导入数据源
from pybroker.data import Alpaca, AlpacaCrypto, YFinance
# 从评估模块导入评估指标和引导结果
from pybroker.eval import EvalMetrics, BootstrapResult
# 从指标模块导入指标相关函数和类
from pybroker.indicator import (
    Indicator,
    IndicatorSet,
    highest,
    indicator,
    lowest,
    returns,
)
# 从模型模块导入模型相关类和函数
from pybroker.model import ModelLoader, ModelSource, ModelTrainer, model
# 从投资组合模块导入交易相关类
from pybroker.portfolio import Entry, Order, Position, Trade
# 从范围模块导入工具函数和参数注册相关函数
from pybroker.scope import (
    disable_logging,
    enable_logging,
    disable_progress_bar,
    enable_progress_bar,
    param,
    register_columns,
    unregister_columns,
)
# 从滑点模块导入随机滑点模型
from pybroker.slippage import RandomSlippageModel
# 从策略模块导入策略和测试结果类
from pybroker.strategy import Strategy, TestResult
# 从向量化模块导入向量化函数
from pybroker.vect import cross, highv, lowv, returnv, sumv

# 临时修复Numba 0.57.0的回归问题
# https://github.com/numba/numba/issues/8940
from numba.np.unsafe import ndarray

__version__ = "1.2.10"  # 库版本号

# PyBroker是一个Python算法交易库，提供回测、指标计算、模型训练和策略评估等功能。
# 该库支持多种数据源，包括Alpaca、YFinance等，并提供了完善的缓存机制以提高性能。
# 核心功能包括回测引擎、技术指标计算、机器学习模型集成和策略评估指标。
