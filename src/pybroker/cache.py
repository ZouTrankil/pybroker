"""Contains caching utilities."""  # 包含缓存工具

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import os
from pybroker.scope import StaticScope
from dataclasses import dataclass
from datetime import datetime
from diskcache import Cache
from typing import Final, Optional

_DEFAULT_CACHE_DIRNAME: Final = ".pybrokercache"  # 默认缓存目录名


@dataclass(frozen=True)
class CacheDateFields:
    """用于键缓存数据的日期字段
    
    属性:
        start_date: 缓存数据的开始日期
        end_date: 缓存数据的结束日期
        tf_seconds: 缓存数据的时间框架分辨率(以秒为单位)
        between_time: 用于过滤缓存数据的一天中的时间(例如9:00-9:30 AM)的元组
        days: 用于过滤缓存数据的星期(例如"mon"、"tues"等)
    """

    start_date: datetime  # 开始日期
    end_date: datetime  # 结束日期
    tf_seconds: int  # 时间框架秒数
    between_time: Optional[tuple[str, str]]  # 时间范围
    days: Optional[tuple[int]]  # 星期


@dataclass(frozen=True)
class DataSourceCacheKey:
    """用于DataSource数据的缓存键"""

    symbol: str  # 股票代码
    tf_seconds: int  # 时间框架秒数
    start_date: datetime  # 开始日期
    end_date: datetime  # 结束日期
    adjust: Optional[str]  # 调整类型


@dataclass(frozen=True)
class IndicatorCacheKey:
    """用于指标数据的缓存键"""

    symbol: str  # 股票代码
    tf_seconds: int  # 时间框架秒数
    start_date: datetime  # 开始日期
    end_date: datetime  # 结束日期
    between_time: Optional[tuple[str, str]]  # 时间范围
    days: Optional[tuple[int]]  # 星期
    ind_name: str  # 指标名称


@dataclass(frozen=True)
class ModelCacheKey:
    """用于训练模型的缓存键"""

    symbol: str  # 股票代码
    tf_seconds: int  # 时间框架秒数
    start_date: datetime  # 开始日期
    end_date: datetime  # 结束日期
    between_time: Optional[tuple[str, str]]  # 时间范围
    days: Optional[tuple[int]]  # 星期
    model_name: str  # 模型名称


def _get_cache_dir(
    cache_dir: Optional[str], namespace: str, sub_dir: str
) -> str:
    """获取缓存目录路径"""
    if not namespace:
        raise ValueError("Cache namespace cannot be empty.")
    base_dir = (
        os.path.join(os.getcwd(), _DEFAULT_CACHE_DIRNAME)
        if cache_dir is None
        else cache_dir
    )
    return os.path.join(base_dir, namespace, sub_dir)


def enable_data_source_cache(
    namespace: str, cache_dir: Optional[str] = None
) -> Cache:
    r"""启用从DataSource获取的数据的缓存
    
    参数:
        namespace: 缓存的命名空间
        cache_dir: 用于存储缓存数据的目录
        
    返回:
        diskcache.Cache实例
    """
    scope = StaticScope.instance()
    cache_dir = _get_cache_dir(cache_dir, namespace, "data_source")
    scope.data_source_cache_ns = namespace
    cache = Cache(directory=cache_dir)
    scope.data_source_cache = cache
    scope.logger.debug_enable_data_source_cache(namespace, cache_dir)
    return cache


def disable_data_source_cache():
    r"""禁用从DataSource获取的数据的缓存"""
    scope = StaticScope.instance()
    scope.data_source_cache = None
    scope.data_source_cache_ns = ""
    scope.logger.debug_disable_data_source_cache()


def clear_data_source_cache():
    r"""清除从DataSource缓存的数据。必须先调用enable_data_source_cache才能清除"""
    scope = StaticScope.instance()
    cache = scope.data_source_cache
    if cache is None:
        raise ValueError(
            "Data source cache needs to be enabled before clearing."
        )
    cache.clear()
    scope.logger.debug_clear_data_source_cache(cache.directory)


def enable_indicator_cache(
    namespace: str, cache_dir: Optional[str] = None
) -> Cache:
    """启用指标数据的缓存
    
    参数:
        namespace: 缓存的命名空间
        cache_dir: 用于存储缓存指标数据的目录
        
    返回:
        diskcache.Cache实例
    """
    scope = StaticScope.instance()
    cache_dir = _get_cache_dir(cache_dir, namespace, "indicator")
    scope.indicator_cache_ns = namespace
    cache = Cache(directory=cache_dir)
    scope.indicator_cache = cache
    scope.logger.debug_enable_indicator_cache(namespace, cache_dir)
    return cache


def disable_indicator_cache():
    """禁用指标数据的缓存"""
    scope = StaticScope.instance()
    scope.indicator_cache = None
    scope.indicator_cache_ns = ""
    scope.logger.debug_disable_indicator_cache()


def clear_indicator_cache():
    """清除缓存的指标数据。必须先调用enable_indicator_cache才能清除"""
    scope = StaticScope.instance()
    cache = scope.indicator_cache
    if cache is None:
        raise ValueError(
            "Indicator cache needs to be enabled before clearing."
        )
    cache.clear()
    scope.logger.debug_clear_indicator_cache(cache.directory)


def enable_model_cache(
    namespace: str, cache_dir: Optional[str] = None
) -> Cache:
    """启用训练模型的缓存
    
    参数:
        namespace: 缓存的命名空间
        cache_dir: 用于存储缓存模型的目录
        
    返回:
        diskcache.Cache实例
    """
    scope = StaticScope.instance()
    cache_dir = _get_cache_dir(cache_dir, namespace, "model")
    scope.model_cache_ns = namespace
    cache = Cache(directory=cache_dir)
    scope.model_cache = cache
    scope.logger.debug_enable_model_cache(namespace, cache_dir)
    return cache


def disable_model_cache():
    """禁用训练模型的缓存"""
    scope = StaticScope.instance()
    scope.model_cache = None
    scope.model_cache_ns = ""
    scope.logger.debug_disable_model_cache()


def clear_model_cache():
    """清除缓存的训练模型。必须先调用enable_model_cache才能清除"""
    scope = StaticScope.instance()
    cache = scope.model_cache
    if cache is None:
        raise ValueError("Model cache needs to be enabled before clearing.")
    cache.clear()
    scope.logger.debug_clear_model_cache(cache.directory)


def enable_caches(namespace, cache_dir: Optional[str] = None):
    """启用所有缓存（数据源、指标和模型）
    
    参数:
        namespace: 缓存的命名空间
        cache_dir: 用于存储缓存数据的目录
    """
    enable_data_source_cache(namespace, cache_dir)
    enable_indicator_cache(namespace, cache_dir)
    enable_model_cache(namespace, cache_dir)


def disable_caches():
    """禁用所有缓存（数据源、指标和模型）"""
    disable_data_source_cache()
    disable_indicator_cache()
    disable_model_cache()


def clear_caches():
    """清除所有缓存（数据源、指标和模型）"""
    try:
        clear_data_source_cache()
    except ValueError:
        pass
    try:
        clear_indicator_cache()
    except ValueError:
        pass
    try:
        clear_model_cache()
    except ValueError:
        pass

# 该模块提供了PyBroker的缓存功能，用于存储和检索数据源、指标和模型数据。
# 缓存机制可以显著提高回测性能，避免重复计算和重复获取数据。
# 每种缓存类型（数据源、指标、模型）都有对应的启用、禁用和清除函数。
# 缓存使用命名空间和目录路径组织，确保不同项目或测试之间的缓存隔离。
# 所有缓存操作都由StaticScope单例管理，确保整个应用程序中缓存状态的一致性。
