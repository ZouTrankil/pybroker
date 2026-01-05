"""
技术指标模块 - 包含PyBroker的技术指标相关功能。

本模块提供了技术指标的创建、计算和管理功能，是量化策略开发的核心组件之一。

主要组件：
=========
1. Indicator - 技术指标类，封装指标计算逻辑
2. indicator() - 创建并注册指标的工厂函数
3. IndicatorsMixin - 指标计算的混入类，提供缓存和并行计算支持
4. IndicatorSet - 批量管理和计算多个指标

内置指标：
=========
趋势类指标：
    - linear_trend: 线性趋势强度
    - quadratic_trend: 二次趋势强度
    - cubic_trend: 三次趋势强度
    - adx: 平均动向指数 (Average Directional Index)
    - aroon_up/down/diff: 阿隆指标

动量类指标：
    - macd: 移动平均收敛散度
    - stochastic: 随机指标
    - stochastic_rsi: 随机RSI
    - detrended_rsi: 去趋势RSI
    - laguerre_rsi: 拉盖尔RSI

成交量类指标：
    - money_flow: 资金流向
    - intraday_intensity: 日内强度
    - normalized_on_balance_volume: 归一化OBV
    - volume_momentum: 成交量动量

价格类指标：
    - highest/lowest: 滚动最高/最低价
    - returns: 收益率
    - close_minus_ma: 收盘价减均线
    - price_intensity: 价格强度

使用示例：
=========
::

    import pybroker as pb
    from pybroker import indicator
    
    # 自定义指标：20日简单移动平均
    def sma_20(data):
        return pd.Series(data.close).rolling(20).mean().values
    
    sma_indicator = indicator('sma_20', sma_20)
    
    # 使用内置指标
    macd_ind = pb.macd('my_macd', short_length=12, long_length=26)
    
    # 在策略中使用
    strategy.add_execution(my_fn, 'AAPL', indicators=[sma_indicator, macd_ind])
"""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import functools
import itertools
import numpy as np
import operator as op
import pandas as pd
import pybroker.vect as vect
from pybroker.cache import CacheDateFields, IndicatorCacheKey
from pybroker.common import BarData, DataCol, IndicatorSymbol, default_parallel
from pybroker.eval import iqr, relative_entropy
from pybroker.scope import StaticScope
from pybroker.vect import highv, lowv, returnv
from collections import defaultdict
from dataclasses import asdict
from joblib import delayed
from numpy.typing import NDArray
from typing import (
    Any,
    Callable,
    Collection,
    Iterable,
    Mapping,
    Optional,
    Union,
)


def _to_bar_data(df: pd.DataFrame) -> BarData:
    """
    将pandas DataFrame转换为BarData对象。
    
    BarData是PyBroker内部使用的K线数据结构，包含OHLCV等字段。
    此函数用于将用户提供的DataFrame转换为统一的内部格式。
    
    参数:
        df: 包含K线数据的DataFrame，必须包含date, open, high, low, close列
        
    返回:
        BarData对象，包含转换后的numpy数组
        
    异常:
        ValueError: 如果DataFrame缺少必需的列
    """
    df = df.reset_index()
    # 必需的列：日期、开盘价、最高价、最低价、收盘价
    required_cols = (
        DataCol.DATE,
        DataCol.OPEN,
        DataCol.HIGH,
        DataCol.LOW,
        DataCol.CLOSE,
    )
    # 验证必需列是否存在
    for col in required_cols:
        if col.value not in df.columns:
            raise ValueError(
                f"DataFrame is missing required column: {col.value}"
            )
    return BarData(
        # 转换必需列
        **{col.value: df[col.value].to_numpy() for col in required_cols},
        # 转换可选列（成交量和VWAP）
        **{
            col.value: (
                df[col.value].to_numpy() if col.value in df.columns else None
            )
            for col in (DataCol.VOLUME, DataCol.VWAP)
        },  # type: ignore[arg-type]
        # 转换用户自定义列
        **{
            col: df[col].to_numpy() if col in df.columns else None
            for col in StaticScope.instance().custom_data_cols
        },  # type: ignore[arg-type]
    )


class Indicator:
    """
    技术指标类 - 封装指标计算逻辑的核心类。
    
    Indicator对象封装了一个指标计算函数及其参数，可以被应用于K线数据
    来计算指标值。指标值作为pandas Series返回，索引为日期。
    
    指标计算函数签名::
    
        def my_indicator(data: BarData, **kwargs) -> NDArray[np.float64]:
            # data包含: date, open, high, low, close, volume, vwap等
            # 返回一维numpy数组，长度与输入数据相同
            pass
    
    使用示例::
    
        # 创建简单移动平均指标
        def sma(data: BarData, period: int):
            return pd.Series(data.close).rolling(period).mean().values
        
        sma_20 = Indicator('sma_20', sma, {'period': 20})
        
        # 计算指标值
        values = sma_20(bar_data)  # 返回pd.Series
    
    参数:
        name: 指标的唯一名称，用于在策略中引用
        fn: 指标计算函数，接收BarData和关键字参数，返回一维numpy数组
        kwargs: 传递给计算函数的关键字参数字典
    
    属性:
        name: 指标名称
    """

    def __init__(
        self,
        name: str,  # 指标的唯一标识名称
        fn: Callable[..., NDArray[np.float64]],  # 指标计算函数
        kwargs: dict[str, Any],  # 计算函数的参数
    ):
        self.name = name
        # 使用functools.partial预绑定参数，提高调用效率
        self._fn = functools.partial(fn, **kwargs)
        self._kwargs = kwargs

    def relative_entropy(self, data: Union[BarData, pd.DataFrame]) -> float:
        """
        计算指标值的相对熵（信息熵）。
        
        相对熵衡量数据分布的不确定性或信息量。
        较高的相对熵表示数据分布更均匀，较低则表示分布更集中。
        
        可用于评估指标的信息含量，帮助特征选择。
        
        参数:
            data: K线数据，可以是BarData对象或DataFrame
            
        返回:
            相对熵值（浮点数）
        """
        return relative_entropy(self(data).values)

    def iqr(self, data: Union[BarData, pd.DataFrame]) -> float:
        """
        计算指标值的四分位距（IQR）。
        
        四分位距 = 第75百分位数 - 第25百分位数
        是一种稳健的离散程度度量，不受异常值影响。
        
        可用于评估指标值的波动范围。
        
        参数:
            data: K线数据，可以是BarData对象或DataFrame
            
        返回:
            四分位距值（浮点数）
        """
        return iqr(self(data).values)

    def __call__(self, data: Union[BarData, pd.DataFrame]) -> pd.Series:
        """
        计算指标值。
        
        这是Indicator的主要接口，将K线数据传入后返回计算的指标值Series。
        
        参数:
            data: K线数据，可以是BarData对象或包含OHLCV列的DataFrame
            
        返回:
            pandas Series，索引为日期，值为指标值
            
        异常:
            ValueError: 如果指标函数返回的不是一维数组
        """
        # 如果输入是DataFrame，先转换为BarData
        if isinstance(data, pd.DataFrame):
            data = _to_bar_data(data)
        # 调用指标计算函数
        values = self._fn(data)
        # 确保返回值是numpy数组
        if isinstance(values, pd.Series):
            values = values.to_numpy()
        # 验证返回值是一维数组
        if len(values.shape) != 1:
            raise ValueError(
                f"Indicator {self.name} must return a one-dimensional array."
            )
        # 返回以日期为索引的Series
        return pd.Series(values, index=data.date)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Indicator({self.name!r}, {self._kwargs})"


def indicator(
    name: str, fn: Callable[..., NDArray[np.float64]], **kwargs
) -> Indicator:
    r"""
    创建并注册技术指标的工厂函数。
    
    这是创建自定义指标的推荐方式。该函数会：
    1. 创建一个Indicator实例
    2. 将其注册到全局作用域，使其可在策略中使用
    
    指标计算函数要求::
    
        def my_fn(data: BarData, **kwargs) -> NDArray[np.float64]:
            # data.close: 收盘价数组
            # data.open: 开盘价数组
            # data.high: 最高价数组
            # data.low: 最低价数组
            # data.volume: 成交量数组（可能为None）
            # data.date: 日期数组
            return calculated_values  # 一维numpy数组
    
    使用示例::
    
        import pybroker as pb
        import numpy as np
        
        # 创建布林带中轨指标
        def bollinger_mid(data, period=20):
            return pd.Series(data.close).rolling(period).mean().values
        
        bb_mid = pb.indicator('bb_mid', bollinger_mid, period=20)
        
        # 在策略中使用
        def my_strategy(ctx):
            mid = ctx.indicator('bb_mid')
            if ctx.close[-1] < mid[-1]:
                ctx.buy_shares = 100
        
        strategy.add_execution(my_strategy, 'AAPL', indicators=[bb_mid])
    
    参数:
        name: 指标的唯一名称，用于全局引用和在策略中访问
        fn: 指标计算函数，接收BarData对象作为第一个参数
        \**kwargs: 传递给计算函数的额外参数
        
    返回:
        已注册的Indicator实例
        
    注意:
        - 指标名称必须唯一，重复注册会覆盖之前的指标
        - 计算函数必须返回与输入数据等长的一维数组
    """
    scope = StaticScope.instance()
    indicator = Indicator(name, fn, kwargs)
    scope.set_indicator(indicator)
    return indicator


def _decorate_indicator_fn(ind_name: str):
    """
    装饰指标函数，用于支持并行计算。
    
    该函数将Indicator的__call__方法包装成可被joblib并行调用的形式。
    包装后的函数接收解构的数组参数，避免在进程间传递复杂对象。
    
    参数:
        ind_name: 指标名称
        
    返回:
        装饰后的函数，可用于并行计算
    """
    fn = StaticScope.instance().get_indicator(ind_name).__call__

    def decorated_indicator_fn(
        symbol: str,  # 股票代码
        ind_name: str,  # 指标名称
        date: NDArray[np.datetime64],  # 日期数组
        open: NDArray[np.float64],  # 开盘价数组
        high: NDArray[np.float64],  # 最高价数组
        low: NDArray[np.float64],  # 最低价数组
        close: NDArray[np.float64],  # 收盘价数组
        volume: Optional[NDArray[np.float64]],  # 成交量数组（可选）
        vwap: Optional[NDArray[np.float64]],  # 成交量加权平均价数组（可选）
        custom_col_data: Mapping[str, Optional[NDArray]],  # 用户自定义列数据
    ) -> tuple[IndicatorSymbol, pd.Series]:
        """并行计算单个股票的指标值"""
        # 重建BarData对象
        bar_data = BarData(
            date=date,
            open=open,
            high=high,
            low=low,
            close=close,
            volume=volume,
            vwap=vwap,
            **custom_col_data,
        )
        # 计算指标值
        series = fn(bar_data)
        # 返回(指标符号, 指标值)元组
        return IndicatorSymbol(ind_name, symbol), series

    return decorated_indicator_fn


class IndicatorsMixin:
    """
    指标计算混入类 - 提供指标计算、缓存和并行处理功能。
    
    该混入类被Strategy类继承，为回测提供高效的指标计算能力。
    
    主要功能：
    =========
    1. 批量计算多个股票的多个指标
    2. 支持多进程并行计算，充分利用多核CPU
    3. 支持指标缓存，避免重复计算
    4. 自动管理缓存的读取和写入
    
    缓存机制：
    =========
    当启用指标缓存时（通过pybroker.enable_indicator_cache()），
    计算结果会被保存到磁盘。下次回测使用相同参数时，
    会直接从缓存加载，显著提升回测速度。
    
    缓存键由以下要素组成：
    - 股票代码
    - 指标名称
    - 开始日期
    - 结束日期
    - 时间框架
    
    并行计算：
    =========
    默认情况下，指标计算会使用多进程并行执行。
    每个(指标, 股票)对作为一个独立任务分配给工作进程。
    可以通过disable_parallel=True禁用并行计算。
    """

    def compute_indicators(
        self,
        df: pd.DataFrame,  # 包含所有股票K线数据的DataFrame
        indicator_syms: Iterable[IndicatorSymbol],  # 需要计算的(指标名, 股票代码)对
        cache_date_fields: Optional[CacheDateFields],  # 缓存配置，None表示不缓存
        disable_parallel: bool,  # 是否禁用并行计算
    ) -> dict[IndicatorSymbol, pd.Series]:
        """
        批量计算指标数据。
        
        该方法是指标计算的核心入口，负责：
        1. 检查缓存中是否已有计算结果
        2. 对未缓存的指标进行计算（串行或并行）
        3. 将新计算的结果写入缓存
        
        参数:
            df: 包含所有股票OHLCV数据的DataFrame，必须包含'symbol'列
            indicator_syms: IndicatorSymbol命名元组的可迭代对象，
                           每个元素为(指标名称, 股票代码)
            cache_date_fields: 缓存的日期配置，包含start_date, end_date,
                              tf_seconds等。传None禁用缓存。
            disable_parallel: 如果为True，串行计算（调试时有用）；
                            如果为False，使用多进程并行计算
        
        返回:
            字典，键为IndicatorSymbol，值为对应的指标值Series
            
        示例::
        
            ind_syms = [
                IndicatorSymbol('sma_20', 'AAPL'),
                IndicatorSymbol('sma_20', 'GOOGL'),
                IndicatorSymbol('rsi_14', 'AAPL'),
            ]
            results = self.compute_indicators(df, ind_syms, cache_fields, False)
        """
        if not indicator_syms or df.empty:
            return {}
        scope = StaticScope.instance()
        indicator_data, uncached_ind_syms = self._get_cached_indicators(
            indicator_syms, cache_date_fields
        )
        if not uncached_ind_syms:
            scope.logger.loaded_indicator_data()
            scope.logger.info_loaded_indicator_data(indicator_syms)
            return indicator_data
        if indicator_data:
            scope.logger.info_loaded_indicator_data(indicator_data.keys())
        scope.logger.indicator_data_start(uncached_ind_syms)
        scope.logger.info_indicator_data_start(uncached_ind_syms)
        sym_data: dict[str, dict[str, Optional[NDArray]]] = defaultdict(dict)
        for _, sym in uncached_ind_syms:
            if sym in sym_data:
                continue
            data = df[df[DataCol.SYMBOL.value] == sym]
            for col in scope.all_data_cols:
                if col not in data.columns:
                    sym_data[sym][col] = None
                    continue
                sym_data[sym][col] = data[col].to_numpy()
        for i, (ind_sym, series) in enumerate(
            self._run_indicators(sym_data, uncached_ind_syms, disable_parallel)
        ):
            indicator_data[ind_sym] = series
            self._set_cached_indicator(series, ind_sym, cache_date_fields)
            scope.logger.indicator_data_loading(i + 1)
        return indicator_data

    def _get_cached_indicators(
        self,
        indicator_syms: Iterable[IndicatorSymbol],
        cache_date_fields: Optional[CacheDateFields],
    ) -> tuple[dict[IndicatorSymbol, pd.Series], list[IndicatorSymbol]]:
        """
        从缓存中获取已计算的指标数据。
        
        遍历所有需要的指标，检查缓存中是否已存在。
        返回已缓存的数据和未缓存的指标列表。
        
        参数:
            indicator_syms: 需要的指标符号列表
            cache_date_fields: 缓存配置
            
        返回:
            元组 (已缓存的指标数据字典, 未缓存的指标符号列表)
        """
        indicator_syms = sorted(indicator_syms)
        indicator_data: dict[IndicatorSymbol, pd.Series] = {}
        # 如果未配置缓存，直接返回空结果
        if cache_date_fields is None:
            return indicator_data, indicator_syms
        scope = StaticScope.instance()
        # 如果缓存未启用，直接返回空结果
        if scope.indicator_cache is None:
            return indicator_data, indicator_syms
        uncached_ind_syms = []
        # 遍历每个指标符号，尝试从缓存获取
        for ind_sym in indicator_syms:
            cache_key = IndicatorCacheKey(
                symbol=ind_sym.symbol,
                ind_name=ind_sym.ind_name,
                **asdict(cache_date_fields),
            )
            scope.logger.debug_get_indicator_cache(cache_key)
            data = scope.indicator_cache.get(repr(cache_key))
            if data is not None:
                # 缓存命中
                indicator_data[ind_sym] = data
            else:
                # 缓存未命中，加入待计算列表
                uncached_ind_syms.append(ind_sym)
        return indicator_data, uncached_ind_syms

    def _set_cached_indicator(
        self,
        series: pd.Series,
        ind_sym: IndicatorSymbol,
        cache_date_fields: Optional[CacheDateFields],
    ):
        """
        将计算的指标数据写入缓存。
        
        参数:
            series: 计算得到的指标值Series
            ind_sym: 指标符号
            cache_date_fields: 缓存配置
        """
        # 如果未配置缓存，直接返回
        if cache_date_fields is None:
            return
        scope = StaticScope.instance()
        # 如果缓存未启用，直接返回
        if scope.indicator_cache is None:
            return
        # 构建缓存键
        cache_key = IndicatorCacheKey(
            symbol=ind_sym.symbol,
            ind_name=ind_sym.ind_name,
            **asdict(cache_date_fields),
        )
        scope.logger.debug_set_indicator_cache(cache_key)
        # 写入缓存
        scope.indicator_cache.set(repr(cache_key), series)

    def _run_indicators(
        self,
        sym_data: Mapping[str, Mapping[str, Optional[NDArray]]],
        ind_syms: Collection[IndicatorSymbol],
        disable_parallel: bool,
    ) -> Iterable[tuple[IndicatorSymbol, pd.Series]]:
        """
        执行指标计算（串行或并行）。
        
        根据配置选择串行或并行方式计算指标。
        并行计算使用joblib的多进程后端。
        
        参数:
            sym_data: 按股票代码组织的K线数据字典
            ind_syms: 需要计算的指标符号集合
            disable_parallel: 是否禁用并行计算
            
        返回:
            (指标符号, 指标值Series)元组的可迭代对象
        """
        # 准备每个指标的装饰函数（用于并行调用）
        fns = {}
        for ind_name, _ in ind_syms:
            if ind_name in fns:
                continue
            fns[ind_name] = _decorate_indicator_fn(ind_name)
        scope = StaticScope.instance()

        def args_fn(ind_name, sym):
            """构建指标计算函数的参数字典"""
            return {
                "symbol": sym,
                "ind_name": ind_name,
                "custom_col_data": {
                    col: sym_data[sym][col] for col in scope.custom_data_cols
                },
                **{col: sym_data[sym][col] for col in scope.default_data_cols},
            }

        # 根据配置选择串行或并行计算
        if disable_parallel or len(ind_syms) == 1:
            # 串行计算：适用于调试或只有一个任务的情况
            scope.logger.debug_compute_indicators(is_parallel=False)
            return tuple(
                fns[ind_name](**args_fn(ind_name, sym))
                for ind_name, sym in ind_syms
            )
        else:
            # 并行计算：使用多进程加速
            scope.logger.debug_compute_indicators(is_parallel=True)

            with default_parallel() as parallel:
                return parallel(
                    delayed(fns[ind_name])(**args_fn(ind_name, sym))
                    for ind_name, sym in ind_syms
                )


class IndicatorSet(IndicatorsMixin):
    """
    指标集合类 - 用于批量管理和计算多个技术指标。
    
    IndicatorSet提供了一个便捷的方式来管理多个指标，
    并将它们一次性应用到K线数据上。
    
    与Strategy中的指标使用不同，IndicatorSet适用于：
    - 独立的指标计算（不依赖Strategy）
    - 数据探索和分析
    - 特征工程
    - 批量生成指标特征
    
    使用示例::
    
        import pybroker as pb
        
        # 创建指标
        sma_20 = pb.indicator('sma_20', lambda d: pd.Series(d.close).rolling(20).mean().values)
        sma_50 = pb.indicator('sma_50', lambda d: pd.Series(d.close).rolling(50).mean().values)
        rsi = pb.indicator('rsi', my_rsi_fn, period=14)
        
        # 创建指标集合
        ind_set = pb.IndicatorSet()
        ind_set.add(sma_20, sma_50, rsi)
        
        # 计算指标（返回包含所有指标列的DataFrame）
        result_df = ind_set(price_data_df)
        # result_df包含列: date, symbol, sma_20, sma_50, rsi
    """

    def __init__(self):
        self._ind_names: set[str] = set()  # 存储指标名称的集合

    def add(self, indicators: Union[Indicator, Iterable[Indicator]], *args):
        """
        添加一个或多个指标到集合中。
        
        参数:
            indicators: 单个Indicator或Indicator的可迭代对象
            *args: 额外的Indicator对象
            
        示例::
        
            ind_set.add(sma_20)  # 添加单个
            ind_set.add(sma_20, sma_50)  # 添加多个
            ind_set.add([sma_20, sma_50, rsi])  # 从列表添加
        """
        if isinstance(indicators, Indicator):
            indicators = (indicators, *args)
        else:
            indicators = (*indicators, *args)
        self._ind_names.update(map(op.attrgetter("name"), indicators))

    def remove(self, indicators: Union[Indicator, Iterable[Indicator]], *args):
        """
        从集合中移除一个或多个指标。
        
        参数:
            indicators: 单个Indicator或Indicator的可迭代对象
            *args: 额外的Indicator对象
        """
        if isinstance(indicators, Indicator):
            indicators = (indicators, *args)
        else:
            indicators = (*indicators, *args)
        self._ind_names.difference_update(
            map(op.attrgetter("name"), indicators)
        )

    def clear(self):
        """清空集合中的所有指标。"""
        self._ind_names.clear()

    def __call__(
        self, df: pd.DataFrame, disable_parallel: bool = False
    ) -> pd.DataFrame:
        """
        计算集合中所有指标的数据。
        
        对输入DataFrame中的每个股票计算所有已添加的指标，
        返回包含指标列的DataFrame。
        
        参数:
            df: 输入K线数据，必须包含symbol, date, open, high, low, close列
            disable_parallel: 如果为True，串行计算指标；
                            如果为False，使用多进程并行计算。
                            默认为False。

        返回:
            DataFrame，包含以下列：
            - date: 日期
            - symbol: 股票代码
            - 每个指标名称对应的列
            
        异常:
            ValueError: 如果没有添加任何指标
        """
        if not self._ind_names:
            raise ValueError("No indicators were added.")
        if df.empty:
            return pd.DataFrame(
                columns=[DataCol.DATE.value, DataCol.SYMBOL.value]
                + list(self._ind_names)
            )
        syms = df[DataCol.SYMBOL.value].unique()
        ind_syms = tuple(
            itertools.starmap(
                IndicatorSymbol, itertools.product(self._ind_names, syms)
            )
        )
        ind_dict = self.compute_indicators(
            df=df,
            indicator_syms=ind_syms,
            cache_date_fields=None,
            disable_parallel=disable_parallel,
        )
        sym_dict: dict[str, dict[str, pd.Series]] = defaultdict(dict)
        for ind_sym, series in ind_dict.items():
            sym_dict[ind_sym.symbol][ind_sym.ind_name] = series
        data: dict[str, list] = defaultdict(list)
        for sym, ind_series in sym_dict.items():
            dates = df[df[DataCol.SYMBOL.value] == sym][DataCol.DATE.value]
            data[DataCol.SYMBOL.value].extend(
                itertools.repeat(sym, len(dates))
            )
            data[DataCol.DATE.value].extend(dates)
            for ind_name, series in ind_series.items():
                data[ind_name].extend(series.values)
        return pd.DataFrame.from_dict(data)


def highest(name: str, field: str, period: int) -> Indicator:
    """创建一个滚动高点 :class:`.Indicator`。

    参数:
        name: 指标名称。
        field: 用于计算滚动高点的 :class:`pybroker.common.BarData` 字段。
        period: 回看周期。

    返回:
        滚动高点 :class:`.Indicator`。
    """

    def _highest(data: BarData):
        values = getattr(data, field)
        return highv(values, period)

    return indicator(name, _highest)


def lowest(name: str, field: str, period: int) -> Indicator:
    """创建一个滚动低点 :class:`.Indicator`。

    参数:
        name: 指标名称。
        field: 用于计算滚动低点的 :class:`pybroker.common.BarData` 字段。
        period: 回看周期。

    返回:
        滚动低点 :class:`.Indicator`。
    """

    def _lowest(data: BarData):
        values = getattr(data, field)
        return lowv(values, period)

    return indicator(name, _lowest)


def returns(name: str, field: str, period: int = 1) -> Indicator:
    """创建一个滚动回报率 :class:`.Indicator`。

    参数:
        name: 指标名称。
        field: 用于计算滚动回报率的 :class:`pybroker.common.BarData` 字段。
        period: 回报率周期。默认为 1。

    返回:
        滚动回报率 :class:`.Indicator`。
    """

    def _returns(data: BarData):
        values = getattr(data, field)
        return returnv(values, period)

    return indicator(name, _returns)


def detrended_rsi(
    name: str, field: str, short_length: int, long_length: int, reg_length: int
) -> Indicator:
    """去趋势相对强弱指数 (RSI)。

    参数:
        name: 指标名称。
        field: :class:`pybroker.common.BarData` 字段名称。
        short_length: 短期 RSI 的回看期。
        long_length: 长期 RSI 的回看期。
        reg_length: 用于线性回归的K线数量。

    返回:
        去趋势 RSI :class:`.Indicator`。
    """

    def _detrended_rsi(data: BarData):
        values = getattr(data, field)
        return vect.detrended_rsi(
            values,
            short_length=short_length,
            long_length=long_length,
            reg_length=reg_length,
        )

    return indicator(name, _detrended_rsi)


def macd(
    name: str,
    short_length: int,
    long_length: int,
    smoothing: float = 0.0,
    scale: float = 1.0,
) -> Indicator:
    """移动平均收敛散度 (MACD)。

    参数:
        name: 指标名称。
        short_length: 短期回看期。
        long_length: 长期回看期。
        smoothing: 如果 >= 2，则计算 MACD 减去平滑值。
        scale: 增加 > 1.0 会对返回值进行更多压缩，
            减少 < 1.0 则压缩更少。默认为 ``1.0``。

    返回:
        移动平均收敛散度 :class:`.Indicator`。
    """

    def _macd(data: BarData):
        return vect.macd(
            high=data.high,
            low=data.low,
            close=data.close,
            short_length=short_length,
            long_length=long_length,
            smoothing=smoothing,
            scale=scale,
        )

    return indicator(name, _macd)


def stochastic(name: str, lookback: int, smoothing: int = 0) -> Indicator:
    """随机指标。

    参数:
        name: 指标名称。
        lookback: 回看K线数量。
        smoothing: 原始随机指标被平滑的次数，可以是 0、
            1 或 2 次。默认为 ``0``。

    返回:
        随机指标 :class:`.Indicator`。
    """

    def _stochastic(data: BarData):
        return vect.stochastic(
            high=data.high,
            low=data.low,
            close=data.close,
            lookback=lookback,
            smoothing=smoothing,
        )

    return indicator(name, _stochastic)


def stochastic_rsi(
    name: str,
    field: str,
    rsi_lookback: int,
    sto_lookback: int,
    smoothing: float = 0.0,
) -> Indicator:
    """随机相对强弱指数 (RSI)。

    参数:
        name: 指标名称。
        field: :class:`pybroker.common.BarData` 字段名称。
        rsi_lookback: RSI 计算的回看长度。
        sto_lookback: 随机指标计算的回看长度。
        smoothing: 平滑量；<= 1 表示不平滑。默认为 ``0``。

    返回:
        随机 RSI :class:`.Indicator`。
    """

    def _stochastic_rsi(data: BarData):
        values = getattr(data, field)
        return vect.stochastic_rsi(
            values,
            rsi_lookback=rsi_lookback,
            sto_lookback=sto_lookback,
            smoothing=smoothing,
        )

    return indicator(name, _stochastic_rsi)


def linear_trend(
    name: str, field: str, lookback: int, atr_length: int, scale: float = 1.0
) -> Indicator:
    """线性趋势强度。

    参数:
        name: 指标名称。
        field: :class:`pybroker.common.BarData` 字段名称。
        lookback: 回看K线数量。
        atr_length: 用于平均真实波幅 (ATR)
            归一化的回看长度。
        scale: 增加 > 1.0 会对返回值进行更多压缩，
            减少 < 1.0 则压缩更少。默认为 ``1.0``。

    返回:
        线性趋势强度 :class:`.Indicator`。
    """

    def _linear_trend(data: BarData):
        values = getattr(data, field)
        return vect.linear_trend(
            values,
            high=data.high,
            low=data.low,
            close=data.close,
            lookback=lookback,
            atr_length=atr_length,
            scale=scale,
        )

    return indicator(name, _linear_trend)


def quadratic_trend(
    name: str, field: str, lookback: int, atr_length: int, scale: float = 1.0
) -> Indicator:
    """二次趋势强度。

    参数:
        name: 指标名称。
        field: :class:`pybroker.common.BarData` 字段名称。
        lookback: 回看K线数量。
        atr_length: 用于平均真实波幅 (ATR)
            归一化的回看长度。
        scale: 增加 > 1.0 会对返回值进行更多压缩，
            减少 < 1.0 则压缩更少。默认为 ``1.0``。

    返回:
        二次趋势强度 :class:`.Indicator`。
    """

    def _quadratic_trend(data: BarData):
        values = getattr(data, field)
        return vect.quadratic_trend(
            values,
            high=data.high,
            low=data.low,
            close=data.close,
            lookback=lookback,
            atr_length=atr_length,
            scale=scale,
        )

    return indicator(name, _quadratic_trend)


def cubic_trend(
    name: str, field: str, lookback: int, atr_length: int, scale: float = 1.0
) -> Indicator:
    """三次趋势强度。

    参数:
        name: 指标名称。
        field: :class:`pybroker.common.BarData` 字段名称。
        lookback: 回看K线数量。
        atr_length: 用于平均真实波幅 (ATR)
            归一化的回看长度。
        scale: 增加 > 1.0 会对返回值进行更多压缩，
            减少 < 1.0 则压缩更少。默认为 ``1.0``。

    返回:
        三次趋势强度 :class:`.Indicator`。
    """

    def _cubic_trend(data: BarData):
        values = getattr(data, field)
        return vect.cubic_trend(
            values,
            high=data.high,
            low=data.low,
            close=data.close,
            lookback=lookback,
            atr_length=atr_length,
            scale=scale,
        )

    return indicator(name, _cubic_trend)


def adx(name: str, lookback: int) -> Indicator:
    """平均动向指数。

    参数:
        name: 指标名称。
        lookback: 回看K线数量。

    返回:
        平均动向指数 :class:`.Indicator`。
    """

    def _adx(data: BarData):
        return vect.adx(
            high=data.high, low=data.low, close=data.close, lookback=lookback
        )

    return indicator(name, _adx)


def aroon_up(name: str, lookback: int) -> Indicator:
    """阿隆上升趋势。

    参数:
        name: 指标名称。
        lookback: 回看K线数量。

    返回:
        阿隆上升趋势 :class:`.Indicator`。
    """

    def _aroon_up(data: BarData):
        return vect.aroon_up(high=data.high, low=data.low, lookback=lookback)

    return indicator(name, _aroon_up)


def aroon_down(name: str, lookback: int) -> Indicator:
    """阿隆下降趋势。

    参数:
        name: 指标名称。
        lookback: 回看K线数量。

    返回:
        阿隆下降趋势 :class:`.Indicator`。
    """

    def _aroon_down(data: BarData):
        return vect.aroon_down(high=data.high, low=data.low, lookback=lookback)

    return indicator(name, _aroon_down)


def aroon_diff(name: str, lookback: int) -> Indicator:
    """阿隆上升趋势减去阿隆下降趋势。

    参数:
        name: 指标名称。
        lookback: 回看K线数量。

    返回:
        阿隆上升趋势减去阿隆下降趋势 :class:`.Indicator`。
    """

    def _aroon_diff(data: BarData):
        return vect.aroon_diff(high=data.high, low=data.low, lookback=lookback)

    return indicator(name, _aroon_diff)


def close_minus_ma(
    name: str, lookback: int, atr_length: int, scale: float = 1.0
) -> Indicator:
    """收盘价减移动平均线。

    参数:
        name: 指标名称。
        lookback: 回看K线数量。
        atr_length: 用于平均真实波幅 (ATR)
            归一化的回看长度。
        scale: 增加 > 1.0 会对返回值进行更多压缩，
            减少 < 1.0 则压缩更少。默认为 ``1.0``。

    返回:
        收盘价减移动平均线 :class:`.Indicator`。
    """

    def _close_minus_ma(data: BarData):
        return vect.close_minus_ma(
            high=data.high,
            low=data.low,
            close=data.close,
            lookback=lookback,
            atr_length=atr_length,
            scale=scale,
        )

    return indicator(name, _close_minus_ma)


def linear_deviation(
    name: str, field: str, lookback: int, scale: float = 0.6
) -> Indicator:
    """线性趋势的偏差。

    参数:
        name: 指标名称。
        field: :class:`pybroker.common.BarData` 字段名称。
        lookback: 回看K线数量。
        scale: 增加 > 1.0 会对返回值进行更多压缩，
            减少 < 1.0 则压缩更少。默认为 ``0.6``。

    返回:
        线性趋势的偏差 :class:`.Indicator`。
    """

    def _linear_deviation(data: BarData):
        values = getattr(data, field)
        return vect.linear_deviation(values, lookback=lookback, scale=scale)

    return indicator(name, _linear_deviation)


def quadratic_deviation(
    name: str, field: str, lookback: int, scale: float = 0.6
) -> Indicator:
    """二次趋势的偏差。

    参数:
        name: 指标名称。
        field: :class:`pybroker.common.BarData` 字段名称。
        lookback: 回看K线数量。
        scale: 增加 > 1.0 会对返回值进行更多压缩，
            减少 < 1.0 则压缩更少。默认为 ``0.6``。

    返回:
        二次趋势的偏差 :class:`.Indicator`。
    """

    def _quadratic_deviation(data: BarData):
        values = getattr(data, field)
        return vect.quadratic_deviation(values, lookback=lookback, scale=scale)

    return indicator(name, _quadratic_deviation)


def cubic_deviation(
    name: str, field: str, lookback: int, scale: float = 0.6
) -> Indicator:
    """三次趋势的偏差。

    参数:
        name: 指标名称。
        field: :class:`pybroker.common.BarData` 字段名称。
        lookback: 回看K线数量。
        scale: 增加 > 1.0 会对返回值进行更多压缩，
            减少 < 1.0 则压缩更少。默认为 ``0.6``。

    返回:
        三次趋势的偏差 :class:`.Indicator`。
    """

    def _cubic_deviation(data: BarData):
        values = getattr(data, field)
        return vect.cubic_deviation(values, lookback=lookback, scale=scale)

    return indicator(name, _cubic_deviation)


def price_intensity(
    name: str, smoothing: float = 0.0, scale: float = 0.8
) -> Indicator:
    """价格强度。

    参数:
        name: 指标名称。
        smoothing: 平滑量。默认为 ``0``。
        scale: 增加 > 1.0 会对返回值进行更多压缩，
            减少 < 1.0 则压缩更少。默认为 ``0.8``。

    返回:
        价格强度 :class:`.Indicator`。
    """

    def _price_intensity(data: BarData):
        return vect.price_intensity(
            open=data.open,
            high=data.high,
            low=data.low,
            close=data.close,
            smoothing=smoothing,
            scale=scale,
        )

    return indicator(name, _price_intensity)


def price_change_oscillator(
    name: str, short_length: int, multiplier: int, scale: float = 4.0
) -> Indicator:
    """价格变动振荡器。

    参数:
        name: 指标名称。
        short_length: 短期回看K线数量。
        multiplier: 用于计算长期回看K线数量的乘数 =
            ``multiplier * short_length``。
        scale: 增加 > 1.0 会对返回值进行更多压缩，
            减少 < 1.0 则压缩更少。默认为 ``4.0``。

    返回:
        价格变动振荡器 :class:`.Indicator`。
    """

    def _price_change_oscillator(data: BarData):
        return vect.price_change_oscillator(
            high=data.high,
            low=data.low,
            close=data.close,
            short_length=short_length,
            multiplier=multiplier,
            scale=scale,
        )

    return indicator(name, _price_change_oscillator)


def intraday_intensity(
    name: str, lookback: int, smoothing: float = 0.0
) -> Indicator:
    """日内强度。

    参数:
        name: 指标名称。
        lookback: 回看K线数量。
        smoothing: 平滑量；<= 1 表示不平滑。默认为 ``0``。

    返回:
        日内强度 :class:`.Indicator`。
    """

    def _intraday_intensity(data: BarData):
        return vect.intraday_intensity(
            high=data.high,
            low=data.low,
            close=data.close,
            volume=data.volume,
            lookback=lookback,
            smoothing=smoothing,
        )

    return indicator(name, _intraday_intensity)


def money_flow(name: str, lookback: int, smoothing: float = 0.0) -> Indicator:
    """蔡金资金流。

    参数:
        name: 指标名称。
        lookback: 回看K线数量。
        smoothing: 平滑量；<= 1 表示不平滑。默认为 ``0``。

    返回:
        蔡金资金流 :class:`.Indicator`。
    """

    def _money_flow(data: BarData):
        return vect.money_flow(
            high=data.high,
            low=data.low,
            close=data.close,
            volume=data.volume,
            lookback=lookback,
            smoothing=smoothing,
        )

    return indicator(name, _money_flow)


def reactivity(
    name: str, lookback: int, smoothing: float = 0.0, scale: float = 0.6
) -> Indicator:
    """反应性。

    参数:
        name: 指标名称。
        lookback: 回看K线数量。
        smoothing: 平滑乘数。
        scale: 增加 > 1.0 会对返回值进行更多压缩，
            减少 < 1.0 则压缩更少。默认为 ``0.6``。

    返回:
        反应性 :class:`.Indicator`。
    """

    def _reactivity(data: BarData):
        return vect.reactivity(
            high=data.high,
            low=data.low,
            close=data.close,
            volume=data.volume,
            lookback=lookback,
            smoothing=smoothing,
            scale=scale,
        )

    return indicator(name, _reactivity)


def price_volume_fit(
    name: str, lookback: int, scale: float = 9.0
) -> Indicator:
    """价量拟合。

    参数:
        name: 指标名称。
        lookback: 回看K线数量。
        scale: 增加 > 1.0 会对返回值进行更多压缩，
            减少 < 1.0 则压缩更少。默认为 ``9.0``。

    返回:
        价量拟合 :class:`.Indicator`。
    """

    def _price_volume_fit(data: BarData):
        return vect.price_volume_fit(
            close=data.close,
            volume=data.volume,
            lookback=lookback,
            scale=scale,
        )

    return indicator(name, _price_volume_fit)


def volume_weighted_ma_ratio(
    name: str, lookback: int, scale: float = 1.0
) -> Indicator:
    """成交量加权移动平均比率。

    参数:
        name: 指标名称。
        lookback: 回看K线数量。
        scale: 增加 > 1.0 会对返回值进行更多压缩，
            减少 < 1.0 则压缩更少。默认为 ``1.0``。

    返回:
        成交量加权移动平均比率 :class:`.Indicator`。
    """

    def _volume_weighted_ma_ratio(data: BarData):
        return vect.volume_weighted_ma_ratio(
            close=data.close,
            volume=data.volume,
            lookback=lookback,
            scale=scale,
        )

    return indicator(name, _volume_weighted_ma_ratio)


def normalized_on_balance_volume(
    name: str, lookback: int, scale: float = 0.6
) -> Indicator:
    """归一化能量潮。

    参数:
        name: 指标名称。
        lookback: 回看K线数量。
        scale: 增加 > 1.0 会对返回值进行更多压缩，
            减少 < 1.0 则压缩更少。默认为 ``0.6``。

    返回:
        归一化能量潮 :class:`.Indicator`。
    """

    def _normalized_on_balance_volume(data: BarData):
        return vect.normalized_on_balance_volume(
            close=data.close,
            volume=data.volume,
            lookback=lookback,
            scale=scale,
        )

    return indicator(name, _normalized_on_balance_volume)


def delta_on_balance_volume(
    name: str, lookback: int, delta_length: int = 0, scale: float = 0.6
) -> Indicator:
    """能量潮变化量。

    参数:
        name: 指标名称。
        lookback: 回看K线数量。
        delta_length: 用于差分的滞后值。
        scale: 增加 > 1.0 会对返回值进行更多压缩，
            减少 < 1.0 则压缩更少。默认为 ``0.6``。

    返回:
        能量潮变化量 :class:`.Indicator`。
    """

    def _delta_on_balance_volume(data: BarData):
        return vect.delta_on_balance_volume(
            close=data.close,
            volume=data.volume,
            lookback=lookback,
            delta_length=delta_length,
            scale=scale,
        )

    return indicator(name, _delta_on_balance_volume)


def normalized_positive_volume_index(
    name: str, lookback: int, scale: float = 0.5
) -> Indicator:
    """归一化正成交量指数。

    参数:
        name: 指标名称。
        lookback: 回看K线数量。
        scale: 增加 > 1.0 会对返回值进行更多压缩，
            减少 < 1.0 则压缩更少。默认为 ``0.5``。

    返回:
        归一化正成交量指数 :class:`.Indicator`。
    """

    def _normalized_positive_volume_index(data: BarData):
        return vect.normalized_positive_volume_index(
            close=data.close,
            volume=data.volume,
            lookback=lookback,
            scale=scale,
        )

    return indicator(name, _normalized_positive_volume_index)


def normalized_negative_volume_index(
    name: str, lookback: int, scale: float = 0.5
) -> Indicator:
    """归一化负成交量指数。

    参数:
        name: 指标名称。
        lookback: 回看K线数量。
        scale: 增加 > 1.0 会对返回值进行更多压缩，
            减少 < 1.0 则压缩更少。默认为 ``0.5``。

    返回:
        归一化负成交量指数 :class:`.Indicator`。
    """

    def _normalized_negative_volume_index(data: BarData):
        return vect.normalized_negative_volume_index(
            close=data.close,
            volume=data.volume,
            lookback=lookback,
            scale=scale,
        )

    return indicator(name, _normalized_negative_volume_index)


def volume_momentum(
    name: str, short_length: int, multiplier: int = 2, scale: float = 3.0
) -> Indicator:
    """成交量动量。

    参数:
        name: 指标名称。
        short_length: 短期回看K线数量。
        multiplier: 回看乘数。默认为 ``2``。
        scale: 增加 > 1.0 会对返回值进行更多压缩，
            减少 < 1.0 则压缩更少。默认为 ``3.0``。

    返回:
        成交量动量 :class:`.Indicator`。
    """

    def _volume_momentum(data: BarData):
        return vect.volume_momentum(
            volume=data.volume,
            short_length=short_length,
            multiplier=multiplier,
            scale=scale,
        )

    return indicator(name, _volume_momentum)


def laguerre_rsi(name: str, fe_length: int = 13) -> Indicator:
    """拉盖尔相对强弱指数 (RSI)。

    参数:
        name: 指标名称。
        fe_length: 分形能量长度。默认为 ``13``。

    返回:
        拉盖尔相对强弱指数 (RSI) :class:`.Indicator`。
    """

    def _laguerre_rsi(data: BarData):
        return vect.laguerre_rsi(
            open=data.open,
            high=data.high,
            low=data.low,
            close=data.close,
            fe_length=fe_length,
        )

    return indicator(name, _laguerre_rsi)

# =============================================================================
# 模块总结
# =============================================================================
#
# 本模块是PyBroker技术指标系统的核心，提供以下功能：
#
# 核心类和函数：
# - Indicator: 技术指标类，封装指标计算逻辑
# - indicator(): 创建并注册指标的工厂函数
# - IndicatorsMixin: 提供指标计算、缓存和并行处理的混入类
# - IndicatorSet: 批量管理和计算多个指标的集合类
#
# 内置指标分类：
#
# 1. 价格指标:
#    - highest/lowest: 滚动最高/最低价
#    - returns: 收益率
#    - close_minus_ma: 收盘价减均线
#
# 2. 趋势指标:
#    - linear_trend: 线性趋势强度
#    - quadratic_trend: 二次趋势强度
#    - cubic_trend: 三次趋势强度
#    - adx: 平均动向指数
#    - aroon_up/down/diff: 阿隆指标
#
# 3. 动量指标:
#    - macd: 移动平均收敛散度
#    - stochastic: 随机指标
#    - stochastic_rsi: 随机RSI
#    - detrended_rsi: 去趋势RSI
#    - laguerre_rsi: 拉盖尔RSI
#    - price_intensity: 价格强度
#    - price_change_oscillator: 价格变动振荡器
#
# 4. 成交量指标:
#    - money_flow: 蔡金资金流
#    - intraday_intensity: 日内强度
#    - normalized_on_balance_volume: 归一化OBV
#    - delta_on_balance_volume: OBV变化量
#    - normalized_positive_volume_index: 归一化正成交量指数
#    - normalized_negative_volume_index: 归一化负成交量指数
#    - volume_momentum: 成交量动量
#    - volume_weighted_ma_ratio: 成交量加权均线比率
#    - price_volume_fit: 价量拟合
#    - reactivity: 反应性
#
# 5. 偏差指标:
#    - linear_deviation: 线性趋势偏差
#    - quadratic_deviation: 二次趋势偏差
#    - cubic_deviation: 三次趋势偏差
#
# 性能优化：
# - 支持多进程并行计算，充分利用多核CPU
# - 提供指标缓存机制，避免重复计算
# - 使用numpy向量化操作，提高计算效率
#
