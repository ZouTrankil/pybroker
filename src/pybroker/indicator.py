"""Contains indicator related functionality."""  # 包含指标相关功能

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
    """将DataFrame转换为BarData对象"""
    df = df.reset_index()
    required_cols = (
        DataCol.DATE,
        DataCol.OPEN,
        DataCol.HIGH,
        DataCol.LOW,
        DataCol.CLOSE,
    )
    for col in required_cols:
        if col.value not in df.columns:
            raise ValueError(
                f"DataFrame is missing required column: {col.value}"
            )
    return BarData(
        **{col.value: df[col.value].to_numpy() for col in required_cols},
        **{
            col.value: (
                df[col.value].to_numpy() if col.value in df.columns else None
            )
            for col in (DataCol.VOLUME, DataCol.VWAP)
        },  # type: ignore[arg-type]
        **{
            col: df[col].to_numpy() if col in df.columns else None
            for col in StaticScope.instance().custom_data_cols
        },  # type: ignore[arg-type]
    )


class Indicator:
    """代表一个技术指标的类
    
    参数:
        name: 指标名称
        fn: 用于计算指标值序列的可调用函数
        kwargs: 传递给fn的关键字参数字典
    """

    def __init__(
        self,
        name: str,  # 指标名称
        fn: Callable[..., NDArray[np.float64]],  # 指标计算函数
        kwargs: dict[str, Any],  # 关键字参数
    ):
        self.name = name
        self._fn = functools.partial(fn, **kwargs)
        self._kwargs = kwargs

    def relative_entropy(self, data: Union[BarData, pd.DataFrame]) -> float:
        """使用data生成指标数据并计算其相对熵值
        
        相对熵是衡量数据分布不确定性的指标
        """
        return relative_entropy(self(data).values)

    def iqr(self, data: Union[BarData, pd.DataFrame]) -> float:
        """使用data生成指标数据并计算其四分位距(IQR)
        
        四分位距是衡量数据分散程度的指标，等于第三四分位减去第一四分位
        """
        return iqr(self(data).values)

    def __call__(self, data: Union[BarData, pd.DataFrame]) -> pd.Series:
        """计算指标值"""
        if isinstance(data, pd.DataFrame):
            data = _to_bar_data(data)
        values = self._fn(data)
        if isinstance(values, pd.Series):
            values = values.to_numpy()
        if len(values.shape) != 1:
            raise ValueError(
                f"Indicator {self.name} must return a one-dimensional array."
            )
        return pd.Series(values, index=data.date)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Indicator({self.name!r}, {self._kwargs})"


def indicator(
    name: str, fn: Callable[..., NDArray[np.float64]], **kwargs
) -> Indicator:
    r"""创建Indicator实例并全局注册该指标
    
    参数:
        name: 全局引用此指标的名称
        fn: 用于计算指标值的可调用函数
        \**kwargs: 传递给fn的额外参数
        
    返回:
        Indicator实例
    """
    scope = StaticScope.instance()
    indicator = Indicator(name, fn, kwargs)
    scope.set_indicator(indicator)
    return indicator


def _decorate_indicator_fn(ind_name: str):
    """装饰指标函数，用于并行计算"""
    fn = StaticScope.instance().get_indicator(ind_name).__call__

    def decorated_indicator_fn(
        symbol: str,  # 股票代码
        ind_name: str,  # 指标名称
        date: NDArray[np.datetime64],  # 日期数组
        open: NDArray[np.float64],  # 开盘价数组
        high: NDArray[np.float64],  # 最高价数组
        low: NDArray[np.float64],  # 最低价数组
        close: NDArray[np.float64],  # 收盘价数组
        volume: Optional[NDArray[np.float64]],  # 成交量数组
        vwap: Optional[NDArray[np.float64]],  # 成交量加权平均价数组
        custom_col_data: Mapping[str, Optional[NDArray]],  # 自定义列数据
    ) -> tuple[IndicatorSymbol, pd.Series]:
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
        series = fn(bar_data)
        return IndicatorSymbol(ind_name, symbol), series

    return decorated_indicator_fn


class IndicatorsMixin:
    """实现指标相关功能的混入类"""

    def compute_indicators(
        self,
        df: pd.DataFrame,  # 数据框
        indicator_syms: Iterable[IndicatorSymbol],  # 指标符号迭代器
        cache_date_fields: Optional[CacheDateFields],  # 缓存日期字段
        disable_parallel: bool,  # 是否禁用并行计算
    ) -> dict[IndicatorSymbol, pd.Series]:
        """计算所提供的IndicatorSymbol对的指标数据
        
        参数:
            df: 用于计算指标值的DataFrame
            indicator_syms: IndicatorSymbol对的可迭代对象
            cache_date_fields: 用于键缓存数据的日期字段，传None禁用缓存
            disable_parallel: 如果为True，则串行计算所有IndicatorSymbol对的指标数据；
                            如果为False，则使用多进程并行计算
        
        返回:
            将每个IndicatorSymbol对映射到计算得出的指标值Series的字典
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
        indicator_syms = sorted(indicator_syms)
        indicator_data: dict[IndicatorSymbol, pd.Series] = {}
        if cache_date_fields is None:
            return indicator_data, indicator_syms
        scope = StaticScope.instance()
        if scope.indicator_cache is None:
            return indicator_data, indicator_syms
        uncached_ind_syms = []
        for ind_sym in indicator_syms:
            cache_key = IndicatorCacheKey(
                symbol=ind_sym.symbol,
                ind_name=ind_sym.ind_name,
                **asdict(cache_date_fields),
            )
            scope.logger.debug_get_indicator_cache(cache_key)
            data = scope.indicator_cache.get(repr(cache_key))
            if data is not None:
                indicator_data[ind_sym] = data
            else:
                uncached_ind_syms.append(ind_sym)
        return indicator_data, uncached_ind_syms

    def _set_cached_indicator(
        self,
        series: pd.Series,
        ind_sym: IndicatorSymbol,
        cache_date_fields: Optional[CacheDateFields],
    ):
        if cache_date_fields is None:
            return
        scope = StaticScope.instance()
        if scope.indicator_cache is None:
            return
        cache_key = IndicatorCacheKey(
            symbol=ind_sym.symbol,
            ind_name=ind_sym.ind_name,
            **asdict(cache_date_fields),
        )
        scope.logger.debug_set_indicator_cache(cache_key)
        scope.indicator_cache.set(repr(cache_key), series)

    def _run_indicators(
        self,
        sym_data: Mapping[str, Mapping[str, Optional[NDArray]]],
        ind_syms: Collection[IndicatorSymbol],
        disable_parallel: bool,
    ) -> Iterable[tuple[IndicatorSymbol, pd.Series]]:
        fns = {}
        for ind_name, _ in ind_syms:
            if ind_name in fns:
                continue
            fns[ind_name] = _decorate_indicator_fn(ind_name)
        scope = StaticScope.instance()

        def args_fn(ind_name, sym):
            return {
                "symbol": sym,
                "ind_name": ind_name,
                "custom_col_data": {
                    col: sym_data[sym][col] for col in scope.custom_data_cols
                },
                **{col: sym_data[sym][col] for col in scope.default_data_cols},
            }

        if disable_parallel or len(ind_syms) == 1:
            scope.logger.debug_compute_indicators(is_parallel=False)
            return tuple(
                fns[ind_name](**args_fn(ind_name, sym))
                for ind_name, sym in ind_syms
            )
        else:
            scope.logger.debug_compute_indicators(is_parallel=True)

            with default_parallel() as parallel:
                return parallel(
                    delayed(fns[ind_name])(**args_fn(ind_name, sym))
                    for ind_name, sym in ind_syms
                )


class IndicatorSet(IndicatorsMixin):
    """计算多个指标的数据。"""

    def __init__(self):
        self._ind_names: set[str] = set()

    def add(self, indicators: Union[Indicator, Iterable[Indicator]], *args):
        """添加指标。"""
        if isinstance(indicators, Indicator):
            indicators = (indicators, *args)
        else:
            indicators = (*indicators, *args)
        self._ind_names.update(map(op.attrgetter("name"), indicators))

    def remove(self, indicators: Union[Indicator, Iterable[Indicator]], *args):
        """移除指标。"""
        if isinstance(indicators, Indicator):
            indicators = (indicators, *args)
        else:
            indicators = (*indicators, *args)
        self._ind_names.difference_update(
            map(op.attrgetter("name"), indicators)
        )

    def clear(self):
        """移除所有指标。"""
        self._ind_names.clear()

    def __call__(
        self, df: pd.DataFrame, disable_parallel: bool = False
    ) -> pd.DataFrame:
        """计算指标数据。

        参数:
            df: 输入数据的 :class:`pandas.DataFrame`。
            disable_parallel: 如果为 ``True``，指标数据将串行计算。
                如果为 ``False``，指标数据将使用多进程并行计算。
                默认为 ``False``。

        返回:
            包含计算出的指标数据的 :class:`pandas.DataFrame`。
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

# 该模块提供了PyBroker的技术指标功能，包括指标的创建、计算和缓存。
# 核心类Indicator表示一个技术指标，IndicatorSet用于管理多个指标。
# 库内置了丰富的技术指标函数，如MACD、RSI、随机指标和趋势指标等。
# 支持并行计算指标，提高大规模回测效率，并提供缓存机制避免重复计算。
