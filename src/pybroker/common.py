"""Contains common classes and utilities."""  # 包含通用类和工具函数

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import numpy as np
import pandas as pd
import os
import re
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from joblib import Parallel
from numpy.typing import NDArray
from typing import (
    Any,
    Callable,
    Final,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Union,
)

_tf_pattern: Final = re.compile(r"(\d+)([A-Za-z]+)")  # 时间框架正则表达式模式
_tf_abbr: Final = {  # 时间单位缩写映射
    "s": "sec",
    "m": "min",
    "h": "hour",
    "d": "day",
    "w": "week",
}
_CENTS: Final = Decimal(".01")  # 一美分的小数表示


class IndicatorSymbol(NamedTuple):
    """指标/股票代码标识符
    
    属性:
        ind_name: 指标名称
        symbol: 股票代码
    """

    ind_name: str  # 指标名称
    symbol: str  # 股票代码


class ModelSymbol(NamedTuple):
    """模型/股票代码标识符
    
    属性:
        model_name: 模型名称
        symbol: 股票代码
    """

    model_name: str  # 模型名称
    symbol: str  # 股票代码


class TrainedModel(NamedTuple):
    """训练好的模型/符号标识符
    
    属性:
        name: 训练模型名称
        instance: 训练模型实例
        predict_fn: 覆盖模型默认predict函数的可调用对象
        input_cols: 用作模型预测输入的列名称
    """

    name: str  # 模型名称
    instance: Any  # 模型实例
    predict_fn: Optional[Callable[[Any, pd.DataFrame], NDArray]]  # 预测函数
    input_cols: Optional[tuple[str]]  # 输入列


class DataCol(Enum):
    """默认数据列名称"""

    DATE = "date"  # 日期
    SYMBOL = "symbol"  # 股票代码
    OPEN = "open"  # 开盘价
    HIGH = "high"  # 最高价
    LOW = "low"  # 最低价
    CLOSE = "close"  # 收盘价
    VOLUME = "volume"  # 成交量
    VWAP = "vwap"  # 成交量加权平均价


class Day(Enum):
    """星期枚举"""

    MON = 0  # 星期一
    TUES = 1  # 星期二
    WEDS = 2  # 星期三
    THURS = 3  # 星期四
    FRI = 4  # 星期五
    SAT = 5  # 星期六
    SUN = 6  # 星期日


class PriceType(Enum):
    """用于在ExecContext中指定成交价格的价格类型枚举
    
    属性:
        OPEN: 当前K线的开盘价
        LOW: 当前K线的最低价
        HIGH: 当前K线的最高价
        CLOSE: 当前K线的收盘价
        MIDDLE: 当前K线最低价和最高价的中点
        AVERAGE: 当前K线开盘价、最低价、最高价和收盘价的平均值
    """

    OPEN = "open"  # 开盘价
    LOW = "low"  # 最低价
    HIGH = "high"  # 最高价
    CLOSE = "close"  # 收盘价
    MIDDLE = "middle"  # 中间价
    AVERAGE = "average"  # 平均价


class StopType(Enum):
    """止损止盈类型
    
    属性:
        BAR: 在n个K线后触发的止损
        LOSS: 止损
        PROFIT: 止盈
        TRAILING: 追踪止损
    """

    BAR = "bar"  # 基于K线数的止损
    LOSS = "loss"  # 止损
    PROFIT = "profit"  # 止盈
    TRAILING = "trailing"  # 追踪止损


class FeeMode(Enum):
    """回测中使用的佣金费用模式
    
    属性:
        ORDER_PERCENT: 费用是订单金额的百分比，订单金额为成交价*股数
        PER_ORDER: 费用是每笔订单的固定金额
        PER_SHARE: 费用是订单中每股的固定金额
    """

    ORDER_PERCENT = "order_percent"  # 订单百分比
    PER_ORDER = "per_order"  # 每笔订单
    PER_SHARE = "per_share"  # 每股


class FeeInfo(NamedTuple):
    """包含自定义费用计算的信息
    
    属性:
        symbol: 交易代码
        shares: 订单股数
        fill_price: 订单成交价
        order_type: 订单类型，"buy"或"sell"
    """

    symbol: str  # 股票代码
    shares: Decimal  # 股数
    fill_price: Decimal  # 成交价
    order_type: Literal["buy", "sell"]  # 订单类型


class PositionMode(Enum):
    """回测的仓位模式
    
    属性:
        DEFAULT: 多头和空头仓位
        LONG_ONLY: 仅多头仓位
        SHORT_ONLY: 仅空头仓位
    """

    DEFAULT = "default"  # 默认
    LONG_ONLY = "long_only"  # 仅多头
    SHORT_ONLY = "short_only"  # 仅空头


class BarData:
    r"""包含一系列K线的数据。每个字段是一个包含序列中K线值的numpy.ndarray。
    这些值按时间顺序升序排列。
    
    参数:
        date: 每个K线的时间戳
        open: 开盘价
        high: 最高价
        low: 最低价
        close: 收盘价
        volume: 成交量
        vwap: 成交量加权平均价
        **kwargs: 额外的K线数据字段
    """

    def __init__(
        self,
        date: NDArray[np.datetime64],  # 日期时间数组
        open: NDArray[np.float64],  # 开盘价数组
        high: NDArray[np.float64],  # 最高价数组
        low: NDArray[np.float64],  # 最低价数组
        close: NDArray[np.float64],  # 收盘价数组
        volume: Optional[NDArray[np.float64]],  # 成交量数组
        vwap: Optional[NDArray[np.float64]],  # 成交量加权平均价数组
        **kwargs,
    ):
        self.date = date
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.vwap = vwap
        self.__dict__.update(kwargs)

    def __getattr__(self, attr):
        """获取属性，如果属性不存在则返回None"""
        return None


def to_datetime(
    date: Union[str, datetime, np.datetime64, pd.Timestamp],
) -> datetime:
    """将多种日期格式转换为datetime对象
    
    参数:
        date: 要转换的日期
        
    返回:
        转换后的datetime对象
    """
    if isinstance(date, (np.datetime64, pd.Timestamp)):
        return pd.Timestamp(date).to_pydatetime()
    if isinstance(date, str):
        return pd.Timestamp(date).to_pydatetime()
    return date


def to_decimal(value: Union[int, float, Decimal]) -> Decimal:
    """将数值转换为Decimal类型
    
    参数:
        value: 要转换的数值
        
    返回:
        转换后的Decimal对象
    """
    return value if isinstance(value, Decimal) else Decimal(str(value))


def parse_timeframe(timeframe: str) -> list[tuple[int, str]]:
    """解析时间框架字符串
    
    参数:
        timeframe: 时间框架字符串，如"1d"、"3h"等
        
    返回:
        解析后的时间框架列表，每项为(数量,单位)的元组
    """
    if not timeframe:
        return []
    parts = []
    for match in _tf_pattern.finditer(timeframe):
        unit = match.group(2).lower()
        if unit in _tf_abbr:
            unit = _tf_abbr[unit]
        if unit.endswith("s"):
            unit = unit[:-1]
        parts.append((int(match.group(1)), unit))
    if not parts:
        raise ValueError(f"Invalid timeframe: {timeframe}")
    return parts


def to_seconds(timeframe: Optional[str]) -> int:
    """将时间框架字符串转换为秒数
    
    参数:
        timeframe: 时间框架字符串
        
    返回:
        对应的秒数
    """
    if not timeframe:
        return 0
    parts = parse_timeframe(timeframe)
    seconds = 0
    for amount, unit in parts:
        if unit == "sec":
            seconds += amount
        elif unit == "min":
            seconds += amount * 60
        elif unit == "hour":
            seconds += amount * 60 * 60
        elif unit == "day":
            seconds += amount * 60 * 60 * 24
        elif unit == "week":
            seconds += amount * 60 * 60 * 24 * 7
        else:
            raise ValueError(f"Unsupported time unit: {unit}")
    return seconds


def quantize(df: pd.DataFrame, col: str, round: bool) -> pd.Series:
    """量化DataFrame中的列值到分
    
    参数:
        df: 包含要量化列的DataFrame
        col: 要量化的列名
        round: 是否四舍五入
        
    返回:
        量化后的Series
    """
    if round:
        return df[col].apply(
            lambda x: Decimal(str(x)).quantize(_CENTS, rounding=ROUND_HALF_UP)
        )
    return df[col].apply(lambda x: Decimal(str(x)).quantize(_CENTS))


def verify_data_source_columns(df: pd.DataFrame):
    """验证数据源DataFrame是否包含所需的列
    
    参数:
        df: 要验证的DataFrame
    
    抛出:
        ValueError: 如果缺少必需的列
    """
    for col in [
        DataCol.DATE.value,
        DataCol.SYMBOL.value,
        DataCol.OPEN.value,
        DataCol.HIGH.value,
        DataCol.LOW.value,
        DataCol.CLOSE.value,
    ]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")


def verify_date_range(start_date: datetime, end_date: datetime):
    """验证日期范围的有效性
    
    参数:
        start_date: 开始日期
        end_date: 结束日期
        
    抛出:
        ValueError: 如果开始日期晚于结束日期
    """
    if start_date > end_date:
        raise ValueError(
            f"start_date ({start_date}) must be <= end_date ({end_date})"
        )


def default_parallel() -> Parallel:
    """获取默认的并行处理器
    
    返回:
        配置为使用所有可用处理器的Parallel对象
    """
    n_jobs = int(os.environ.get("PYBROKER_NUM_JOBS", -1))
    return Parallel(n_jobs=n_jobs)


def get_unique_sorted_dates(col: pd.Series) -> Sequence[np.datetime64]:
    """获取唯一排序的日期序列
    
    参数:
        col: 包含日期的Series
        
    返回:
        唯一且排序的日期numpy数组
    """
    dates = pd.unique(col)
    dates.sort()
    return dates

# 此模块提供了PyBroker库的基础数据结构和工具函数。
# 包含了交易系统所需的各种枚举类型，如价格类型、费用模式和仓位模式等。
# 还包括一些实用工具函数，用于处理日期时间、数据验证和并行处理等。
# BarData类提供了K线数据的标准表示方式，便于在回测系统中处理行情数据。
