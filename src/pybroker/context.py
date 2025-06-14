"""Contains context related classes. A context provides data during the
execution of a :class:`pybroker.strategy.Strategy`."""  # 包含上下文相关类。上下文在执行Strategy时提供数据。

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import numpy as np
import pandas as pd
from pybroker.common import (
    BarData,
    DataCol,
    ModelSymbol,
    PriceType,
    StopType,
    to_datetime,
    to_decimal,
)
from pybroker.config import StrategyConfig
from pybroker.model import TrainedModel
from pybroker.portfolio import Entry, Order, Portfolio, Position, Stop, Trade
from pybroker.scope import (
    ColumnScope,
    IndicatorScope,
    ModelInputScope,
    PendingOrder,
    PendingOrderScope,
    PredictionScope,
    StaticScope,
)
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from numpy.typing import NDArray
from typing import (
    Any,
    Callable,
    Iterator,
    Literal,
    Mapping,
    MutableMapping,
    NamedTuple,
    Optional,
    Union,
)


class BaseContext:
    """基础上下文类

    属性:
        config: 策略配置StrategyConfig对象
    """

    def __init__(
        self,
        config: StrategyConfig,
        portfolio: Portfolio,
        col_scope: ColumnScope,
        ind_scope: IndicatorScope,
        input_scope: ModelInputScope,
        pred_scope: PredictionScope,
        pending_order_scope: PendingOrderScope,
        models: Mapping[ModelSymbol, TrainedModel],
        sym_end_index: Mapping[str, int],
    ):
        self.config = config  # 策略配置
        self._portfolio = portfolio  # 投资组合
        self._col_scope = col_scope  # 列作用域
        self._ind_scope = ind_scope  # 指标作用域
        self._input_scope = input_scope  # 模型输入作用域
        self._pred_scope = pred_scope  # 预测作用域
        self._models = models  # 训练模型
        self._sym_end_index = sym_end_index  # 符号结束索引
        self._pending_order_scope = pending_order_scope  # 待处理订单作用域

    @property
    def total_equity(self) -> Decimal:
        """当前在Portfolio中持有的总权益"""
        return self._portfolio.equity

    @property
    def cash(self) -> Decimal:
        """当前在Portfolio中持有的总现金"""
        return self._portfolio.cash

    @property
    def total_margin(self) -> Decimal:
        """当前在Portfolio中持有的总保证金"""
        return self._portfolio.margin

    @property
    def total_market_value(self) -> Decimal:
        """当前在Portfolio中持有的总市场价值。市场价值定义为
        持有现金和多头头寸的权益与所有未平仓空头头寸的未实现损益之和。
        """
        return self._portfolio.market_value

    @property
    def win_rate(self) -> Decimal:
        """交易的运行胜率"""
        return self._portfolio.win_rate

    @property
    def loss_rate(self) -> Decimal:
        """交易的运行亏损率"""
        return self._portfolio.loss_rate

    def orders(self) -> Iterator[Order]:
        """已下单并已成交的所有订单的迭代器"""
        for order in self._portfolio.orders:
            yield order

    def pending_orders(
        self, symbol: Optional[str] = None
    ) -> Iterator[PendingOrder]:
        """待处理订单的迭代器，可选按符号筛选"""
        for order in self._pending_order_scope.orders(symbol):
            yield order

    def trades(self) -> Iterator[Trade]:
        """已完成的所有交易的迭代器"""
        for trade in self._portfolio.trades:
            yield trade

    def pos(
        self,
        symbol: str,
        pos_type: Literal["long", "short"],
    ) -> Optional[Position]:
        """检索某个符号的当前多头或空头头寸
        
        参数:
            symbol: 要返回头寸的股票代码
            pos_type: 指定是返回"long"(多头)还是"short"(空头)头寸
            
        返回:
            如果头寸存在，则返回Position对象，否则返回None
        """
        self._verify_pos_type(pos_type)
        if pos_type == "long" and symbol in self._portfolio.long_positions:
            return self._portfolio.long_positions[symbol]
        elif pos_type == "short" and symbol in self._portfolio.short_positions:
            return self._portfolio.short_positions[symbol]
        return None

    def positions(
        self,
        symbol: Optional[str] = None,
        pos_type: Optional[Literal["long", "short"]] = None,
    ) -> Iterator[Position]:
        """检索所有当前头寸
        
        参数:
            symbol: 用于筛选头寸的股票代码。如果为None，则返回所有符号的头寸。默认为None
            pos_type: 要返回的头寸类型。如果为None，则返回"long"和"short"头寸
            
        返回:
            当前持有的Position对象的迭代器
        """
        if pos_type is not None:
            self._verify_pos_type(pos_type)
        if symbol is None:
            if pos_type != "short":
                for pos in self._portfolio.long_positions.values():
                    yield pos
            if pos_type != "long":
                for pos in self._portfolio.short_positions.values():
                    yield pos
        else:
            if (
                pos_type != "short"
                and symbol in self._portfolio.long_positions
            ):
                yield self._portfolio.long_positions[symbol]
            if (
                pos_type != "long"
                and symbol in self._portfolio.short_positions
            ):
                yield self._portfolio.short_positions[symbol]

    def long_positions(
        self, symbol: Optional[str] = None
    ) -> Iterator[Position]:
        """检索所有当前多头头寸
        
        参数:
            symbol: 用于筛选头寸的股票代码。如果为None，则返回所有符号的多头头寸
            
        返回:
            当前持有的多头Position对象的迭代器
        """
        return self.positions(symbol, "long")

    def short_positions(
        self, symbol: Optional[str] = None
    ) -> Iterator[Position]:
        """检索所有当前空头头寸
        
        参数:
            symbol: 用于筛选头寸的股票代码。如果为None，则返回所有符号的空头头寸
            
        返回:
            当前持有的空头Position对象的迭代器
        """
        return self.positions(symbol, "short")

    def _verify_pos_type(self, pos_type: str):
        """验证头寸类型是否有效"""
        if pos_type != "long" and pos_type != "short":
            raise ValueError(f"Invalid position type: {pos_type}")

    def calc_target_shares(
        self, target_size: float, price: float, cash: Optional[float] = None
    ) -> Union[Decimal, int]:
        """计算目标股票数量
        
        参数:
            target_size: 相对于当前可用现金的目标头寸规模(0到1.0之间)
            price: 用于计算股份的价格
            cash: 可选的现金金额。如果未提供，则使用当前投资组合现金
            
        返回:
            计算出的股票数量，舍入到整数(如果配置中启用了整数股)
        """
        if target_size <= 0:
            return Decimal(0)
        if cash is None:
            cash = float(self._portfolio.cash)
        elif cash <= 0:
            return Decimal(0)
        shares = cash * target_size / price
        if shares <= 0:
            return Decimal(0)
        shares = to_decimal(shares)
        return shares if not self.config.round_shares else int(shares)

    def model(self, name: str, symbol: str) -> Any:
        """获取特定符号的训练模型
        
        参数:
            name: 模型名称
            symbol: 股票代码
            
        返回:
            请求的模型对象
        """
        model_sym = ModelSymbol(model_name=name, symbol=symbol)
        if model_sym not in self._models:
            raise ValueError(
                f"Model not found for name={name} and symbol={symbol}."
            )
        return self._models[model_sym].model

    def indicator(self, name: str, symbol: str) -> NDArray[np.float64]:
        """获取特定符号的指标数据
        
        参数:
            name: 指标名称
            symbol: 股票代码
            
        返回:
            指标数据的NumPy数组
        """
        if name not in self._ind_scope.indicators(symbol):
            raise ValueError(
                f"Indicator not found for name={name} and symbol={symbol}."
            )
        return self._ind_scope.get_indicator(name, symbol)

    def input(self, model_name: str, symbol: str) -> pd.DataFrame:
        """获取特定符号的模型输入数据
        
        参数:
            model_name: 模型名称
            symbol: 股票代码
            
        返回:
            模型输入数据的DataFrame
        """
        if not self._input_scope.has_input(model_name, symbol):
            raise ValueError(
                f"Input not found for model={model_name} and symbol={symbol}."
            )
        return self._input_scope.get_input(model_name, symbol)

    def preds(self, model_name: str, symbol: str) -> NDArray:
        """获取特定符号的模型预测数据
        
        参数:
            model_name: 模型名称
            symbol: 股票代码
            
        返回:
            模型预测的NumPy数组
        """
        if not self._pred_scope.has_preds(model_name, symbol):
            raise ValueError(
                f"Predictions not found for model={model_name} and "
                f"symbol={symbol}."
            )
        return self._pred_scope.get_preds(model_name, symbol)


@dataclass
class ExecResult:
    r"""持有在Strategy执行期间设置的数据
    
    属性:
        symbol: 用于执行的股票代码
        date: 用于执行的K线时间戳
        buy_fill_price: 用于买入(多头)订单的成交价格
        sell_fill_price: 用于卖出(空头)订单的成交价格
        score: 用于对多头和空头信号进行排名的分数。订单会为具有最高分数的股票下单
        hold_bars: 持有多头或空头头寸的K线数，之后头寸会自动清算
        buy_shares: 要买入的股票数量
        buy_limit_price: 用于买入(多头)订单的限价
        sell_shares: 要卖出的股票数量
        sell_limit_price: 用于卖出(空头)订单的限价
        long_stops: 多头头寸的止损设置
        short_stops: 空头头寸的止损设置
        cover: 是否使用buy_shares来平仓空头头寸。如果为True，结果的买入订单将在卖出订单之前下单
        pending_order_id: 创建的待处理订单ID
    """

    symbol: str  # 股票代码
    date: np.datetime64  # 日期时间
    buy_fill_price: Union[
        int,
        float,
        np.floating,
        Decimal,
        PriceType,
        Callable[[str, BarData], Union[int, float, Decimal]],
    ]  # 买入成交价格
    sell_fill_price: Union[
        int,
        float,
        np.floating,
        Decimal,
        PriceType,
        Callable[[str, BarData], Union[int, float, Decimal]],
    ]  # 卖出成交价格
    score: Optional[float]  # 得分
    hold_bars: Optional[int]  # 持有K线数
    buy_shares: Optional[Decimal]  # 买入股数
    buy_limit_price: Optional[Decimal]  # 买入限价
    sell_shares: Optional[Decimal]  # 卖出股数
    sell_limit_price: Optional[Decimal]  # 卖出限价
    long_stops: Optional[frozenset[Stop]]  # 多头止损
    short_stops: Optional[frozenset[Stop]]  # 空头止损
    cover: bool = field(default=False)  # 是否平仓空头
    pending_order_id: Optional[int] = field(default=None)  # 待处理订单ID


class ExecSignal(NamedTuple):
    """持有买入/卖出信号的数据
    
    属性:
        id: 唯一标识符
        symbol: 股票代码
        shares: Strategy执行设置的股票数量
        score: Strategy执行设置的分数
        bar_data: 符号的K线数据
        type: 信号类型，"buy"或"sell"
    """

    id: int  # 信号ID
    symbol: str  # 股票代码
    shares: Union[int, float, Decimal]  # 股票数量
    score: Optional[float]  # 分数
    bar_data: BarData  # K线数据
    type: Literal["buy", "sell"]  # 信号类型


class PosSizeContext(BaseContext):
    """用于确定头寸大小的上下文类"""

    def __init__(
        self,
        config: StrategyConfig,
        portfolio: Portfolio,
        col_scope: ColumnScope,
        ind_scope: IndicatorScope,
        input_scope: ModelInputScope,
        pred_scope: PredictionScope,
        pending_order_scope: PendingOrderScope,
        models: Mapping[ModelSymbol, TrainedModel],
        sessions: Mapping[str, Mapping],
        sym_end_index: Mapping[str, int],
    ):
        """初始化PosSizeContext实例"""
        super().__init__(
            config,
            portfolio,
            col_scope,
            ind_scope,
            input_scope,
            pred_scope,
            pending_order_scope,
            models,
            sym_end_index,
        )
        self._signals: list[ExecSignal] = []
        self._buy_signals: list[ExecSignal] = []
        self._sell_signals: list[ExecSignal] = []
        self._sessions = sessions

    def signals(
        self, signal_type: Optional[Literal["buy", "sell"]] = None
    ) -> Iterator[ExecSignal]:
        """获取当前排序后的信号列表
        
        参数:
            signal_type: 可选的信号类型筛选器("buy"或"sell")
            
        返回:
            ExecSignal对象的迭代器
        """
        if signal_type is None:
            for signal in sorted(
                self._signals, key=lambda signal: -float("inf") if signal.score is None else -signal.score
            ):
                yield signal
        elif signal_type == "buy":
            for signal in sorted(
                self._buy_signals,
                key=lambda signal: -float("inf") if signal.score is None else -signal.score,
            ):
                yield signal
        elif signal_type == "sell":
            for signal in sorted(
                self._sell_signals,
                key=lambda signal: -float("inf") if signal.score is None else -signal.score,
            ):
                yield signal
        else:
            raise ValueError(f"Invalid signal_type: {signal_type}")

    def set_shares(
        self, signal: ExecSignal, shares: Union[int, float, Decimal]
    ):
        """设置信号的股票数量"""
        signal.shares = to_decimal(shares)  # type: ignore


def set_pos_size_ctx_data(
    ctx: PosSizeContext,
    buy_results: Optional[list[ExecResult]],
    sell_results: Optional[list[ExecResult]],
):
    """设置头寸大小上下文数据"""
    pass  # 该函数在源代码中未实现


class ExecContext(BaseContext):
    r"""在Strategy执行期间包含上下文数据。包括当前K线、投资组合头寸和其他相关上下文的数据。
    该类还用于设置买卖信号以下单。
    
    此类中包含的数据是针对已经完成的最新K线。下单将在由StrategyConfig的buy_delay和sell_delay
    指定的未来K线上执行。
    
    属性:
        symbol: 执行的当前股票代码
        buy_fill_price: 用于买入(多头)订单的成交价格
        buy_shares: 要买入的股票数量
        buy_limit_price: 用于买入(多头)订单的限价
        sell_fill_price: 用于卖出(空头)订单的成交价格
        sell_shares: 要卖出的股票数量
        sell_limit_price: 用于卖出(空头)订单的限价
        hold_bars: 持有多头或空头头寸的K线数，之后头寸会自动清算
        score: 用于对买卖信号进行排名的分数
        session: 在Strategy执行期间用于存储每个K线的自定义持久数据的字典
        stop_loss: 对新Entry设置止损，值以入场价格为基准点数计
        stop_loss_pct: 对新Entry设置止损，值以入场价格为基准百分比计
        stop_loss_limit: 止损使用的限价
        stop_loss_exit_price: 止损退出使用的价格类型
        stop_profit: 对新Entry设置止盈，值以入场价格为基准点数计
        stop_profit_pct: 对新Entry设置止盈，值以入场价格为基准百分比计
        stop_profit_limit: 止盈使用的限价
        stop_profit_exit_price: 止盈退出使用的价格类型
        stop_trailing: 对新Entry设置跟踪止损，值以入场价格为基准点数计
        stop_trailing_pct: 对新Entry设置跟踪止损，值以入场价格为基准百分比计
        stop_trailing_limit: 跟踪止损使用的限价
        stop_trailing_exit_price: 跟踪止损退出使用的价格类型
    """

    _stop_id: int = 0  # 止损ID计数器

    def __init__(
        self,
        symbol: str,
        config: StrategyConfig,
        portfolio: Portfolio,
        col_scope: ColumnScope,
        ind_scope: IndicatorScope,
        input_scope: ModelInputScope,
        pred_scope: PredictionScope,
        pending_order_scope: PendingOrderScope,
        models: Mapping[ModelSymbol, TrainedModel],
        sym_end_index: Mapping[str, int],
        session: MutableMapping,
    ):
        """初始化ExecContext实例"""
        super().__init__(
            config,
            portfolio,
            col_scope,
            ind_scope,
            input_scope,
            pred_scope,
            pending_order_scope,
            models,
            sym_end_index,
        )
        self.symbol = symbol  # 当前股票代码
        self.buy_fill_price = None  # 买入成交价格
        self.buy_shares = None  # 买入股数
        self.buy_limit_price = None  # 买入限价
        self.sell_fill_price = None  # 卖出成交价格
        self.sell_shares = None  # 卖出股数
        self.sell_limit_price = None  # 卖出限价
        self.hold_bars = None  # 持有K线数
        self.score = None  # 得分
        self.session = session  # 会话数据
        self.stop_loss = None  # 止损点数
        self.stop_loss_pct = None  # 止损百分比
        self.stop_loss_limit = None  # 止损限价
        self.stop_loss_exit_price = None  # 止损退出价格类型
        self.stop_profit = None  # 止盈点数
        self.stop_profit_pct = None  # 止盈百分比
        self.stop_profit_limit = None  # 止盈限价
        self.stop_trailing = None  # 跟踪止损点数
        self.stop_trailing_pct = None  # 跟踪止损百分比
        self.stop_trailing_limit = None  # 跟踪止损限价
        self.stop_trailing_exit_price = None  # 跟踪止损退出价格类型
        self._cover_fill_price = None  # 平仓成交价格
        self._cover_shares = None  # 平仓股数
        self._cover_limit_price = None  # 平仓限价
        self._date = None  # 当前日期
        self._stops = set()  # 止损集合
        self._pending_order_id = None  # 待处理订单ID

    def _verify_symbol(self):
        """验证当前符号是否有效"""
        if not self.symbol:
            raise ValueError("Symbol cannot be empty.")

    @property
    def bars(self) -> int:
        """返回可用K线数量"""
        return self._col_scope.bar_count(self.symbol)

    @property
    def dt(self) -> datetime:
        """返回当前K线的日期时间"""
        if self._date is None:
            raise ValueError("Date is not set.")
        return to_datetime(self._date)

    @property
    def date(self) -> NDArray[np.datetime64]:
        """返回所有K线的日期数组"""
        self._verify_symbol()
        return self._col_scope.get_column(self.symbol, DataCol.DATE.value)

    @property
    def open(self) -> NDArray[np.float64]:
        """返回所有K线的开盘价数组"""
        self._verify_symbol()
        return self._col_scope.get_column(self.symbol, DataCol.OPEN.value)

    @property
    def high(self) -> NDArray[np.float64]:
        """返回所有K线的最高价数组"""
        self._verify_symbol()
        return self._col_scope.get_column(self.symbol, DataCol.HIGH.value)

    @property
    def low(self) -> NDArray[np.float64]:
        """返回所有K线的最低价数组"""
        self._verify_symbol()
        return self._col_scope.get_column(self.symbol, DataCol.LOW.value)

    @property
    def close(self) -> NDArray[np.float64]:
        """返回所有K线的收盘价数组"""
        self._verify_symbol()
        return self._col_scope.get_column(self.symbol, DataCol.CLOSE.value)

    @property
    def volume(self) -> Optional[NDArray[np.float64]]:
        """返回所有K线的成交量数组，如果可用"""
        self._verify_symbol()
        if not self._col_scope.has_column(self.symbol, DataCol.VOLUME.value):
            return None
        return self._col_scope.get_column(self.symbol, DataCol.VOLUME.value)

    @property
    def vwap(self) -> Optional[NDArray[np.float64]]:
        """返回所有K线的成交量加权平均价数组，如果可用"""
        self._verify_symbol()
        if not self._col_scope.has_column(self.symbol, DataCol.VWAP.value):
            return None
        return self._col_scope.get_column(self.symbol, DataCol.VWAP.value)

    def foreign(
        self, symbol: str, col: Optional[str] = None
    ) -> Union[BarData, Optional[NDArray]]:
        """获取其他符号的K线数据
        
        参数:
            symbol: 要获取数据的股票代码
            col: 可选的要获取的特定列(例如"close"、"open"等)
            
        返回:
            如果指定了col，则返回该列的数组；否则返回完整的BarData对象
        """
        if not self._col_scope.has_symbol(symbol):
            raise ValueError(f"No data found for symbol: {symbol}")
        if col is not None:
            if not self._col_scope.has_column(symbol, col):
                return None
            return self._col_scope.get_column(symbol, col)
        return BarData(
            date=self._col_scope.get_column(symbol, DataCol.DATE.value),
            open=self._col_scope.get_column(symbol, DataCol.OPEN.value),
            high=self._col_scope.get_column(symbol, DataCol.HIGH.value),
            low=self._col_scope.get_column(symbol, DataCol.LOW.value),
            close=self._col_scope.get_column(symbol, DataCol.CLOSE.value),
            volume=(
                self._col_scope.get_column(symbol, DataCol.VOLUME.value)
                if self._col_scope.has_column(symbol, DataCol.VOLUME.value)
                else None
            ),
            vwap=(
                self._col_scope.get_column(symbol, DataCol.VWAP.value)
                if self._col_scope.has_column(symbol, DataCol.VWAP.value)
                else None
            ),
        )

    def model(self, name: str, symbol: Optional[str] = None) -> Any:
        """获取训练模型
        
        参数:
            name: 模型名称
            symbol: 可选的股票代码。如果未提供，则使用当前上下文的symbol
            
        返回:
            请求的模型对象
        """
        symbol = self._get_symbol(symbol)
        model_sym = ModelSymbol(model_name=name, symbol=symbol)
        if model_sym not in self._models:
            raise ValueError(
                f"Model not found for name={name} and symbol={symbol}."
            )
        return self._models[model_sym].model

    def indicator(
        self, name: str, symbol: Optional[str] = None
    ) -> NDArray[np.float64]:
        """获取指标数据
        
        参数:
            name: 指标名称
            symbol: 可选的股票代码。如果未提供，则使用当前上下文的symbol
            
        返回:
            指标数据的NumPy数组
        """
        symbol = self._get_symbol(symbol)
        if name not in self._ind_scope.indicators(symbol):
            raise ValueError(
                f"Indicator not found for name={name} and symbol={symbol}."
            )
        return self._ind_scope.get_indicator(name, symbol)

    def input(
        self, model_name: str, symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """获取模型输入数据
        
        参数:
            model_name: 模型名称
            symbol: 可选的股票代码。如果未提供，则使用当前上下文的symbol
            
        返回:
            模型输入数据的DataFrame
        """
        symbol = self._get_symbol(symbol)
        if not self._input_scope.has_input(model_name, symbol):
            raise ValueError(
                f"Input not found for model={model_name} and symbol={symbol}."
            )
        return self._input_scope.get_input(model_name, symbol)

    def preds(self, model_name: str, symbol: Optional[str] = None) -> NDArray:
        """获取模型预测数据
        
        参数:
            model_name: 模型名称
            symbol: 可选的股票代码。如果未提供，则使用当前上下文的symbol
            
        返回:
            模型预测的NumPy数组
        """
        symbol = self._get_symbol(symbol)
        if not self._pred_scope.has_preds(model_name, symbol):
            raise ValueError(
                f"Predictions not found for model={model_name} and symbol={symbol}."
            )
        return self._pred_scope.get_preds(model_name, symbol)

    def long_pos(
        self,
        symbol: Optional[str] = None,
    ) -> Optional[Position]:
        """获取多头头寸
        
        参数:
            symbol: 可选的股票代码。如果未提供，则使用当前上下文的symbol
            
        返回:
            如果存在，则返回多头Position对象，否则返回None
        """
        symbol = self._get_symbol(symbol)
        if symbol in self._portfolio.long_positions:
            return self._portfolio.long_positions[symbol]
        return None

    def short_pos(
        self,
        symbol: Optional[str] = None,
    ) -> Optional[Position]:
        """获取空头头寸
        
        参数:
            symbol: 可选的股票代码。如果未提供，则使用当前上下文的symbol
            
        返回:
            如果存在，则返回空头Position对象，否则返回None
        """
        symbol = self._get_symbol(symbol)
        if symbol in self._portfolio.short_positions:
            return self._portfolio.short_positions[symbol]
        return None

    def _get_symbol(self, symbol: Optional[str] = None) -> str:
        """获取要使用的股票代码
        
        如果提供了symbol参数，则使用它；否则使用当前上下文的symbol
        """
        if symbol is not None:
            return symbol
        self._verify_symbol()
        return self.symbol

    def calc_target_shares(
        self,
        target_size: float,
        price: Optional[float] = None,
        cash: Optional[float] = None,
    ) -> Union[Decimal, int]:
        r"""Calculates the number of shares given a ``target_size`` allocation
        and share ``price``.

        Args:
            target_size: Proportion of cash used to calculate the number of
                shares, where the max ``target_size`` is ``1``. For example, a
                ``target_size`` of ``0.1`` would represent 10% of cash.
            price: Share price used to calculate the number of shares. If
                ``None``, the share price of the ``ExecContext``\ 's
                :attr:`.symbol` is used.
            cash: Cash used to calculate the number of number of shares. If
                ``None``, then the :class:`pybroker.portfolio.Portfolio` equity
                is used to calculate the number of shares.

        Returns:
            Number of shares given ``target_size`` and share ``price``. If
            :attr:`pybroker.config.StrategyConfig.enable_fractional_shares` is
            ``True``, then a Decimal is returned.
        """
        price = self.close[-1] if price is None else price
        return super().calc_target_shares(target_size, price, cash)

    def cancel_pending_order(self, order_id: int) -> bool:
        """Cancels a :class:`pybroker.scope.PendingOrder` with ``order_id``."""
        return self._pending_order_scope.remove(order_id)

    def cancel_all_pending_orders(self, symbol: Optional[str] = None):
        r"""Cancels all :class:`pybroker.scope.PendingOrder`\ s for ``symbol``.
        When ``symbol`` is ``None``, all pending orders are canceled.
        """
        self._pending_order_scope.remove_all(symbol)

    def cancel_stop(self, stop_id: int) -> bool:
        """Cancels a :class:`pybroker.portfolio.Stop` with ``stop_id``."""
        return self._portfolio.remove_stop(stop_id)

    def cancel_stops(
        self,
        val: Union[str, Position, Entry],
        stop_type: Optional[StopType] = None,
    ):
        r"""Cancels :class:`pybroker.portfolio.Stop`\ s.

        Args:
            val: Ticker symbol, :class:`pybroker.portfolio.Position`, or
                :class:`pybroker.portfolio.Entry` for which to cancel stops.
            stop_type: :class:`pybroker.common.StopType`.
        """
        self._portfolio.remove_stops(val, stop_type)

    def to_result(self) -> Optional[ExecResult]:
        """Creates an :class:`.ExecResult` from the data set on
        :class:`.ExecContext`.
        """
        if self._date is None:
            raise ValueError("Date is not set.")
        if self.symbol is None:
            raise ValueError("Symbol is not set.")
        if self.buy_shares is None:
            if self.buy_limit_price is not None:
                raise ValueError(
                    "buy_shares must be set when buy_limit_price is set."
                )
            if self.buy_fill_price is not None and self.hold_bars is None:
                raise ValueError(
                    "buy_shares or hold_bars must be set when "
                    "buy_fill_price is set."
                )
        if self.sell_shares is None:
            if self.sell_limit_price is not None:
                raise ValueError(
                    "sell_shares must be set when sell_limit_price is set."
                )
            if self.sell_fill_price is not None and self.hold_bars is None:
                raise ValueError(
                    "sell_shares or hold_bars must be set when "
                    "sell_fill_price is set."
                )

        if self.buy_shares is None and self.sell_shares is None:
            if (
                self.stop_loss is not None
                or self.stop_loss_pct is not None
                or self.stop_loss_limit is not None
                or self.stop_profit is not None
                or self.stop_profit_pct is not None
                or self.stop_profit_limit is not None
                or self.stop_trailing is not None
                or self.stop_trailing_pct is not None
                or self.stop_trailing_limit is not None
            ):
                raise ValueError(
                    "Either buy_shares or sell_shares must be set when a stop "
                    "is set."
                )
            if self.hold_bars is not None:
                raise ValueError(
                    "Either buy_shares or sell_shares must be set when "
                    "hold_bars is set."
                )
        if self.buy_shares is not None and self.sell_shares is not None:
            raise ValueError(
                "For each symbol, only one of buy_shares or sell_shares can be"
                " set per bar."
            )
        if not self.buy_shares and not self.sell_shares:
            return None
        buy_fill_price = (
            self.buy_fill_price
            if self.buy_fill_price is not None
            else PriceType.MIDDLE
        )
        sell_fill_price = (
            self.sell_fill_price
            if self.sell_fill_price is not None
            else PriceType.MIDDLE
        )
        buy_shares = (
            to_decimal(self.buy_shares)
            if self.buy_shares is not None
            else None
        )
        buy_limit_price = (
            to_decimal(self.buy_limit_price)
            if self.buy_limit_price is not None
            else None
        )
        sell_limit_price = (
            to_decimal(self.sell_limit_price)
            if self.sell_limit_price is not None
            else None
        )
        sell_shares = (
            to_decimal(self.sell_shares)
            if self.sell_shares is not None
            else None
        )
        long_stops, short_stops = self._get_stops()
        return ExecResult(
            symbol=self.symbol,
            date=self._date,
            buy_fill_price=buy_fill_price,
            sell_fill_price=sell_fill_price,
            score=self.score,
            hold_bars=self.hold_bars,
            buy_shares=buy_shares,
            buy_limit_price=buy_limit_price,
            sell_shares=sell_shares,
            sell_limit_price=sell_limit_price,
            long_stops=long_stops,
            short_stops=short_stops,
            cover=self._cover_fill_price is not None,
            pending_order_id=self._pending_order_id,
        )

    def __getattr__(self, attr):
        if attr in self._scope.custom_data_cols:
            if self.symbol is None:
                raise ValueError("Symbol is not set.")
            return self._col_scope.fetch(
                self.symbol, attr, self._sym_end_index[self.symbol]
            )
        raise AttributeError(f"Attribute {attr!r} not found.")


def set_exec_ctx_data(ctx: ExecContext, date: np.datetime64):
    """Sets data on an :class:`.ExecContext` instance.

    Args:
        ctx: :class:`.ExecContext`.
        date: Current bar's date.
    """
    ctx._date = date
    ctx._stops.clear()
    ctx._cover_fill_price = None
    ctx._cover_shares = None
    ctx._cover_limit_price = None
    ctx.buy_fill_price = None
    ctx.buy_shares = None
    ctx.buy_limit_price = None
    ctx.sell_fill_price = None
    ctx.sell_shares = None
    ctx.sell_limit_price = None
    ctx.hold_bars = None
    ctx.score = None
    ctx.stop_loss = None
    ctx.stop_loss_pct = None
    ctx.stop_loss_limit = None
    ctx.stop_profit = None
    ctx.stop_profit_pct = None
    ctx.stop_profit_limit = None
    ctx.stop_trailing = None
    ctx.stop_trailing_pct = None
    ctx.stop_trailing_limit = None
    ctx.stop_trailing_exit_price = None
