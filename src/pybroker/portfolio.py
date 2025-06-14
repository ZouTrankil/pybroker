"""Contains portfolio related functionality, such as portfolio metrics and
placing orders.
"""  # 包含投资组合相关功能，如投资组合指标和下单

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import itertools
import numpy as np
import pandas as pd
from pybroker.common import (
    BarData,
    DataCol,
    FeeInfo,
    FeeMode,
    PositionMode,
    PriceType,
    StopType,
    to_decimal,
)
from pybroker.scope import PriceScope, StaticScope
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import (
    Callable,
    Final,
    Iterable,
    Literal,
    NamedTuple,
    Optional,
    Union,
)

_DECIMAL_100: Final = Decimal(100)  # 用于百分比计算的常量


class Stop(NamedTuple):
    """包含有关设置在Entry上的止损信息
    
    属性:
        id: 唯一标识符
        symbol: 止损的股票代码
        stop_type: 止损类型
        pos_type: 仓位类型，"long"(多头)或"short"(空头)
        percent: 入场价格的百分比
        points: 入场价格的现金金额
        bars: 触发止损的K线数量
        fill_price: 止损将成交的价格
        limit_price: 止损使用的限价
        exit_price: 用于止损退出的价格类型，如果设置，止损将根据exit_price检查并以exit_price退出
    """

    id: int  # 唯一标识符
    symbol: str  # 股票代码
    stop_type: StopType  # 止损类型
    pos_type: Literal["long", "short"]  # 仓位类型
    percent: Optional[Decimal]  # 百分比
    points: Optional[Decimal]  # 点数
    bars: Optional[int]  # K线数
    fill_price: Optional[  # 成交价格
        Union[
            int,
            float,
            np.floating,
            Decimal,
            PriceType,
            Callable[[str, BarData], Union[int, float, Decimal]],
        ]
    ]
    limit_price: Optional[Decimal]  # 限价
    exit_price: Optional[PriceType]  # 退出价格类型


class StopRecord(NamedTuple):
    """记录每个K线关于止损的数据
    
    属性:
        date: K线的日期
        symbol: 止损的股票代码
        stop_id: 唯一标识符
        stop_type: 止损类型
        pos_type: 仓位类型，"long"(多头)或"short"(空头)
        curr_value: 止损的当前值
        curr_bars: 止损的当前K线数
        percent: 入场价格的百分比
        points: 入场价格的现金金额
        bars: 触发止损的K线数量
        fill_price: 止损将成交的价格
        limit_price: 止损使用的限价
        exit_price: 用于止损退出的价格类型
    """

    date: np.datetime64  # 日期
    symbol: str  # 股票代码
    stop_id: int  # 止损ID
    stop_type: str  # 止损类型
    pos_type: Literal["long", "short"]  # 仓位类型
    curr_value: Optional[Decimal]  # 当前值
    curr_bars: Optional[int]  # 当前K线数
    percent: Optional[Decimal]  # 百分比
    points: Optional[Decimal]  # 点数
    bars: Optional[int]  # K线数
    fill_price: Optional[Decimal]  # 成交价格
    limit_price: Optional[Decimal]  # 限价
    exit_price: Optional[PriceType]  # 退出价格类型


@dataclass
class Entry:
    """包含进入Position的入场信息
    
    属性:
        id: 唯一标识符
        date: 入场日期
        symbol: 入场的股票代码
        shares: 股数
        price: 入场价格
        type: 仓位类型，"long"(多头)或"short"(空头)
        bars: 自入场以来的当前K线数量
        stops: 设置在入场上的止损
        mae: 最大不利偏移(Maximum Adverse Excursion)
        mfe: 最大有利偏移(Maximum Favorable Excursion)
    """

    id: int  # 唯一标识符
    date: np.datetime64  # 入场日期
    symbol: str  # 股票代码
    shares: Decimal  # 股数
    price: Decimal  # 价格
    type: Literal["long", "short"]  # 仓位类型
    bars: int = field(default=0)  # K线数
    stops: list[Stop] = field(default_factory=list)  # 止损列表
    mae: Decimal = field(default_factory=Decimal)  # 最大不利偏移
    mfe: Decimal = field(default_factory=Decimal)  # 最大有利偏移


@dataclass
class _StopData:
    """内部使用的止损数据类"""
    value: Decimal  # 值
    stop: Stop  # 止损对象
    entry: Entry  # 入场对象


@dataclass
class Position:
    r"""包含symbol中的持仓信息
    
    属性:
        symbol: 持仓的股票代码
        shares: 股数
        type: 持仓类型，"long"(多头)或"short"(空头)
        close: symbol的最新收盘价
        equity: 持仓的权益
        market_value: 持仓的市场价值
        margin: 持仓中的保证金金额
        pnl: 未实现的盈亏
        entries: 按时间升序排列的入场信息队列
        bars: 自入场以来的当前K线数量
    """

    symbol: str  # 股票代码
    shares: Decimal  # 股数
    type: Literal["long", "short"]  # 仓位类型
    close: Decimal = field(default_factory=Decimal)  # 收盘价
    equity: Decimal = field(default_factory=Decimal)  # 权益
    market_value: Decimal = field(default_factory=Decimal)  # 市值
    margin: Decimal = field(default_factory=Decimal)  # 保证金
    pnl: Decimal = field(default_factory=Decimal)  # 盈亏
    entries: deque[Entry] = field(default_factory=deque)  # 入场队列
    bars: int = field(default=0)  # K线数


class Trade(NamedTuple):
    """持有关于完成交易(入场和出场)的信息
    
    属性:
        id: 唯一标识符
        type: 交易类型，"long"(多头)或"short"(空头)
        symbol: 交易的股票代码
        entry_date: 入场日期
        exit_date: 出场日期
        entry: 入场价格
        exit: 出场价格
        shares: 股数
        pnl: 盈亏
        return_pct: 以百分比计量的回报
        agg_pnl: 交易后策略的累计盈亏
        bars: 交易持有的K线数量
        pnl_per_bar: 每个K线的盈亏
        stop: 触发的止损类型，如果有的话
        mae: 最大不利偏移
        mfe: 最大有利偏移
    """

    id: int  # 唯一标识符
    type: Literal["long", "short"]  # 交易类型
    symbol: str  # 股票代码
    entry_date: np.datetime64  # 入场日期
    exit_date: np.datetime64  # 出场日期
    entry: Decimal  # 入场价格
    exit: Decimal  # 出场价格
    shares: Decimal  # 股数
    pnl: Decimal  # 盈亏
    return_pct: Decimal  # 回报率
    agg_pnl: Decimal  # 累计盈亏
    bars: int  # K线数
    pnl_per_bar: Decimal  # 每K线盈亏
    stop: Optional[Literal["bar", "loss", "profit", "trailing"]]  # 止损类型
    mae: Decimal  # 最大不利偏移
    mfe: Decimal  # 最大有利偏移


class Order(NamedTuple):
    """持有关于已成交订单的信息
    
    属性:
        id: 唯一标识符
        type: 订单类型，"buy"(买入)或"sell"(卖出)
        symbol: 订单的股票代码
        date: 订单成交的日期
        shares: 买入或卖出的股数
        limit_price: 订单使用的限价
        fill_price: 订单成交的价格
        fees: 订单的经纪费用
    """

    id: int  # 唯一标识符
    type: Literal["buy", "sell"]  # 订单类型
    symbol: str  # 股票代码
    date: np.datetime64  # 日期
    shares: Decimal  # 股数
    limit_price: Optional[Decimal]  # 限价
    fill_price: Decimal  # 成交价格
    fees: Decimal  # 费用


class PortfolioBar(NamedTuple):
    """每个K线捕获的Portfolio状态快照
    
    属性:
        date: K线的日期
        cash: Portfolio中的现金金额
        equity: Portfolio中的权益金额
        margin: Portfolio中的保证金金额
        market_value: Portfolio的市场价值
        pnl: Portfolio的已实现盈亏
        unrealized_pnl: Portfolio的未实现盈亏
        fees: 经纪费用
    """

    date: np.datetime64  # 日期
    cash: Decimal  # 现金
    equity: Decimal  # 权益
    margin: Decimal  # 保证金
    market_value: Decimal  # 市值
    pnl: Decimal  # 已实现盈亏
    unrealized_pnl: Decimal  # 未实现盈亏
    fees: Decimal  # 费用


class PositionBar(NamedTuple):
    r"""每个K线捕获的持仓状态快照
    
    属性:
        symbol: 持仓的股票代码
        date: K线的日期
        long_shares: 持仓中的多头股数
        short_shares: 持仓中的空头股数
        close: symbol的最新收盘价
        equity: 持仓中的权益金额
        market_value: 持仓的市场价值
        margin: 持仓中的保证金金额
        unrealized_pnl: 持仓的未实现盈亏
    """

    symbol: str  # 股票代码
    date: np.datetime64  # 日期
    long_shares: Decimal  # 多头股数
    short_shares: Decimal  # 空头股数
    close: Decimal  # 收盘价
    equity: Decimal  # 权益
    market_value: Decimal  # 市值
    margin: Decimal  # 保证金
    unrealized_pnl: Decimal  # 未实现盈亏


class _OrderResult(NamedTuple):
    """内部使用的订单结果类"""
    filled_shares: Decimal  # 成交股数
    rem_shares: Decimal  # 剩余股数


def _calculate_pnl_mae_mfe(
    pos: Position,
    close: Decimal,
    low: Optional[Decimal],
    high: Optional[Decimal],
):
    """计算持仓的盈亏、最大不利偏移(MAE)和最大有利偏移(MFE)
    
    根据当前价格、最低价和最高价更新持仓的盈亏和偏移指标
    """
    if pos.type != "long" and pos.type != "short":
        raise ValueError(f"Unknown position type: {pos.type}")
    pnl = Decimal()
    for entry in pos.entries:
        loss = Decimal()
        profit = Decimal()
        if pos.type == "long":
            pnl += (close - entry.price) * entry.shares
            if low is not None:
                loss = low - entry.price
            if high is not None:
                profit = high - entry.price
        elif pos.type == "short":
            pnl += (entry.price - close) * entry.shares
            if high is not None:
                loss = entry.price - high
            if low is not None:
                profit = entry.price - low
        if loss < 0 and loss < entry.mae:
            entry.mae = loss
        if profit > 0 and profit > entry.mfe:
            entry.mfe = profit
    pos.pnl = pnl


class Portfolio:
    r"""Class representing a portfolio of holdings. The portfolio contains
    information about open positions and balances, and is also used to place
    buy and sell orders.

    Args:
        cash: Starting cash balance.
        fee_mode: Brokerage fee mode.
        fee_amount: Brokerage fee amount.
        subtract_fees: Whether to subtract fees from the cash balance after an
            order is filled.
        enable_fractional_shares: Whether to enable trading fractional shares.
        position_mode: Position mode for :class:`.Portfolio`.
        max_long_positions: Maximum number of long :class:`.Position`\ s that
            can be held at a time. If ``None``, then unlimited.
        max_short_positions: Maximum number of short :class:`.Position`\ s that
            can be held at a time. If ``None``, then unlimited.
        record_stops: Whether to record stop data per-bar.

    Attributes:
        cash: Current cash balance.
        equity: Current amount of equity.
        market_value: Current market value. The market value is defined as
            the amount of equity held in cash and long positions added together
            with the unrealized PnL of all open short positions.
        fees: Current brokerage fees.
        fee_amount: Brokerage fee amount.
        subtract_fees: Whether to subtract fees from the cash balance.
        enable_fractional_shares: Whether to enable trading fractional shares.
        orders: ``deque`` of all filled orders, sorted in ascending
            chronological order.
        margin: Current amount of margin held in open positions.
        pnl: Realized profit and loss (PnL).
        long_positions: ``dict`` mapping ticker symbols to open long
            :class:`.Position`\ s.
        short_positions: ``dict`` mapping ticker symbols to open short
            :class:`.Position`\ s.
        symbols: Ticker symbols of all currently open positions.
        bars: ``deque`` of snapshots of :class:`.Portfolio` state on every bar,
            sorted in ascending chronological order.
        position_bars: ``deque`` of snapshots of :class:`.Position` states on
            every bar, sorted in ascending chronological order.
        win_rate: Running win rate of trades.
        loss_rate: Running loss rate of trades.
    """

    def __init__(
        self,
        cash: float,
        fee_mode: Optional[
            Union[FeeMode, Callable[[FeeInfo], Decimal], None]
        ] = None,
        fee_amount: Optional[float] = None,
        subtract_fees: bool = False,
        enable_fractional_shares: bool = False,
        position_mode: PositionMode = PositionMode.DEFAULT,
        max_long_positions: Optional[int] = None,
        max_short_positions: Optional[int] = None,
        record_stops: Optional[bool] = False,
    ):
        self.cash: Decimal = to_decimal(cash)
        self._initial_market_value = self.cash
        self._fee_mode = fee_mode
        self._fee_amount: Optional[Decimal] = (
            None if fee_amount is None else to_decimal(fee_amount)
        )
        self._subtract_fees = subtract_fees
        self._enable_fractional_shares = enable_fractional_shares
        self._position_mode = position_mode
        self.equity: Decimal = self.cash
        self.market_value: Decimal = self.cash
        self.fees = Decimal()
        self._max_long_positions = max_long_positions
        self._max_short_positions = max_short_positions
        self._record_stops = record_stops
        self.orders: deque[Order] = deque()
        self.trades: deque[Trade] = deque()
        self.margin: Decimal = Decimal()
        self.pnl: Decimal = Decimal()
        self.long_positions: dict[str, Position] = {}
        self.short_positions: dict[str, Position] = {}
        self.symbols: set[str] = set()
        self.bars: deque[PortfolioBar] = deque()
        self.position_bars: deque[PositionBar] = deque()
        self.win_rate: Decimal = Decimal()
        self.loss_rate: Decimal = Decimal()
        self._wins: Decimal = Decimal()
        self._logger = StaticScope.instance().logger
        self._stop_data: dict[int, _StopData] = {}
        self._order_id: int = 0
        self._entry_id: int = 0
        self._trade_id: int = 0
        self._stop_records: list[StopRecord] = []

    def _calculate_fees(
        self,
        symbol: str,
        fill_price: Decimal,
        shares: Decimal,
        order_type: Literal["buy", "sell"],
    ) -> Decimal:
        fees = Decimal()
        if self._fee_mode is None or self._fee_amount is None:
            return fees
        if callable(self._fee_mode):
            fees = to_decimal(
                self._fee_mode(
                    FeeInfo(
                        symbol=symbol,
                        shares=shares,
                        fill_price=fill_price,
                        order_type=order_type,
                    )
                )
            )
        elif self._fee_mode == FeeMode.ORDER_PERCENT:
            fees = self._fee_amount / _DECIMAL_100 * fill_price * shares
        elif self._fee_mode == FeeMode.PER_ORDER:
            fees = self._fee_amount
        elif self._fee_mode == FeeMode.PER_SHARE:
            fees = self._fee_amount * shares
        else:
            raise ValueError(f"Unknown FeeMode: {self._fee_mode!r}")
        return fees

    def _verify_input(
        self,
        shares: Union[int, float, Decimal],
        fill_price: Decimal,
        limit_price: Optional[Decimal],
    ):
        if shares < 0:
            raise ValueError(f"Shares cannot be negative: {shares}")
        if fill_price <= 0:
            raise ValueError(f"Fill price must be > 0: {fill_price}")
        if limit_price is not None and limit_price <= 0:
            raise ValueError(f"Limit price must be > 0: {limit_price}")

    def _add_entry(
        self,
        date: np.datetime64,
        symbol: str,
        shares: Decimal,
        price: Decimal,
        type: Literal["long", "short"],
        pos: Position,
    ) -> Entry:
        self._entry_id += 1
        entry = Entry(
            id=self._entry_id,
            symbol=symbol,
            shares=shares,
            price=price,
            date=date,
            type=type,
        )
        pos.entries.append(entry)
        return entry

    def _add_order(
        self,
        date: np.datetime64,
        symbol: str,
        type: Literal["buy", "sell"],
        limit_price: Optional[Decimal],
        fill_price: Decimal,
        shares: Decimal,
    ) -> Order:
        self._order_id += 1
        fees = self._calculate_fees(symbol, fill_price, shares, type)
        order = Order(
            id=self._order_id,
            date=date,
            symbol=symbol,
            type=type,
            limit_price=limit_price,
            fill_price=fill_price,
            shares=shares,
            fees=fees,
        )
        self.orders.append(order)
        self.fees += fees
        if self._subtract_fees:
            self.cash -= fees
        return order

    def _add_trade(
        self,
        type: Literal["long", "short"],
        symbol: str,
        entry_date: np.datetime64,
        exit_date: np.datetime64,
        entry_price: Decimal,
        exit_price: Decimal,
        shares: Decimal,
        pnl: Decimal,
        return_pct: Decimal,
        agg_pnl: Decimal,
        bars: int,
        pnl_per_bar: Decimal,
        stop_type: Optional[StopType],
        mae: Decimal,
        mfe: Decimal,
    ):
        self._trade_id += 1
        trade = Trade(
            id=self._trade_id,
            type=type,
            symbol=symbol,
            entry_date=entry_date,
            exit_date=exit_date,
            entry=entry_price,
            exit=exit_price,
            shares=shares,
            pnl=pnl,
            return_pct=return_pct,
            agg_pnl=agg_pnl,
            bars=bars,
            pnl_per_bar=pnl_per_bar,
            stop=None if stop_type is None else stop_type.value,
            mae=mae,
            mfe=mfe,
        )
        self.trades.append(trade)
        if pnl > 0:
            self._wins += 1
        self.win_rate = self._wins / len(self.trades)
        self.loss_rate = 1 - self.win_rate

    def _get_stop_amount(self, stop: Stop, price: Decimal) -> Decimal:
        if stop.percent is not None:
            return price * stop.percent / 100
        elif stop.points is not None:
            return stop.points
        else:
            raise ValueError("Stop amount not set.")

    def _add_stops(self, entry: Entry, stops: Iterable[Stop]):
        for stop in stops:
            if stop.id in self._stop_data:
                raise ValueError(f"Duplicate stop ID: {stop.id}")
            entry.stops.append(stop)
            if stop.stop_type == StopType.BAR:
                continue
            amount = self._get_stop_amount(stop, entry.price)
            if (
                stop.pos_type == "long" and stop.stop_type == StopType.PROFIT
            ) or (
                stop.pos_type == "short"
                and (
                    stop.stop_type == StopType.LOSS
                    or stop.stop_type == StopType.TRAILING
                )
            ):
                stop_value = entry.price + amount
            else:
                stop_value = entry.price - amount
            self._stop_data[stop.id] = _StopData(
                value=stop_value, stop=stop, entry=entry
            )

    def _remove_stop_data(self, entry: Entry):
        for stop in entry.stops:
            if stop.id in self._stop_data:
                del self._stop_data[stop.id]

    def _clamp_shares(self, fill_price: Decimal, shares: Decimal) -> Decimal:
        if self.cash < 0:
            return Decimal()
        max_shares = (
            Decimal(self.cash / fill_price)
            if self._enable_fractional_shares
            else Decimal(self.cash // fill_price)
        )
        return min(shares, max_shares)

    def buy(
        self,
        date: np.datetime64,
        symbol: str,
        shares: Decimal,
        fill_price: Decimal,
        limit_price: Optional[Decimal] = None,
        stops: Optional[Iterable[Stop]] = None,
    ) -> Optional[Order]:
        r"""Places a buy order.

        Args:
            date: Date when the :class:`.Order` is placed.
            symbol: Ticker symbol to buy.
            shares: Number of shares to buy.
            fill_price: If filled, the price used to fill the :class:`.Order`.
            limit_price: Limit price of the :class:`.Order`.
            stops: :class:`.Stop`\ s to set on the :class:`.Entry` created from
                the :class:`.Order`, if filled.

        Returns:
            :class:`.Order` if the order was filled, otherwise ``None``.
        """
        self._verify_input(shares, fill_price, limit_price)
        self._logger.debug_place_buy_order(
            date=date,
            symbol=symbol,
            shares=shares,
            fill_price=fill_price,
            limit_price=limit_price,
        )
        if limit_price is not None and limit_price < fill_price:
            return None
        if shares == 0:
            return None
        covered = self._cover(date, symbol, shares, fill_price)
        bought_shares = self._long(
            date, symbol, covered.rem_shares, fill_price, limit_price, stops
        )
        if not covered.filled_shares and not bought_shares:
            return None
        order = self._add_order(
            date=date,
            symbol=symbol,
            type="buy",
            limit_price=limit_price,
            fill_price=fill_price,
            shares=covered.filled_shares + bought_shares,
        )
        return order

    def _cover(
        self,
        date: np.datetime64,
        symbol: str,
        shares: Decimal,
        fill_price: Decimal,
    ) -> _OrderResult:
        if symbol not in self.short_positions:
            return _OrderResult(Decimal(), shares)
        rem_shares = shares
        if rem_shares <= 0:
            return _OrderResult(Decimal(), shares)
        pos = self.short_positions[symbol]
        while pos.entries:
            entry = pos.entries[0]
            if rem_shares >= entry.shares:
                rem_shares -= entry.shares
                self._exit_short(
                    date, pos, entry, entry.shares, fill_price, stop_type=None
                )
                self._remove_stop_data(entry)
                pos.entries.popleft()
            else:
                self._exit_short(
                    date, pos, entry, rem_shares, fill_price, stop_type=None
                )
                rem_shares = Decimal()
                break
        self._update_position(pos)
        return _OrderResult(shares - rem_shares, rem_shares)

    def _exit_short(
        self,
        date: np.datetime64,
        pos: Position,
        entry: Entry,
        shares: Decimal,
        fill_price: Decimal,
        stop_type: Optional[StopType],
    ):
        order_amount = shares * fill_price
        entry_amount = shares * entry.price
        entry_pnl = entry_amount - order_amount
        self.pnl += entry_pnl
        self.cash += entry_pnl
        pos.shares -= shares
        entry.shares -= shares
        pnl_per_bar = entry_pnl if not entry.bars else entry_pnl / entry.bars
        return_pct = ((entry.price / fill_price) - 1) * 100
        pnl = entry.price - fill_price
        mae = pnl if pnl < 0 and pnl < entry.mae else entry.mae
        mfe = pnl if pnl > 0 and pnl > entry.mfe else entry.mfe
        self._add_trade(
            type=entry.type,
            symbol=entry.symbol,
            entry_date=entry.date,
            exit_date=date,
            entry_price=entry.price,
            exit_price=fill_price,
            shares=shares,
            pnl=entry_pnl,
            return_pct=return_pct,
            agg_pnl=self.pnl,
            bars=entry.bars,
            pnl_per_bar=pnl_per_bar,
            stop_type=stop_type,
            mae=mae,
            mfe=mfe,
        )

    def _long(
        self,
        date: np.datetime64,
        symbol: str,
        shares: Decimal,
        fill_price: Decimal,
        limit_price: Optional[Decimal],
        stops: Optional[Iterable[Stop]],
    ) -> Decimal:
        if self._position_mode == PositionMode.SHORT_ONLY:
            return Decimal()
        clamped_shares = self._clamp_shares(fill_price, shares)
        if clamped_shares < shares:
            self._logger.debug_buy_shares_exceed_cash(
                date=date,
                symbol=symbol,
                shares=shares,
                fill_price=fill_price,
                limit_price=limit_price,
                cash=self.cash,
                clamped_shares=clamped_shares,
            )
            shares = clamped_shares
        if shares <= 0:
            return Decimal()
        if (
            self._max_long_positions is not None
            and symbol not in self.long_positions
            and len(self.long_positions) == self._max_long_positions
        ):
            return Decimal()
        order_amount = shares * fill_price
        self.cash -= order_amount
        if symbol not in self.long_positions:
            self.symbols.add(symbol)
            pos = Position(symbol=symbol, shares=shares, type="long")
            self.long_positions[symbol] = pos
        else:
            pos = self.long_positions[symbol]
            pos.shares += shares
        entry = self._add_entry(
            date=date,
            symbol=symbol,
            shares=shares,
            price=fill_price,
            type="long",
            pos=pos,
        )
        if stops is not None:
            self._add_stops(entry, stops)
        return shares

    def sell(
        self,
        date: np.datetime64,
        symbol: str,
        shares: Decimal,
        fill_price: Decimal,
        limit_price: Optional[Decimal] = None,
        stops: Optional[Iterable[Stop]] = None,
    ) -> Optional[Order]:
        r"""Places a sell order.

        Args:
            date: Date when the :class:`.Order` is placed.
            symbol: Ticker symbol to sell.
            shares: Number of shares to sell.
            fill_price: If filled, the price used to fill the :class:`.Order`.
            limit_price: Limit price of the :class:`.Order`.
            stops: :class:`.Stop`\ s to set on the :class:`.Entry` created from
                the :class:`.Order`, if filled.

        Returns:
            :class:`.Order` if the order was filled, otherwise ``None``.
        """
        self._verify_input(shares, fill_price, limit_price)
        self._logger.debug_place_sell_order(
            date=date,
            symbol=symbol,
            shares=shares,
            fill_price=fill_price,
            limit_price=limit_price,
        )
        if limit_price is not None and limit_price > fill_price:
            return None
        if shares == 0:
            return None
        sold = self._sell_existing(date, symbol, shares, fill_price)
        short_shares = self._short(
            date, symbol, sold.rem_shares, fill_price, stops
        )
        if not sold.filled_shares and not short_shares:
            return None
        order = self._add_order(
            date=date,
            symbol=symbol,
            type="sell",
            limit_price=limit_price,
            fill_price=fill_price,
            shares=sold.filled_shares + short_shares,
        )
        return order

    def _sell_existing(
        self,
        date: np.datetime64,
        symbol: str,
        shares: Decimal,
        fill_price: Decimal,
    ) -> _OrderResult:
        if symbol not in self.long_positions:
            return _OrderResult(Decimal(), shares)
        rem_shares = shares
        pos = self.long_positions[symbol]
        while pos.entries:
            entry = pos.entries[0]
            if rem_shares >= entry.shares:
                rem_shares -= entry.shares
                self._exit_long(
                    date, pos, entry, entry.shares, fill_price, stop_type=None
                )
                self._remove_stop_data(entry)
                pos.entries.popleft()
            else:
                self._exit_long(
                    date, pos, entry, rem_shares, fill_price, stop_type=None
                )
                rem_shares = Decimal()
                break
        self._update_position(pos)
        return _OrderResult(shares - rem_shares, rem_shares)

    def _exit_long(
        self,
        date: np.datetime64,
        pos: Position,
        entry: Entry,
        shares: Decimal,
        fill_price: Decimal,
        stop_type: Optional[StopType],
    ):
        order_amount = shares * fill_price
        entry_amount = shares * entry.price
        entry_pnl = order_amount - entry_amount
        self.pnl += entry_pnl
        self.cash += order_amount
        pos.shares -= shares
        entry.shares -= shares
        pnl_per_bar = entry_pnl if not entry.bars else entry_pnl / entry.bars
        return_pct = ((fill_price / entry.price) - 1) * 100
        pnl = fill_price - entry.price
        mae = pnl if pnl < 0 and pnl < entry.mae else entry.mae
        mfe = pnl if pnl > 0 and pnl > entry.mfe else entry.mfe
        self._add_trade(
            type=entry.type,
            symbol=entry.symbol,
            entry_date=entry.date,
            exit_date=date,
            entry_price=entry.price,
            exit_price=fill_price,
            shares=shares,
            pnl=entry_pnl,
            return_pct=return_pct,
            agg_pnl=self.pnl,
            bars=entry.bars,
            pnl_per_bar=pnl_per_bar,
            stop_type=stop_type,
            mae=mae,
            mfe=mfe,
        )

    def _update_position(self, pos: Position):
        if pos.entries:
            return
        if pos.type == "long":
            if pos.symbol in self.long_positions:
                del self.long_positions[pos.symbol]
        else:
            if pos.symbol in self.short_positions:
                del self.short_positions[pos.symbol]
        if (
            pos.symbol in self.symbols
            and pos.symbol not in self.long_positions
            and pos.symbol not in self.short_positions
        ):
            self.symbols.remove(pos.symbol)

    def _short(
        self,
        date: np.datetime64,
        symbol: str,
        shares: Decimal,
        fill_price: Decimal,
        stops: Optional[Iterable[Stop]],
    ) -> Decimal:
        if shares <= 0:
            return Decimal()
        if (
            self._max_short_positions is not None
            and symbol not in self.short_positions
            and len(self.short_positions) == self._max_short_positions
        ):
            return Decimal()
        if self._position_mode == PositionMode.LONG_ONLY:
            return Decimal()
        if symbol not in self.short_positions:
            self.symbols.add(symbol)
            pos = Position(symbol=symbol, shares=shares, type="short")
            self.short_positions[symbol] = pos
        else:
            pos = self.short_positions[symbol]
            pos.shares += shares
        entry = self._add_entry(
            date=date,
            symbol=symbol,
            shares=shares,
            price=fill_price,
            type="short",
            pos=pos,
        )
        if stops is not None:
            self._add_stops(entry, stops)
        return shares

    def exit_position(
        self,
        date: np.datetime64,
        symbol: str,
        buy_fill_price: Decimal,
        sell_fill_price: Decimal,
    ):
        """Exits any long and short positions for ``symbol`` at
        ``buy_fill_price`` and ``sell_fill_price``.
        """
        if symbol in self.long_positions:
            self.sell(
                date=date,
                symbol=symbol,
                shares=self.long_positions[symbol].shares,
                fill_price=sell_fill_price,
            )
        if symbol in self.short_positions:
            self.buy(
                date=date,
                symbol=symbol,
                shares=self.short_positions[symbol].shares,
                fill_price=buy_fill_price,
            )

    def capture_bar(self, date: np.datetime64, df: pd.DataFrame):
        """Captures portfolio state of the current bar.

        Args:
            date: Date of current bar.
            df: :class:`pandas.DataFrame` containing close prices.
        """
        total_equity = self.cash
        total_market_value = total_equity
        total_margin = Decimal()
        for sym in self.symbols:
            index = (sym, date)
            close = None
            low = None
            high = None
            if index in df.index:
                df_row = df.loc[index].squeeze()
                if isinstance(df_row, pd.core.frame.DataFrame):
                    raise ValueError(
                        "df has the same index. index:" + str(index)
                    )
                close = to_decimal(df_row[DataCol.CLOSE.value])
                low = to_decimal(df_row[DataCol.LOW.value])
                high = to_decimal(df_row[DataCol.HIGH.value])
            pos_long_shares = Decimal()
            pos_short_shares = Decimal()
            pos_equity = Decimal()
            pos_market_value = Decimal()
            pos_margin = Decimal()
            pos_pnl = Decimal()
            if sym in self.long_positions:
                pos = self.long_positions[sym]
                if close is not None:
                    _calculate_pnl_mae_mfe(
                        pos, close=close, low=low, high=high
                    )
                    pos.equity = pos.shares * close
                    pos.market_value = pos.equity
                    pos.close = close
                    pos_long_shares += pos.shares
                    pos_equity += pos.equity
                    pos_market_value += pos.market_value
                    pos_pnl += pos.pnl
                total_equity += pos.equity
                total_market_value += pos.equity
            if sym in self.short_positions:
                pos = self.short_positions[sym]
                if close is not None:
                    _calculate_pnl_mae_mfe(
                        pos, close=close, low=low, high=high
                    )
                    pos.close = close
                    pos.margin = close * pos.shares
                    pos.market_value = pos.margin + pos.pnl
                    pos_margin += pos.margin
                    pos_short_shares += pos.shares
                    pos_market_value += pos.market_value
                    pos_pnl += pos.pnl
                total_margin += pos.margin
                total_market_value += pos.pnl
            if close is not None:
                self.position_bars.append(
                    PositionBar(
                        symbol=sym,
                        date=date,
                        long_shares=pos_long_shares,
                        short_shares=pos_short_shares,
                        close=close,
                        equity=pos_equity,
                        market_value=pos_market_value,
                        margin=pos_margin,
                        unrealized_pnl=pos_pnl,
                    )
                )

        self.equity = total_equity
        self.market_value = total_market_value
        self.margin = total_margin

        self.bars.append(
            PortfolioBar(
                date=date,
                cash=self.cash,
                equity=self.equity,
                market_value=self.market_value,
                margin=self.margin,
                pnl=self.equity - self._initial_market_value,
                unrealized_pnl=self.market_value - self.equity,
                fees=self.fees,
            )
        )

    def incr_bars(self):
        """Increments the number of bars held by every trade entry."""
        for pos in itertools.chain(
            self.long_positions.values(), self.short_positions.values()
        ):
            pos.bars += 1
            for entry in pos.entries:
                entry.bars += 1

    def remove_stop(self, stop_id: int) -> bool:
        """Removes a :class:`.Stop` with ``stop_id``."""
        if stop_id in self._stop_data:
            stop_data = self._stop_data[stop_id]
            del self._stop_data[stop_id]
            if stop_data.stop in stop_data.entry.stops:
                stop_data.entry.stops.remove(stop_data.stop)
            return True
        return False

    def remove_stops(
        self,
        val: Union[str, Position, Entry],
        stop_type: Optional[StopType] = None,
    ):
        r"""Removes :class:`.Stop`\ s.

        Args:
            val: Ticker symbol, :class:`.Position`, or :class:`.Entry` for
                which to cancel stops.
            stop_type: :class:`pybroker.common.StopType`.
        """
        if isinstance(val, str):
            if val in self.long_positions:
                self._remove_position_stops(
                    self.long_positions[val], stop_type
                )
            if val in self.short_positions:
                self._remove_position_stops(
                    self.short_positions[val], stop_type
                )
        elif isinstance(val, Position):
            self._remove_position_stops(val, stop_type)
        elif isinstance(val, Entry):
            self._remove_entry_stops(val, stop_type)

    def _remove_position_stops(
        self, pos: Position, stop_type: Optional[StopType]
    ):
        for entry in pos.entries:
            self._remove_entry_stops(entry, stop_type)

    def _remove_entry_stops(self, entry: Entry, stop_type: Optional[StopType]):
        if stop_type is None:
            self._remove_stop_data(entry)
            entry.stops.clear()
        else:
            stop_id = None
            for stop in entry.stops:
                if stop.stop_type == stop_type:
                    stop_id = stop.id
                    break
            if stop_id is not None:
                self.remove_stop(stop_id)

    def check_stops(self, date: np.datetime64, price_scope: PriceScope):
        """Checks whether stops are triggered."""
        executed: deque[tuple[Position, Entry]] = deque()
        for pos in itertools.chain(
            self.long_positions.values(), self.short_positions.values()
        ):
            for entry in pos.entries:
                for stop in entry.stops:
                    triggered, fill_price = self._trigger_stop(
                        date, price_scope, pos, entry, stop
                    )
                    if self._record_stops:
                        self._capture_stop(date, entry, stop, fill_price)
                    if triggered:
                        executed.append((pos, entry))
                        break

        for pos, entry in executed:
            pos.entries.remove(entry)
            self._remove_stop_data(entry)
            self._update_position(pos)

    def _capture_stop(
        self,
        date: np.datetime64,
        entry: Entry,
        stop: Stop,
        fill_price: Optional[Decimal],
    ):
        stop_record = StopRecord(
            date=date,
            stop_id=stop.id,
            symbol=stop.symbol,
            stop_type=stop.stop_type.value,
            pos_type=stop.pos_type,
            curr_value=(
                self._stop_data[stop.id].value
                if stop.id in self._stop_data
                else None
            ),
            curr_bars=entry.bars if stop.stop_type == StopType.BAR else None,
            bars=stop.bars,
            percent=stop.percent,
            points=stop.points,
            limit_price=stop.limit_price,
            exit_price=stop.exit_price,
            fill_price=fill_price,
        )
        self._stop_records.append(stop_record)

    def _trigger_stop(
        self,
        date: np.datetime64,
        price_scope: PriceScope,
        pos: Position,
        entry: Entry,
        stop: Stop,
    ) -> tuple[bool, Optional[Decimal]]:
        fill_price = None
        if stop.pos_type == "long" and stop.symbol not in self.long_positions:
            return False, fill_price
        if (
            stop.pos_type == "short"
            and stop.symbol not in self.short_positions
        ):
            return False, fill_price
        if stop.stop_type == StopType.BAR:
            fill_price = self._trigger_bar_stop(stop, price_scope, entry)
        elif (
            stop.stop_type == StopType.LOSS
            or stop.stop_type == StopType.PROFIT
        ):
            fill_price = self._trigger_profit_or_loss_stop(stop, price_scope)
        elif stop.stop_type == StopType.TRAILING:
            fill_price = self._trigger_trailing_stop(stop, price_scope)
        else:
            raise ValueError(f"Unknown stop type: {stop.stop_type}")
        if fill_price is None:
            return False, fill_price
        order_type: Literal["buy", "sell"]
        stop_shares = entry.shares
        if stop.pos_type == "long":
            if stop.limit_price is not None and fill_price < stop.limit_price:
                return False, fill_price
            self._exit_long(
                date, pos, entry, entry.shares, fill_price, stop.stop_type
            )
            order_type = "sell"
        elif stop.pos_type == "short":
            if stop.limit_price is not None and fill_price > stop.limit_price:
                return False, fill_price
            self._exit_short(
                date, pos, entry, entry.shares, fill_price, stop.stop_type
            )
            order_type = "buy"
        else:
            raise ValueError(f"Unknown pos_type: {stop.pos_type}")
        self._add_order(
            date=date,
            symbol=pos.symbol,
            type=order_type,
            limit_price=stop.limit_price,
            fill_price=fill_price,
            shares=stop_shares,
        )
        return True, fill_price

    def _trigger_bar_stop(
        self, stop: Stop, price_scope: PriceScope, entry: Entry
    ) -> Optional[Decimal]:
        if stop.bars is None:
            raise ValueError("Bars not set on bar stop.")
        if entry.bars >= stop.bars:
            return price_scope.fetch(
                stop.symbol,
                (
                    PriceType.MIDDLE
                    if stop.fill_price is None
                    else stop.fill_price
                ),
            )
        return None

    def _trigger_profit_or_loss_stop(
        self, stop: Stop, price_scope: PriceScope
    ) -> Optional[Decimal]:
        if (
            stop.pos_type == "long"
            and (
                stop.stop_type == StopType.LOSS
                or stop.stop_type == StopType.TRAILING
            )
        ) or (stop.pos_type == "short" and stop.stop_type == StopType.PROFIT):
            if stop.exit_price is not None:
                exit_price = price_scope.fetch(stop.symbol, stop.exit_price)
                if exit_price <= self._stop_data[stop.id].value:
                    return exit_price
            else:
                low = price_scope.fetch(stop.symbol, PriceType.LOW)
                if low <= self._stop_data[stop.id].value:
                    high = price_scope.fetch(stop.symbol, PriceType.HIGH)
                    return min(self._stop_data[stop.id].value, high)
        elif (
            stop.pos_type == "long" and stop.stop_type == StopType.PROFIT
        ) or (
            stop.pos_type == "short"
            and (
                stop.stop_type == StopType.LOSS
                or stop.stop_type == StopType.TRAILING
            )
        ):
            if stop.exit_price is not None:
                exit_price = price_scope.fetch(stop.symbol, stop.exit_price)
                if exit_price >= self._stop_data[stop.id].value:
                    return exit_price
            else:
                high = price_scope.fetch(stop.symbol, PriceType.HIGH)
                if high >= self._stop_data[stop.id].value:
                    low = price_scope.fetch(stop.symbol, PriceType.LOW)
                    return max(self._stop_data[stop.id].value, low)
        return None

    def _trigger_trailing_stop(
        self, stop: Stop, price_scope: PriceScope
    ) -> Optional[Decimal]:
        fill_price = self._trigger_profit_or_loss_stop(stop, price_scope)
        if fill_price is not None:
            return fill_price
        if stop.pos_type == "long":
            high = price_scope.fetch(stop.symbol, PriceType.HIGH)
            amount = self._get_stop_amount(stop, high)
            self._stop_data[stop.id].value = max(
                high - amount, self._stop_data[stop.id].value
            )
        else:
            low = price_scope.fetch(stop.symbol, PriceType.LOW)
            amount = self._get_stop_amount(stop, low)
            self._stop_data[stop.id].value = min(
                low + amount, self._stop_data[stop.id].value
            )
        return None
