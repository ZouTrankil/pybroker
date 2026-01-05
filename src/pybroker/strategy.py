"""
包含回测交易策略的实现。

本模块是PyBroker回测框架的核心，主要包含以下组件：
1. BacktestMixin - 回测执行的核心逻辑
2. WalkforwardMixin - 滚动分析(Walkforward Analysis)的实现
3. Strategy - 交易策略的主类，整合所有功能

回测流程概述：
================
1. 数据准备阶段：获取历史数据、计算指标、训练模型
2. 回测执行阶段：按时间顺序遍历每个交易日
3. 信号生成阶段：执行用户定义的交易逻辑，生成买卖信号
4. 订单调度阶段：根据延迟配置安排订单执行时间
5. 订单执行阶段：在指定日期执行买卖订单
6. 结果汇总阶段：计算评估指标，生成回测报告
"""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import dataclasses
import numpy as np
import pandas as pd
from pybroker.cache import CacheDateFields
from pybroker.common import (
    BarData,  # K线数据
    DataCol,  # 数据列枚举
    Day,  # 日期枚举
    IndicatorSymbol,  # 指标符号类型
    ModelSymbol,  # 模型符号类型
    PriceType,  # 价格类型枚举
    get_unique_sorted_dates,  # 获取唯一排序日期
    quantize,  # 量化函数
    to_datetime,  # 转换为日期时间
    to_decimal,  # 转换为小数
    to_seconds,  # 转换为秒
    verify_data_source_columns,  # 验证数据源列
    verify_date_range,  # 验证日期范围
)
from pybroker.config import StrategyConfig  # 策略配置
from pybroker.context import (
    ExecContext,  # 执行上下文
    ExecResult,  # 执行结果
    PosSizeContext,  # 仓位大小上下文
    set_exec_ctx_data,  # 设置执行上下文数据
    set_pos_size_ctx_data,  # 设置仓位大小上下文数据
)
from pybroker.data import AlpacaCrypto, DataSource  # 数据源
from pybroker.eval import BootstrapResult, EvalMetrics, EvaluateMixin  # 评估工具
from pybroker.indicator import Indicator, IndicatorsMixin  # 指标
from pybroker.model import ModelSource, ModelsMixin, TrainedModel  # 模型
from pybroker.portfolio import (
    Order,  # 订单
    Portfolio,  # 投资组合
    PortfolioBar,  # 投资组合K线
    PositionBar,  # 仓位K线
    StopRecord,  # 止损止盈记录
    Trade,  # 交易
)
from pybroker.scope import (
    ColumnScope,  # 列作用域
    IndicatorScope,  # 指标作用域
    ModelInputScope,  # 模型输入作用域
    PendingOrderScope,  # 待处理订单作用域
    PredictionScope,  # 预测作用域
    PriceScope,  # 价格作用域
    StaticScope,  # 静态作用域
    get_signals,  # 获取信号
)
from pybroker.slippage import SlippageModel  # 滑点模型
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from numpy.typing import NDArray
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    MutableMapping,
    NamedTuple,
    Optional,
    Union,
)


def _between(
    df: pd.DataFrame, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    """
    筛选指定日期范围内的数据框。
    
    该函数用于从完整的数据集中提取指定日期范围的数据。
    会处理时区信息，确保日期比较的准确性。
    
    参数:
        df: 原始数据框，必须包含'date'列
        start_date: 开始日期（包含）
        end_date: 结束日期（包含）
        
    返回:
        筛选后的数据框，仅包含指定日期范围内的行
    """
    if df.empty:
        return df
    return df[
        (df[DataCol.DATE.value].dt.tz_localize(None) >= start_date)
        & (df[DataCol.DATE.value].dt.tz_localize(None) <= end_date)
    ]


def _sort_by_score(result: ExecResult) -> float:
    """
    根据执行结果的分数进行排序的键函数。
    
    当有多个交易信号竞争有限的仓位时（如设置了max_long_positions），
    需要按分数对信号进行排序，优先执行分数高的信号。
    
    分数由用户在交易逻辑中通过ctx.score设置，用于表示信号的强度或优先级。
    
    参数:
        result: 执行结果对象
        
    返回:
        分数值，如果未设置分数则返回0.0
    """
    return 0.0 if result.score is None else result.score


class Execution(NamedTuple):
    r"""
    执行对象 - 代表一个交易策略的执行单元。
    
    每个Execution将一组股票代码与一个交易逻辑函数关联起来。
    在回测过程中，对于每个股票的每根K线，都会调用对应的交易逻辑函数。
    
    交易逻辑函数接收一个ExecContext对象，可以通过它：
    - 访问历史价格和成交量数据
    - 获取技术指标值
    - 获取模型预测结果
    - 查询当前持仓状态
    - 发出买卖信号
    
    示例::
    
        def my_strategy(ctx):
            # 获取最近20天的收盘价
            closes = ctx.close[-20:]
            # 获取SMA指标值
            sma = ctx.indicator('sma20')[-1]
            # 如果当前价格低于SMA，买入100股
            if ctx.close[-1] < sma:
                ctx.buy_shares = 100
    
    属性:
        id: 唯一标识符，用于区分不同的执行
        symbols: 该执行适用的股票代码集合
        fn: 交易逻辑函数，接收ExecContext参数
        model_names: 该执行使用的模型名称集合
        indicator_names: 该执行使用的指标名称集合
    """

    id: int  # 执行的唯一ID
    symbols: frozenset[str]  # 适用的股票代码集合（不可变）
    fn: Optional[Callable[[ExecContext], None]]  # 交易逻辑函数（可为None用于仅计算指标/模型）
    model_names: frozenset[str]  # 使用的模型名称集合
    indicator_names: frozenset[str]  # 使用的指标名称集合


class BacktestMixin:
    """
    回测功能的混入类。
    
    该类实现了回测的核心逻辑，包括：
    - 遍历历史数据的每个时间点
    - 执行用户定义的交易逻辑
    - 处理买卖订单的调度和执行
    - 管理止损止盈订单
    - 应用滑点模型
    """

    def backtest_executions(
        self,
        config: StrategyConfig,  # 策略配置对象，包含回测的各项参数设置
        executions: set[Execution],  # 执行集合，包含所有需要运行的交易逻辑
        before_exec_fn: Optional[Callable[[Mapping[str, ExecContext]], None]],  # 在所有交易逻辑执行前调用的函数
        after_exec_fn: Optional[Callable[[Mapping[str, ExecContext]], None]],  # 在所有交易逻辑执行后调用的函数
        sessions: Mapping[str, MutableMapping],  # 会话数据，用于在K线之间保持状态
        models: Mapping[ModelSymbol, TrainedModel],  # 已训练模型的映射，键为(模型名, 股票代码)
        indicator_data: Mapping[IndicatorSymbol, pd.Series],  # 指标数据映射，键为(指标名, 股票代码)
        test_data: pd.DataFrame,  # 测试数据集，包含所有股票的历史K线数据
        portfolio: Portfolio,  # 投资组合对象，管理持仓和资金
        pos_size_handler: Optional[Callable[[PosSizeContext], None]],  # 仓位大小处理函数，用于动态调整下单数量
        exit_dates: Mapping[str, np.datetime64],  # 各股票的强制退出日期映射
        train_only: bool = False,  # 如果为True，仅返回信号数据，不执行实际回测
        slippage_model: Optional[SlippageModel] = None,  # 滑点模型，模拟真实交易中的滑点
        enable_fractional_shares: bool = False,  # 是否允许交易分数股（如0.5股）
        round_fill_price: bool = True,  # 是否将成交价四舍五入到分
        warmup: Optional[int] = None,  # 预热期K线数量，在此期间不执行交易逻辑
    ) -> dict[str, pd.DataFrame]:
        r"""回测一组实现交易逻辑的执行。

        参数:
            config: 策略配置
            executions: 要运行的执行集合
            sessions: 在每个K线期间持续存在的自定义数据映射
            models: 模型符号对到已训练模型的映射
            indicator_data: 指标符号对到指标值的映射
            test_data: 测试数据框
            portfolio: 投资组合
            pos_size_handler: 设置买卖信号下单时仓位大小的函数
            exit_dates: 符号到退出日期的映射
            train_only: 是否仅用交易规则运行回测或仅训练模型
            enable_fractional_shares: 是否启用分数股交易
            round_fill_price: 是否将成交价四舍五入到最接近的分
            warmup: 运行执行前需要经过的K线数量

        返回:
            当StrategyConfig.return_signals为True时，返回包含每个符号的K线数据、
            指标数据和模型预测的数据框字典。
        """
        # =====================================================================
        # 第一步：数据准备和作用域初始化
        # =====================================================================
        
        # 获取所有唯一的交易日期，按时间排序
        test_dates = get_unique_sorted_dates(test_data[DataCol.DATE.value])
        # 获取所有参与回测的股票代码
        test_syms = sorted(test_data[DataCol.SYMBOL.value].unique())
        # 重设数据框索引为(股票代码, 日期)的多级索引，便于后续快速查询
        test_data = (
            test_data.reset_index(drop=True)
            .set_index([DataCol.SYMBOL.value, DataCol.DATE.value])
            .sort_index()
        )
        
        # 创建各种作用域对象，用于在回测过程中提供数据访问
        col_scope = ColumnScope(test_data)  # 列作用域：提供对K线数据列的访问
        ind_scope = IndicatorScope(indicator_data, test_dates)  # 指标作用域：提供对技术指标的访问
        input_scope = ModelInputScope(col_scope, ind_scope, models)  # 模型输入作用域：为模型准备输入特征
        pred_scope = PredictionScope(models, input_scope)  # 预测作用域：提供模型预测结果的访问
        
        # 如果仅训练模式，直接返回信号数据，不执行回测
        if train_only:
            if config.return_signals:
                return get_signals(test_syms, col_scope, ind_scope, pred_scope)
            return {}
        
        # =====================================================================
        # 第二步：初始化回测状态变量
        # =====================================================================
        
        # 记录每个股票当前处理到的K线索引位置
        sym_end_index: dict[str, int] = defaultdict(int)
        # 价格作用域：用于获取成交价格（开盘价、收盘价等）
        price_scope = PriceScope(col_scope, sym_end_index, round_fill_price)
        # 待处理订单作用域：管理已调度但尚未执行的订单
        pending_order_scope = PendingOrderScope()
        # 执行上下文字典：每个股票对应一个执行上下文
        exec_ctxs: dict[str, ExecContext] = {}
        # 执行函数字典：每个股票对应的交易逻辑函数
        exec_fns: dict[str, Callable[[ExecContext], None]] = {}
        # =====================================================================
        # 第三步：为每个股票创建执行上下文
        # =====================================================================
        
        for sym in test_syms:  # 遍历所有参与回测的股票代码
            for exec in executions:  # 遍历所有执行对象
                if sym not in exec.symbols:
                    continue
                # 为每个股票创建执行上下文，包含该股票在回测过程中需要的所有信息
                exec_ctxs[sym] = ExecContext(
                    symbol=sym,  # 股票代码
                    config=config,  # 策略配置
                    portfolio=portfolio,  # 投资组合引用
                    col_scope=col_scope,  # K线数据访问
                    ind_scope=ind_scope,  # 指标数据访问
                    input_scope=input_scope,  # 模型输入访问
                    pred_scope=pred_scope,  # 模型预测访问
                    pending_order_scope=pending_order_scope,  # 待处理订单访问
                    models=models,  # 已训练模型
                    sym_end_index=sym_end_index,  # 当前K线索引
                    session=sessions[sym],  # 会话数据（用户自定义状态）
                )
                # 保存该股票对应的交易逻辑函数
                if exec.fn is not None:
                    exec_fns[sym] = exec.fn
        
        # 创建每个股票的交易日期集合，用于快速判断某天是否有该股票的数据
        sym_exec_dates = {
            sym: frozenset(test_data.loc[pd.IndexSlice[sym, :]].index.values)
            for sym in exec_ctxs.keys()
        }
        
        # =====================================================================
        # 第四步：初始化订单调度器
        # =====================================================================
        # 订单不会立即执行，而是先调度到未来某个日期执行（由buy_delay/sell_delay控制）
        
        # 平仓买入订单调度器（用于空头平仓）
        cover_sched: dict[np.datetime64, list[ExecResult]] = defaultdict(list)
        # 开仓买入订单调度器（用于多头开仓）
        buy_sched: dict[np.datetime64, list[ExecResult]] = defaultdict(list)
        # 卖出订单调度器（用于多头平仓或空头开仓）
        sell_sched: dict[np.datetime64, list[ExecResult]] = defaultdict(list)
        # 如果设置了仓位大小处理函数，创建仓位大小上下文
        if pos_size_handler is not None:
            pos_ctx = PosSizeContext(
                config=config,
                portfolio=portfolio,
                col_scope=col_scope,
                ind_scope=ind_scope,
                input_scope=input_scope,
                pred_scope=pred_scope,
                pending_order_scope=pending_order_scope,
                models=models,
                sessions=sessions,
                sym_end_index=sym_end_index,
            )
        
        # 初始化日志记录器
        logger = StaticScope.instance().logger
        logger.backtest_executions_start(test_dates)
        
        # =====================================================================
        # 第五步：初始化结果收集队列
        # =====================================================================
        
        # 平仓买入结果队列（空头平仓信号）
        cover_results: deque[ExecResult] = deque()
        # 开仓买入结果队列（多头开仓信号）
        buy_results: deque[ExecResult] = deque()
        # 卖出结果队列（多头平仓或空头开仓信号）
        sell_results: deque[ExecResult] = deque()
        # 需要强制退出仓位的上下文队列
        exit_ctxs: deque[ExecContext] = deque()
        # 当前日期活跃的执行上下文（有数据的股票）
        active_ctxs: dict[str, ExecContext] = {}
        # =====================================================================
        # 第六步：主回测循环 - 遍历每个交易日期
        # =====================================================================
        # 这是回测的核心循环，按时间顺序处理每一个交易日
        
        for i, date in enumerate(test_dates):
            # -----------------------------------------------------------------
            # 6.1 更新当前日期的活跃上下文
            # -----------------------------------------------------------------
            active_ctxs.clear()  # 清空上一个日期的活跃上下文
            
            for sym, ctx in exec_ctxs.items():
                # 检查该股票在当前日期是否有数据
                if date not in sym_exec_dates[sym]:
                    continue
                # 更新该股票的K线索引
                sym_end_index[sym] += 1
                # 如果还在预热期内，跳过执行
                if warmup and sym_end_index[sym] <= warmup:
                    continue
                # 将该股票加入活跃上下文
                active_ctxs[sym] = ctx
                # 设置执行上下文的当前日期数据
                set_exec_ctx_data(ctx, date)
                # 检查是否到达该股票的强制退出日期
                if (
                    exit_dates
                    and sym in exit_dates
                    and date == exit_dates[sym]
                ):
                    exit_ctxs.append(ctx)  # 加入退出队列
            
            # -----------------------------------------------------------------
            # 6.2 检查今天是否有预定的订单需要执行
            # -----------------------------------------------------------------
            is_cover_sched = date in cover_sched  # 是否有空头平仓订单
            is_buy_sched = date in buy_sched  # 是否有多头开仓订单
            is_sell_sched = date in sell_sched  # 是否有卖出订单
            
            # -----------------------------------------------------------------
            # 6.3 按分数排序订单（用于仓位限制和仓位大小分配）
            # -----------------------------------------------------------------
            # 如果设置了最大多头仓位数或仓位大小处理函数，需要按分数排序
            if (
                config.max_long_positions is not None
                or pos_size_handler is not None
            ):
                if is_cover_sched:
                    # 按分数从高到低排序平仓买入订单
                    cover_sched[date].sort(key=_sort_by_score, reverse=True)
                elif is_buy_sched:
                    # 按分数从高到低排序开仓买入订单
                    buy_sched[date].sort(key=_sort_by_score, reverse=True)
            
            # 如果设置了最大空头仓位数或仓位大小处理函数，需要按分数排序卖出订单
            if is_sell_sched and (
                config.max_short_positions is not None
                or pos_size_handler is not None
            ):
                sell_sched[date].sort(key=_sort_by_score, reverse=True)
            
            # -----------------------------------------------------------------
            # 6.4 调用仓位大小处理函数（如果设置）
            # -----------------------------------------------------------------
            # 仓位大小处理函数允许用户动态调整每个订单的下单数量
            if pos_size_handler is not None and (
                is_cover_sched or is_buy_sched or is_sell_sched
            ):
                pos_size_buy_results = None
                if is_cover_sched:
                    pos_size_buy_results = cover_sched[date]
                elif is_buy_sched:
                    pos_size_buy_results = buy_sched[date]
                self._set_pos_sizes(
                    pos_size_handler=pos_size_handler,
                    pos_ctx=pos_ctx,
                    buy_results=pos_size_buy_results,
                    sell_results=sell_sched[date] if is_sell_sched else None,
                )
            # -----------------------------------------------------------------
            # 6.5 检查止损止盈条件
            # -----------------------------------------------------------------
            # 遍历所有持仓，检查是否触发止损或止盈条件
            portfolio.check_stops(date, price_scope)
            
            # -----------------------------------------------------------------
            # 6.6 执行预定的订单
            # -----------------------------------------------------------------
            # 订单执行顺序：平仓买入 -> 卖出 -> 开仓买入
            # 这个顺序确保平仓操作优先执行，释放资金后再进行新的开仓
            
            # 6.6.1 执行空头平仓买入订单
            if is_cover_sched:
                self._place_buy_orders(
                    date=date,
                    price_scope=price_scope,
                    pending_order_scope=pending_order_scope,
                    buy_sched=cover_sched,
                    portfolio=portfolio,
                    enable_fractional_shares=enable_fractional_shares,
                )
            
            # 6.6.2 执行卖出订单（多头平仓或空头开仓）
            if is_sell_sched:
                self._place_sell_orders(
                    date=date,
                    price_scope=price_scope,
                    pending_order_scope=pending_order_scope,
                    sell_sched=sell_sched,
                    portfolio=portfolio,
                    enable_fractional_shares=enable_fractional_shares,
                )
            
            # 6.6.3 执行多头开仓买入订单
            if is_buy_sched:
                self._place_buy_orders(
                    date=date,
                    price_scope=price_scope,
                    pending_order_scope=pending_order_scope,
                    buy_sched=buy_sched,
                    portfolio=portfolio,
                    enable_fractional_shares=enable_fractional_shares,
                )
            
            # -----------------------------------------------------------------
            # 6.7 记录当前K线的投资组合状态
            # -----------------------------------------------------------------
            portfolio.capture_bar(date, test_data)
            # -----------------------------------------------------------------
            # 6.8 执行用户定义的交易逻辑
            # -----------------------------------------------------------------
            
            # 6.8.1 执行before_exec回调（在所有交易逻辑之前）
            if before_exec_fn is not None and active_ctxs:
                before_exec_fn(active_ctxs)
            
            # 6.8.2 执行每个股票的交易逻辑函数
            # 这里调用用户通过add_execution添加的交易策略函数
            for sym, ctx in active_ctxs.items():
                if sym in exec_fns:
                    exec_fns[sym](ctx)  # 执行交易逻辑，可能设置ctx.buy_shares或ctx.sell_shares
            
            # 6.8.3 执行after_exec回调（在所有交易逻辑之后）
            if after_exec_fn is not None and active_ctxs:
                after_exec_fn(active_ctxs)
            
            # -----------------------------------------------------------------
            # 6.9 收集交易信号并处理滑点
            # -----------------------------------------------------------------
            for ctx in active_ctxs.values():
                # 如果设置了滑点模型且有交易信号，应用滑点
                if (
                    slippage_model
                    and not ctx._exiting_pos  # 非强制退出的情况
                    and (ctx.buy_shares or ctx.sell_shares)  # 有买入或卖出信号
                ):
                    self._apply_slippage(slippage_model, ctx)
                
                # 将执行上下文转换为执行结果
                result = ctx.to_result()
                if result is None:
                    continue  # 没有交易信号
                
                # 根据信号类型分类到不同的结果队列
                if result.buy_shares is not None:
                    if result.cover:
                        # 空头平仓买入信号
                        cover_results.append(result)
                    else:
                        # 多头开仓买入信号
                        buy_results.append(result)
                if result.sell_shares is not None:
                    # 卖出信号（多头平仓或空头开仓）
                    sell_results.append(result)
            # -----------------------------------------------------------------
            # 6.10 调度新产生的订单
            # -----------------------------------------------------------------
            # 将收集到的交易信号调度到未来的日期执行
            # 延迟由config.buy_delay和config.sell_delay控制
            
            # 6.10.1 调度空头平仓买入订单
            while cover_results:
                self._schedule_order(
                    result=cover_results.popleft(),
                    created=date,  # 信号产生日期
                    sym_end_index=sym_end_index,
                    delay=config.buy_delay,  # 买入延迟（K线数）
                    sched=cover_sched,  # 目标调度器
                    col_scope=col_scope,
                    pending_order_scope=pending_order_scope,
                )
            
            # 6.10.2 调度多头开仓买入订单
            while buy_results:
                self._schedule_order(
                    result=buy_results.popleft(),
                    created=date,
                    sym_end_index=sym_end_index,
                    delay=config.buy_delay,
                    sched=buy_sched,
                    col_scope=col_scope,
                    pending_order_scope=pending_order_scope,
                )
            
            # 6.10.3 调度卖出订单
            while sell_results:
                self._schedule_order(
                    result=sell_results.popleft(),
                    created=date,
                    sym_end_index=sym_end_index,
                    delay=config.sell_delay,  # 卖出延迟（K线数）
                    sched=sell_sched,
                    col_scope=col_scope,
                    pending_order_scope=pending_order_scope,
                )
            
            # -----------------------------------------------------------------
            # 6.11 处理强制退出仓位
            # -----------------------------------------------------------------
            # 在最后一个交易日或指定退出日期强制平仓
            while exit_ctxs:
                self._exit_position(
                    portfolio=portfolio,
                    date=date,
                    ctx=exit_ctxs.popleft(),
                    exit_cover_fill_price=config.exit_cover_fill_price,
                    exit_sell_fill_price=config.exit_sell_fill_price,
                    price_scope=price_scope,
                )
            
            # -----------------------------------------------------------------
            # 6.12 更新K线计数并记录进度
            # -----------------------------------------------------------------
            portfolio.incr_bars()  # 增加投资组合的K线计数
            # 每10个K线或最后一个K线时更新进度日志
            if i % 10 == 0 or i == len(test_dates) - 1:
                logger.backtest_executions_loading(i + 1)
        
        # =====================================================================
        # 第七步：返回信号数据（如果配置了return_signals）
        # =====================================================================
        return (
            get_signals(test_syms, col_scope, ind_scope, pred_scope)
            if config.return_signals
            else {}
        )

    def _apply_slippage(
        self,
        slippage_model: SlippageModel,
        ctx: ExecContext,
    ):
        """
        应用滑点模型。
        
        滑点模型会调整执行上下文中的成交价格，模拟真实交易中因市场冲击、
        流动性不足等原因导致的实际成交价与预期价格之间的差异。
        
        参数:
            slippage_model: 滑点模型实例
            ctx: 执行上下文，包含买卖信号信息
        """
        buy_shares = to_decimal(ctx.buy_shares) if ctx.buy_shares else None
        sell_shares = to_decimal(ctx.sell_shares) if ctx.sell_shares else None
        slippage_model.apply_slippage(
            ctx, buy_shares=buy_shares, sell_shares=sell_shares
        )

    def _exit_position(
        self,
        portfolio: Portfolio,
        date: np.datetime64,
        ctx: ExecContext,
        exit_cover_fill_price: Union[
            PriceType, Callable[[str, BarData], Union[int, float, Decimal]]
        ],
        exit_sell_fill_price: Union[
            PriceType, Callable[[str, BarData], Union[int, float, Decimal]]
        ],
        price_scope: PriceScope,
    ):
        """
        强制退出指定股票的所有仓位。
        
        当达到最后一个交易日或指定的退出日期时调用此方法，
        会同时平掉该股票的多头和空头仓位。
        
        参数:
            portfolio: 投资组合对象
            date: 退出日期
            ctx: 执行上下文
            exit_cover_fill_price: 空头平仓的成交价格类型
            exit_sell_fill_price: 多头平仓的成交价格类型
            price_scope: 价格作用域，用于获取具体价格
        """
        # 获取空头平仓的成交价格
        buy_fill_price = price_scope.fetch(ctx.symbol, exit_cover_fill_price)
        # 获取多头平仓的成交价格
        sell_fill_price = price_scope.fetch(ctx.symbol, exit_sell_fill_price)
        # 执行退出仓位操作
        portfolio.exit_position(
            date,
            ctx.symbol,
            buy_fill_price=buy_fill_price,
            sell_fill_price=sell_fill_price,
        )

    def _set_pos_sizes(
        self,
        pos_size_handler: Callable[[PosSizeContext], None],
        pos_ctx: PosSizeContext,
        buy_results: Optional[list[ExecResult]],
        sell_results: Optional[list[ExecResult]],
    ):
        """
        设置各订单的仓位大小。
        
        此方法调用用户定义的仓位大小处理函数，允许用户根据当前市场状况、
        投资组合状态等因素动态调整每个订单的下单数量。
        
        仓位大小处理函数可以实现诸如：
        - 等权重分配资金
        - 根据信号强度分配仓位
        - 基于波动率的仓位调整
        - 凯利公式等资金管理策略
        
        参数:
            pos_size_handler: 用户定义的仓位大小处理函数
            pos_ctx: 仓位大小上下文，提供给处理函数使用
            buy_results: 待处理的买入信号列表
            sell_results: 待处理的卖出信号列表
        """
        # 设置仓位大小上下文的数据
        set_pos_size_ctx_data(
            ctx=pos_ctx, buy_results=buy_results, sell_results=sell_results
        )
        # 调用用户定义的仓位大小处理函数
        pos_size_handler(pos_ctx)
        
        # 将处理函数设置的仓位大小应用到对应的执行结果
        for id, shares in pos_ctx._signal_shares.items():
            if id < 0:
                raise ValueError(f"Invalid ExecSignal id: {id}")
            if buy_results is not None and sell_results is not None:
                # 同时有买入和卖出信号的情况
                if id >= (len(buy_results) + len(sell_results)):
                    raise ValueError(f"Invalid ExecSignal id: {id}")
                if id < len(buy_results):
                    # 信号ID对应买入结果
                    buy_results[id].buy_shares = to_decimal(shares)
                else:
                    # 信号ID对应卖出结果（需要减去买入结果的数量）
                    sell_results[
                        id - len(buy_results)
                    ].sell_shares = to_decimal(shares)
            elif buy_results is not None:
                # 只有买入信号的情况
                if id >= len(buy_results):
                    raise ValueError(f"Invalid ExecSignal id: {id}")
                buy_results[id].buy_shares = to_decimal(shares)
            elif sell_results is not None:
                # 只有卖出信号的情况
                if id >= len(sell_results):
                    raise ValueError(f"Invalid ExecSignal id: {id}")
                sell_results[id].sell_shares = to_decimal(shares)
            else:
                raise ValueError(
                    "buy_results and sell_results cannot both be None."
                )

    def _schedule_order(
        self,
        result: ExecResult,
        created: np.datetime64,
        sym_end_index: Mapping[str, int],
        delay: int,
        sched: Mapping[np.datetime64, list[ExecResult]],
        col_scope: ColumnScope,
        pending_order_scope: PendingOrderScope,
    ):
        """
        将订单调度到未来的某个日期执行。
        
        在PyBroker中，交易信号产生后不会立即执行，而是会延迟一定数量的K线后执行。
        这模拟了真实交易中信号产生后需要等待下一个交易时段才能下单的情况。
        
        例如：如果buy_delay=1，今天收盘后产生的买入信号会在明天执行。
        
        参数:
            result: 执行结果，包含交易信号的详细信息
            created: 信号产生的日期
            sym_end_index: 各股票当前的K线索引
            delay: 延迟执行的K线数量
            sched: 订单调度器（按日期索引的订单列表）
            col_scope: 列作用域，用于获取日期数据
            pending_order_scope: 待处理订单作用域
        """
        # 获取当前股票的K线位置（从0开始）
        date_loc = sym_end_index[result.symbol] - 1
        # 获取该股票的所有日期
        dates = col_scope.fetch(result.symbol, DataCol.DATE.value)
        if dates is None:
            raise ValueError("Dates not found.")
        
        logger = StaticScope.instance().logger
        
        # 检查延迟后的日期是否在数据范围内
        if date_loc + delay < len(dates):
            # 计算订单执行日期
            date = dates[date_loc + delay]
            
            # 确定订单类型和参数
            order_type: Literal["buy", "sell"]
            if result.buy_shares is not None:
                order_type = "buy"
                shares = result.buy_shares
                limit_price = result.buy_limit_price
                fill_price = result.buy_fill_price
            elif result.sell_shares is not None:
                order_type = "sell"
                shares = result.sell_shares
                limit_price = result.sell_limit_price
                fill_price = result.sell_fill_price
            else:
                raise ValueError("buy_shares or sell_shares needs to be set.")
            
            # 将订单添加到待处理订单列表，获取订单ID
            result.pending_order_id = pending_order_scope.add(
                type=order_type,
                symbol=result.symbol,
                created=created,
                exec_date=date,
                shares=shares,
                limit_price=limit_price,
                fill_price=fill_price,
            )
            # 将执行结果添加到对应日期的调度列表
            sched[date].append(result)
            logger.debug_schedule_order(date, result)
        else:
            # 如果延迟后超出数据范围，订单无法调度
            logger.debug_unscheduled_order(result)

    def _place_buy_orders(
        self,
        date: np.datetime64,
        price_scope: PriceScope,
        pending_order_scope: PendingOrderScope,
        buy_sched: dict[np.datetime64, list[ExecResult]],
        portfolio: Portfolio,
        enable_fractional_shares: bool,
    ):
        """
        执行预定的买入订单。
        
        处理在指定日期调度的所有买入订单，包括：
        1. 验证订单仍然有效（未被取消）
        2. 获取成交价格
        3. 调用投资组合执行买入
        4. 设置止损止盈（如果有）
        
        参数:
            date: 执行日期
            price_scope: 价格作用域，用于获取成交价格
            pending_order_scope: 待处理订单作用域
            buy_sched: 买入订单调度器
            portfolio: 投资组合对象
            enable_fractional_shares: 是否允许分数股
        """
        buy_results = buy_sched[date]
        
        for result in buy_results:
            # 跳过没有买入数量的结果
            if result.buy_shares is None:
                continue
            # 检查订单是否仍在待处理列表中（可能被止损等机制取消）
            if (
                result.pending_order_id is None
                or not pending_order_scope.contains(result.pending_order_id)
            ):
                continue
            
            # 从待处理列表中移除该订单
            pending_order_scope.remove(result.pending_order_id)
            
            # 处理股数（是否允许分数股）
            buy_shares = self._get_shares(
                result.buy_shares, enable_fractional_shares
            )
            # 获取成交价格
            fill_price = price_scope.fetch(
                result.symbol, result.buy_fill_price
            )
            
            # 执行买入操作
            order = portfolio.buy(
                date=date,
                symbol=result.symbol,
                shares=buy_shares,
                fill_price=fill_price,
                limit_price=result.buy_limit_price,  # 限价（如有）
                stops=result.long_stops,  # 多头止损止盈设置
            )
            
            # 记录订单执行结果
            logger = StaticScope.instance().logger
            if order is None:
                # 订单未成交（可能因限价未达到或资金不足）
                logger.debug_unfilled_buy_order(
                    date=date,
                    symbol=result.symbol,
                    shares=buy_shares,
                    fill_price=fill_price,
                    limit_price=result.buy_limit_price,
                )
            else:
                # 订单成交
                logger.debug_filled_buy_order(
                    date=date,
                    symbol=result.symbol,
                    shares=buy_shares,
                    fill_price=fill_price,
                    limit_price=result.buy_limit_price,
                )
        
        # 清理该日期的调度记录
        del buy_sched[date]

    def _place_sell_orders(
        self,
        date: np.datetime64,
        price_scope: PriceScope,
        pending_order_scope: PendingOrderScope,
        sell_sched: dict[np.datetime64, list[ExecResult]],
        portfolio: Portfolio,
        enable_fractional_shares: bool,
    ):
        """
        执行预定的卖出订单。
        
        处理在指定日期调度的所有卖出订单，包括：
        1. 验证订单仍然有效（未被取消）
        2. 获取成交价格
        3. 调用投资组合执行卖出
        4. 设置空头止损止盈（如果是做空）
        
        参数:
            date: 执行日期
            price_scope: 价格作用域，用于获取成交价格
            pending_order_scope: 待处理订单作用域
            sell_sched: 卖出订单调度器
            portfolio: 投资组合对象
            enable_fractional_shares: 是否允许分数股
        """
        sell_results = sell_sched[date]
        
        for result in sell_results:
            # 跳过没有卖出数量的结果
            if result.sell_shares is None:
                continue
            # 检查订单是否仍在待处理列表中
            if (
                result.pending_order_id is None
                or not pending_order_scope.contains(result.pending_order_id)
            ):
                continue
            
            # 从待处理列表中移除该订单
            pending_order_scope.remove(result.pending_order_id)
            
            # 处理股数
            sell_shares = self._get_shares(
                result.sell_shares, enable_fractional_shares
            )
            # 获取成交价格
            fill_price = price_scope.fetch(
                result.symbol, result.sell_fill_price
            )
            
            # 执行卖出操作
            order = portfolio.sell(
                date=date,
                symbol=result.symbol,
                shares=sell_shares,
                fill_price=fill_price,
                limit_price=result.sell_limit_price,  # 限价（如有）
                stops=result.short_stops,  # 空头止损止盈设置
            )
            
            # 记录订单执行结果
            logger = StaticScope.instance().logger
            if order is None:
                # 订单未成交
                logger.debug_unfilled_sell_order(
                    date=date,
                    symbol=result.symbol,
                    shares=sell_shares,
                    fill_price=fill_price,
                    limit_price=result.sell_limit_price,
                )
            else:
                # 订单成交
                logger.debug_filled_sell_order(
                    date=date,
                    symbol=result.symbol,
                    shares=sell_shares,
                    fill_price=fill_price,
                    limit_price=result.sell_limit_price,
                )
        
        # 清理该日期的调度记录
        del sell_sched[date]

    def _get_shares(
        self,
        shares: Union[int, float, Decimal],
        enable_fractional_shares: bool,
    ) -> Decimal:
        """
        处理股数，根据配置决定是否允许分数股。
        
        参数:
            shares: 原始股数
            enable_fractional_shares: 是否允许分数股
            
        返回:
            处理后的股数（Decimal类型）
        """
        if enable_fractional_shares:
            # 允许分数股，直接转换
            return to_decimal(shares)
        else:
            # 不允许分数股，取整
            return to_decimal(int(shares))


class WalkforwardWindow(NamedTuple):
    """
    滚动分析时间窗口 - 包含训练数据和测试数据的索引。
    
    在滚动分析(Walkforward Analysis)中，历史数据被分割成多个时间窗口。
    每个窗口包含：
    - 训练数据：用于训练机器学习模型
    - 测试数据：用于回测评估模型表现
    
    这种方法确保模型始终只使用"过去"的数据进行训练，
    避免未来数据泄露(look-ahead bias)。
    
    属性:
        train_data: 训练数据的索引数组
        test_data: 测试数据的索引数组
    """

    train_data: NDArray[np.int_]  # 训练数据索引
    test_data: NDArray[np.int_]  # 测试数据索引


class WalkforwardMixin:
    """
    滚动分析混入类 - 实现滚动分析(Walkforward Analysis)的数据分割逻辑。
    
    滚动分析是一种用于评估机器学习模型在实际交易中表现的方法。
    它模拟了真实的交易场景：
    1. 使用历史数据训练模型
    2. 用训练好的模型在未来数据上进行交易
    3. 随时间推移，不断用新数据重新训练模型
    
    示例（3个窗口，train_size=0.9）::
    
        时间线: [==============================]
        
        窗口1:  [训练数据(90%)|测试(10%)]
        窗口2:           [训练数据(90%)|测试(10%)]
        窗口3:                    [训练数据(90%)|测试(10%)]
    
    这种方法的优点：
    - 避免未来数据泄露
    - 更真实地评估模型表现
    - 可以观察模型随时间的表现变化
    """

    def walkforward_split(
        self,
        df: pd.DataFrame,
        windows: int,
        lookahead: int,
        train_size: float = 0.9,
        shuffle: bool = False,
    ) -> Iterator[WalkforwardWindow]:
        r"""Splits a :class:`pandas.DataFrame` containing data for multiple
        ticker symbols into an :class:`Iterator` of train/test time windows for
        `Walkforward Analysis
        <https://www.pybroker.com/en/latest/notebooks/6.%20Training%20a%20Model.html#Walkforward-Analysis>`_.

        Args:
            df: :class:`pandas.DataFrame` of data to split into train/test
                windows for Walkforward Analysis.
            windows: Number of walkforward time windows.
            lookahead: Number of bars in the future of the target prediction.
                For example, predicting returns for the next bar would have a
                ``lookahead`` of ``1``. This quantity is needed to prevent
                training data from leaking into the test boundary.
            train_size: Amount of data in ``df`` to use for training, where
                the max ``train_size`` is ``1``. For example, a ``train_size``
                of ``0.9`` would result in 90% of data in ``df`` being used for
                training and the remaining 10% of data being used for testing.
            shuffle: Whether to randomly shuffle the data used for training.
                Defaults to ``False``.

        Returns:
            :class:`Iterator` of :class:`.WalkforwardWindow`\ s containing
            train and test data.
        """
        if windows <= 0:
            raise ValueError("windows needs to be > 0.")
        if lookahead <= 0:
            raise ValueError("lookahead needs to be > 0.")
        if train_size < 0:
            raise ValueError("train_size cannot be negative.")
        if df.empty:
            raise ValueError("DataFrame is empty.")
        date_col = DataCol.DATE.value
        dates = df[[date_col]]
        window_dates = get_unique_sorted_dates(df[date_col])
        error_msg = f"""
        Invalid params for {len(window_dates)} dates:
        windows: {windows}
        lookahead: {lookahead}
        train_size: {train_size}
        """
        if train_size == 0 or train_size == 1:
            window_length = int(len(window_dates) / windows)
            offset = len(window_dates) - window_length * windows
            for i in range(windows):
                start = offset + i * window_length
                end = start + window_length
                if train_size == 0:
                    test_idx = dates[
                        (dates[date_col] >= window_dates[start])
                        & (dates[date_col] <= window_dates[end - 1])
                    ]
                    test_idx = test_idx.index.to_numpy()
                    yield WalkforwardWindow(np.array(tuple()), test_idx)
                else:
                    train_idx = dates[
                        (dates[date_col] >= window_dates[start])
                        & (dates[date_col] <= window_dates[end - 1])
                    ]
                    train_idx = train_idx.index.to_numpy()
                    if shuffle:
                        np.random.shuffle(train_idx)
                    yield WalkforwardWindow(train_idx, np.array(tuple()))
        elif windows == 1:
            res = len(window_dates) - 1 - lookahead
            if res <= 0:
                raise ValueError(error_msg)
            train_length = int(res * train_size)
            test_length = int(res * (1 - train_size))
            train_start = (
                len(window_dates) - lookahead - train_length - test_length - 1
            )
            train_end = train_start + train_length
            test_start = train_end + lookahead
            if test_start >= len(window_dates):
                raise ValueError(error_msg)
            test_end = len(window_dates) - 1
            train_idx = dates[
                (dates[date_col] >= window_dates[train_start])
                & (dates[date_col] <= window_dates[train_end])
            ]
            test_idx = dates[
                (dates[date_col] >= window_dates[test_start])
                & (dates[date_col] <= window_dates[test_end])
            ]
            train_idx = train_idx.index.to_numpy()
            test_idx = test_idx.index.to_numpy()
            if shuffle:
                np.random.shuffle(train_idx)
            yield WalkforwardWindow(train_idx, test_idx)
        else:
            res = len(window_dates) - (lookahead - 1) * windows
            window_length = res / windows  # type: ignore[assignment]
            train_length = int(window_length * train_size)
            test_length = int(window_length * (1 - train_size))
            if train_length < 0 or test_length < 0:
                raise ValueError(error_msg)
            while True:
                rem = (res - (train_length + test_length * windows)) / windows
                train_incr = int(rem * train_size)
                test_incr = int(rem * (1 - train_size))
                if train_incr == 0 or test_incr == 0:
                    break
                train_length += train_incr
                test_length += test_incr
            if train_length == 0 and test_length == 0:
                raise ValueError(error_msg)
            window_idx = []
            for i in range(windows):
                test_end = i * test_length
                test_start = test_end + test_length
                train_end = test_start + lookahead - 1
                train_start = train_end + train_length
                window_idx.append(
                    (train_start, train_end, test_start, test_end)
                )
            window_idx.reverse()
            window_dates = window_dates[::-1]
            for train_start, train_end, test_start, test_end in window_idx:
                train_idx = dates[
                    (dates[date_col] > window_dates[train_start])
                    & (dates[date_col] <= window_dates[train_end])
                ]
                test_idx = dates[
                    (dates[date_col] > window_dates[test_start])
                    & (dates[date_col] <= window_dates[test_end])
                ]
                train_idx = train_idx.index.to_numpy()
                test_idx = test_idx.index.to_numpy()
                if shuffle:
                    np.random.shuffle(train_idx)
                yield WalkforwardWindow(train_idx, test_idx)


@dataclass(frozen=True)
class TestResult:
    r"""
    回测结果类 - 包含策略回测的所有结果数据。
    
    该类是一个不可变的数据类，包含回测完成后的所有重要信息，
    可用于分析策略表现、生成报告或进一步优化。
    
    属性说明：
    =========
    
    时间范围:
        - start_date: 回测开始日期
        - end_date: 回测结束日期
    
    投资组合数据:
        - portfolio: 每根K线的投资组合状态DataFrame，包含：
            - cash: 现金余额
            - equity: 总权益
            - market_value: 持仓市值
            - margin: 保证金
            - pnl: 已实现盈亏
            - unrealized_pnl: 未实现盈亏
            - fees: 累计手续费
    
    持仓数据:
        - positions: 每根K线的各股票持仓DataFrame，包含：
            - long_shares: 多头持仓数量
            - short_shares: 空头持仓数量
            - equity: 持仓权益
            - market_value: 持仓市值
            - unrealized_pnl: 未实现盈亏
    
    订单数据:
        - orders: 所有下单记录DataFrame，包含：
            - date: 下单日期
            - symbol: 股票代码
            - type: 订单类型（buy/sell）
            - shares: 股数
            - limit_price: 限价
            - fill_price: 成交价
            - fees: 手续费
    
    交易数据:
        - trades: 所有完成的交易DataFrame，包含：
            - entry: 入场价格
            - exit: 出场价格
            - pnl: 盈亏
            - return_pct: 收益率
            - bars: 持仓K线数
            - mae: 最大不利偏移
            - mfe: 最大有利偏移
    
    评估指标:
        - metrics: EvalMetrics对象，包含各项评估指标
        - metrics_df: 评估指标的DataFrame格式
        - bootstrap: 自助法统计结果（可选）
    
    信号数据:
        - signals: 各股票的信号数据字典（需开启return_signals配置）
        - stops: 止损止盈触发记录（需开启return_stops配置）
    """

    start_date: datetime  # 回测开始日期
    end_date: datetime  # 回测结束日期
    portfolio: pd.DataFrame  # 投资组合每日状态
    positions: pd.DataFrame  # 各股票每日持仓
    orders: pd.DataFrame  # 所有订单记录
    trades: pd.DataFrame  # 所有交易记录
    metrics: EvalMetrics  # 评估指标对象
    metrics_df: pd.DataFrame  # 评估指标DataFrame
    bootstrap: Optional[BootstrapResult]  # 自助法统计结果
    signals: Optional[dict[str, pd.DataFrame]]  # 信号数据
    stops: Optional[pd.DataFrame]  # 止损止盈记录


class Strategy(
    BacktestMixin,
    EvaluateMixin,
    IndicatorsMixin,
    ModelsMixin,
    WalkforwardMixin,
):
    """
    交易策略类 - PyBroker框架的核心类。
    
    该类整合了回测所需的所有功能，包括：
    - 数据获取和管理
    - 技术指标计算
    - 机器学习模型训练和预测
    - 回测执行和订单管理
    - 评估指标计算
    - 滚动分析（Walkforward Analysis）
    
    使用流程：
    =========
    1. 创建Strategy实例，指定数据源和日期范围
    2. 使用add_execution()添加交易逻辑
    3. 可选：设置仓位大小处理函数、滑点模型等
    4. 调用backtest()或walkforward()执行回测
    5. 分析返回的TestResult对象
    
    示例::
    
        from pybroker import Strategy, YFinance
        
        def buy_low_fn(ctx):
            # 当价格低于20日均线时买入
            if ctx.close[-1] < ctx.indicator('sma20')[-1]:
                ctx.buy_shares = 100
        
        strategy = Strategy(YFinance(), '2020-01-01', '2023-01-01')
        strategy.add_execution(buy_low_fn, ['AAPL', 'GOOGL'])
        result = strategy.backtest()
        print(result.metrics_df)
    
    参数:
        data_source: 数据源，可以是DataSource子类实例或pandas.DataFrame
        start_date: 数据开始日期（包含）
        end_date: 数据结束日期（包含）
        config: 可选的策略配置对象
    """

    _execution_id: int = 0

    def __init__(
        self,
        data_source: Union[DataSource, pd.DataFrame],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        config: Optional[StrategyConfig] = None,
    ):
        self._verify_data_source(data_source)
        self._data_source = data_source
        self._start_date = to_datetime(start_date)
        self._end_date = to_datetime(end_date)
        verify_date_range(self._start_date, self._end_date)
        if config is not None:
            self._verify_config(config)
            self._config = config
        else:
            self._config = StrategyConfig()
        self._executions: set[Execution] = set()
        self._before_exec_fn: Optional[
            Callable[[Mapping[str, ExecContext]], None]
        ] = None
        self._after_exec_fn: Optional[
            Callable[[Mapping[str, ExecContext]], None]
        ] = None
        self._pos_size_handler: Optional[Callable[[PosSizeContext], None]] = (
            None
        )
        self._slippage_model: Optional[SlippageModel] = None
        self._scope = StaticScope.instance()
        self._logger = self._scope.logger

    def _verify_config(self, config: StrategyConfig):
        if config.initial_cash <= 0:
            raise ValueError("initial_cash must be greater than 0.")
        if (
            config.max_long_positions is not None
            and config.max_long_positions <= 0
        ):
            raise ValueError("max_long_positions must be greater than 0.")
        if (
            config.max_short_positions is not None
            and config.max_short_positions <= 0
        ):
            raise ValueError("max_short_positions must be greater than 0.")
        if config.buy_delay <= 0:
            raise ValueError("buy_delay must be greater than 0.")
        if config.sell_delay <= 0:
            raise ValueError("sell_delay must be greater than 0.")
        if config.bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be greater than 0.")
        if config.bootstrap_sample_size <= 0:
            raise ValueError("bootstrap_sample_size must be greater than 0.")

    def _verify_data_source(
        self, data_source: Union[DataSource, pd.DataFrame]
    ):
        if isinstance(data_source, pd.DataFrame):
            verify_data_source_columns(data_source)
        elif not isinstance(data_source, DataSource):
            raise TypeError(f"Invalid data_source type: {type(data_source)}")

    def set_slippage_model(self, slippage_model: Optional[SlippageModel]):
        """
        设置滑点模型。
        
        滑点模型用于模拟真实交易中的滑点效应，即实际成交价格与预期价格之间的差异。
        滑点可能由以下因素导致：
        - 市场冲击：大额订单对价格的影响
        - 流动性不足：买卖价差较大
        - 延迟：信号产生到订单执行之间的时间差
        
        参数:
            slippage_model: SlippageModel实例，设为None则禁用滑点模拟
        """
        self._slippage_model = slippage_model

    def add_execution(
        self,
        fn: Optional[Callable[[ExecContext], None]],
        symbols: Union[str, Iterable[str]],
        models: Optional[Union[ModelSource, Iterable[ModelSource]]] = None,
        indicators: Optional[Union[Indicator, Iterable[Indicator]]] = None,
    ):
        r"""
        添加一个执行到回测中。
        
        执行(Execution)是交易策略的基本单元，将一组股票代码与交易逻辑函数关联起来。
        在回测过程中，每当处理到这些股票的新K线数据时，就会调用对应的交易逻辑函数。
        
        示例::
        
            def buy_on_dip(ctx):
                # 当RSI低于30时买入
                if ctx.indicator('rsi')[-1] < 30:
                    ctx.buy_shares = 100
            
            # 为AAPL和GOOGL添加这个策略
            strategy.add_execution(buy_on_dip, ['AAPL', 'GOOGL'], 
                                   indicators=[rsi_indicator])
        
        注意事项：
        - 每个股票只能属于一个执行，不能重复添加
        - 如果fn为None，可以用于仅计算指标或训练模型
        - 指标和模型必须先通过pybroker.indicator()和pybroker.model()注册
        
        参数:
            fn: 交易逻辑函数，接收ExecContext参数。在每根K线时调用。
                可以为None（仅用于计算指标/模型）。
            symbols: 应用此执行的股票代码，可以是单个字符串或字符串列表。
            models: 此执行需要使用的ModelSource对象列表。
                模型会在回测前训练（或从缓存加载）。
            indicators: 此执行需要使用的Indicator对象列表。
                指标会在回测前计算（或从缓存加载）。
        """
        symbols = (
            frozenset((symbols,))
            if isinstance(symbols, str)
            else frozenset(symbols)
        )
        if not symbols:
            raise ValueError("symbols cannot be empty.")
        for sym in symbols:
            for exec in self._executions:
                if sym in exec.symbols:
                    raise ValueError(
                        f"{sym} was already added to an execution."
                    )
        if models is not None:
            for model in (
                (models,) if isinstance(models, ModelSource) else models
            ):
                if not self._scope.has_model_source(model.name):
                    raise ValueError(
                        f"ModelSource {model.name!r} was not registered."
                    )
                if model is not self._scope.get_model_source(model.name):
                    raise ValueError(
                        f"ModelSource {model.name!r} does not match "
                        "registered ModelSource."
                    )
        model_names = (
            (
                frozenset((models.name,))
                if isinstance(models, ModelSource)
                else frozenset(model.name for model in models)
            )
            if models is not None
            else frozenset()
        )
        if indicators is not None:
            for ind in (
                (indicators,)
                if isinstance(indicators, Indicator)
                else indicators
            ):
                if not self._scope.has_indicator(ind.name):
                    raise ValueError(
                        f"Indicator {ind.name!r} was not registered."
                    )
                if ind is not self._scope.get_indicator(ind.name):
                    raise ValueError(
                        f"Indicator {ind.name!r} does not match registered "
                        "Indicator."
                    )
        ind_names = (
            (
                frozenset((indicators.name,))
                if isinstance(indicators, Indicator)
                else frozenset(ind.name for ind in indicators)
            )
            if indicators is not None
            else frozenset()
        )
        self._execution_id += 1
        self._executions.add(
            Execution(
                id=self._execution_id,
                symbols=symbols,
                fn=fn,
                model_names=model_names,
                indicator_names=ind_names,
            )
        )

    def set_before_exec(
        self, fn: Optional[Callable[[Mapping[str, ExecContext]], None]]
    ):
        r"""
        设置在所有交易逻辑执行前运行的回调函数。
        
        该函数在每根K线的所有交易逻辑函数执行前调用，可以用于：
        - 在多个股票之间进行协调决策
        - 计算跨股票的统计量
        - 设置全局状态
        
        示例::
        
            def before_all(ctxs):
                # 计算所有股票的平均收益率
                avg_return = np.mean([ctx.close[-1]/ctx.close[-2] 
                                      for ctx in ctxs.values()])
                for ctx in ctxs.values():
                    ctx.session['avg_return'] = avg_return
            
            strategy.set_before_exec(before_all)
        
        参数:
            fn: 回调函数，接收一个字典参数，键为股票代码，值为对应的ExecContext。
        """
        self._before_exec_fn = fn

    def set_after_exec(
        self, fn: Optional[Callable[[Mapping[str, ExecContext]], None]]
    ):
        r"""
        设置在所有交易逻辑执行后运行的回调函数。
        
        该函数在每根K线的所有交易逻辑函数执行后调用，可以用于：
        - 汇总所有股票的交易信号
        - 进行风险管理检查
        - 调整订单参数
        
        参数:
            fn: 回调函数，接收一个字典参数，键为股票代码，值为对应的ExecContext。
        """
        self._after_exec_fn = fn

    def clear_executions(self):
        """
        清除所有已添加的执行。
        
        在需要重新配置策略时使用，例如在参数优化循环中。
        """
        self._executions.clear()

    def set_pos_size_handler(
        self, fn: Optional[Callable[[PosSizeContext], None]]
    ):
        r"""
        设置仓位大小处理函数。
        
        仓位大小处理函数在订单执行前被调用，允许动态调整每个订单的下单数量。
        这对于实现复杂的资金管理策略非常有用，例如：
        - 等权重分配：将资金平均分配给所有信号
        - 波动率调整：根据股票波动率调整仓位
        - 凯利公式：基于历史胜率和赔率计算最优仓位
        - 风险预算：限制单个仓位的风险敞口
        
        示例::
        
            def equal_weight(ctx):
                # 将资金平均分配给所有买入信号
                n_signals = len(ctx.long_signals())
                if n_signals > 0:
                    cash_per_signal = ctx.cash / n_signals
                    for signal in ctx.long_signals():
                        shares = cash_per_signal / signal.fill_price
                        ctx.set_shares(signal, shares)
            
            strategy.set_pos_size_handler(equal_weight)
        
        参数:
            fn: 仓位大小处理函数，接收PosSizeContext参数。
        """
        self._pos_size_handler = fn

    def backtest(
        self,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        timeframe: str = "",
        between_time: Optional[tuple[str, str]] = None,
        days: Optional[Union[str, Day, Iterable[Union[str, Day]]]] = None,
        lookahead: int = 1,
        train_size: int = 0,
        shuffle: bool = False,
        calc_bootstrap: bool = False,
        disable_parallel: bool = False,
        warmup: Optional[int] = None,
        portfolio: Optional[Portfolio] = None,
        adjust: Optional[Any] = None,
    ) -> TestResult:
        """
        执行回测 - 运行通过add_execution()添加的所有交易策略。
        
        这是执行回测的主要方法。它会：
        1. 获取指定日期范围的历史数据
        2. 计算所需的技术指标
        3. 训练机器学习模型（如有）
        4. 按时间顺序遍历每根K线，执行交易逻辑
        5. 模拟订单执行、持仓管理、止损止盈
        6. 计算评估指标
        
        回测流程详解：
        =============
        
        数据准备阶段：
            - 从数据源获取历史K线数据
            - 根据timeframe参数重采样数据
            - 根据between_time和days参数过滤数据
            - 计算所有已注册的技术指标
        
        模型训练阶段（如果train_size > 0）：
            - 将数据分为训练集和测试集
            - 使用训练集训练所有已注册的模型
        
        回测执行阶段：
            - 按时间顺序遍历每根K线
            - 检查止损止盈条件
            - 执行预定的买卖订单
            - 调用用户定义的交易逻辑函数
            - 收集并调度新产生的交易信号
        
        结果汇总阶段：
            - 计算收益率、夏普比率等评估指标
            - 生成订单历史、交易历史、持仓历史
        
        参数:
            start_date: 回测开始日期（包含）。必须在Strategy初始化时指定的日期范围内。
                如果为None，使用初始化时的start_date。
            end_date: 回测结束日期（包含）。必须在Strategy初始化时指定的日期范围内。
                如果为None，使用初始化时的end_date。
            timeframe: 时间框架字符串，用于指定K线周期。支持的单位：
                - "s"/"sec": 秒
                - "m"/"min": 分钟
                - "h"/"hour": 小时
                - "d"/"day": 日
                - "w"/"week": 周
                例如："1h 30m"表示1小时30分钟。
            between_time: 时间过滤范围，格式为(开始时间, 结束时间)。
                例如：('9:30', '16:00')只保留交易时间段的数据。
            days: 交易日过滤，可以是单个字符串或字符串列表。
                例如："mon"只在周一交易，["mon", "wed", "fri"]在周一三五交易。
            lookahead: 预测目标的前瞻期（K线数）。
                例如：预测下一根K线的收益率时，lookahead=1。
                这个参数用于防止训练数据泄露到测试期。
            train_size: 用于训练的数据比例（0到1之间）。
                例如：train_size=0.9表示90%数据用于训练，10%用于测试。
                设为0表示不训练模型（纯规则策略）。
            shuffle: 是否打乱训练数据。默认False。
                注意：启用模型缓存时此选项会被禁用。
            calc_bootstrap: 是否计算自助法统计指标。默认False。
                自助法可以估计评估指标的置信区间。
            disable_parallel: 是否禁用并行计算指标。默认False。
                如果遇到多进程相关问题，可以设为True。
            warmup: 预热期K线数。在此期间不执行交易逻辑。
                用于确保技术指标有足够的历史数据。
            portfolio: 自定义投资组合对象。
                如果为None，使用默认配置创建新的Portfolio。
            adjust: 数据调整类型（用于股票分红、拆股等调整）。
        
        返回:
            TestResult对象，包含：
            - portfolio: 投资组合每日状态
            - positions: 各股票每日持仓
            - orders: 所有订单记录
            - trades: 所有交易记录
            - metrics: 评估指标
            - signals: 信号数据（如果配置了return_signals）
        """
        return self.walkforward(
            windows=1,
            lookahead=lookahead,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            between_time=between_time,
            days=days,
            train_size=train_size,
            shuffle=shuffle,
            calc_bootstrap=calc_bootstrap,
            disable_parallel=disable_parallel,
            warmup=warmup,
            portfolio=portfolio,
            adjust=adjust,
        )

    def walkforward(
        self,
        windows: int,
        lookahead: int = 1,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        timeframe: str = "",
        between_time: Optional[tuple[str, str]] = None,
        days: Optional[Union[str, Day, Iterable[Union[str, Day]]]] = None,
        train_size: float = 0.5,
        shuffle: bool = False,
        calc_bootstrap: bool = False,
        disable_parallel: bool = False,
        warmup: Optional[int] = None,
        portfolio: Optional[Portfolio] = None,
        adjust: Optional[Any] = None,
    ) -> TestResult:
        """Backtests the trading strategy using `Walkforward Analysis
        <https://www.pybroker.com/en/latest/notebooks/6.%20Training%20a%20Model.html#Walkforward-Analysis>`_.
        Backtesting data supplied by the :class:`pybroker.data.DataSource` is
        divided into ``windows`` number of equal sized time windows, with each
        window split into train and test data as specified by ``train_size``.
        The backtest "walks forward" in time through each window, running
        executions that were added with :meth:`.add_execution`.

        Args:
            windows: Number of walkforward time windows.
            start_date: Starting date of the Walkforward Analysis (inclusive).
                Must be within ``start_date`` and ``end_date`` range that was
                passed to :meth:`.__init__`.
            end_date: Ending date of the Walkforward Analysis (inclusive). Must
                be within ``start_date`` and ``end_date`` range that was passed
                to :meth:`.__init__`.
            timeframe: Formatted string that specifies the timeframe
                resolution of the backtesting data. The timeframe string
                supports the following units:

                - ``"s"``/``"sec"``: seconds
                - ``"m"``/``"min"``: minutes
                - ``"h"``/``"hour"``: hours
                - ``"d"``/``"day"``: days
                - ``"w"``/``"week"``: weeks

                An example timeframe string is ``1h 30m``.
            between_time: ``tuple[str, str]`` of times of day e.g.
                ('9:30', '16:00') used to filter the backtesting data
                (inclusive).
            days: Days (e.g. ``"mon"``, ``"tues"`` etc.) used to filter the
                backtesting data.
            lookahead: Number of bars in the future of the target prediction.
                For example, predicting returns for the next bar would have a
                ``lookahead`` of ``1``. This quantity is needed to prevent
                training data from leaking into the test boundary.
            train_size: Amount of :class:`pybroker.data.DataSource` data to use
                for training, where the max ``train_size`` is ``1``. For
                example, a ``train_size`` of ``0.9`` would result in 90% of
                data being used for training and the remaining 10% of data
                being used for testing.
            shuffle: Whether to randomly shuffle the data used for training.
                Defaults to ``False``. Disabled when model caching is enabled
                via :meth:`pybroker.cache.enable_model_cache`.
            calc_bootstrap: Whether to compute randomized bootstrap evaluation
                metrics. Defaults to ``False``.
            disable_parallel: If ``True``,
                :class:`pybroker.indicator.Indicator` data is computed
                serially. If ``False``, :class:`pybroker.indicator.Indicator`
                data is computed in parallel using multiple processes.
                Defaults to ``False``.
            warmup: Number of bars that need to pass before running the
                executions.
            portfolio: Custom :class:`pybroker.portfolio.Portfolio` to use for
                backtests.
            adjust: The type of adjustment to make to the
                :class:`pybroker.data.DataSource`.

        Returns:
            :class:`.TestResult` containing portfolio balances, order
            history, and evaluation metrics.
        """
        if warmup is not None and warmup < 1:
            raise ValueError("warmup must be > 0.")
        scope = StaticScope.instance()
        try:
            scope.freeze_data_cols()
            if not self._executions:
                raise ValueError("No executions were added.")
            start_dt = (
                self._start_date
                if start_date is None
                else to_datetime(start_date)
            )
            if start_dt < self._start_date or start_dt > self._end_date:
                raise ValueError(
                    f"start_date must be between {self._start_date} and "
                    f"{self._end_date}."
                )
            end_dt = (
                self._end_date if end_date is None else to_datetime(end_date)
            )
            if end_dt < self._start_date or end_dt > self._end_date:
                raise ValueError(
                    f"end_date must be between {self._start_date} and "
                    f"{self._end_date}."
                )
            if start_dt is not None and end_dt is not None:
                verify_date_range(start_dt, end_dt)
            self._logger.walkforward_start(start_dt, end_dt)
            df = self._fetch_data(timeframe, adjust)
            day_ids = self._to_day_ids(days)
            df = self._filter_dates(
                df=df,
                start_date=start_dt,
                end_date=end_dt,
                between_time=between_time,
                days=day_ids,
            )
            tf_seconds = to_seconds(timeframe)
            indicator_data = self._fetch_indicators(
                df=df,
                cache_date_fields=CacheDateFields(
                    start_date=start_dt,
                    end_date=end_dt,
                    tf_seconds=tf_seconds,
                    between_time=between_time,
                    days=day_ids,
                ),
                disable_parallel=disable_parallel,
            )
            train_only = (
                self._before_exec_fn is None
                and self._after_exec_fn is None
                and all(map(lambda e: e.fn is None, self._executions))
            )
            if portfolio is None:
                portfolio = Portfolio(
                    self._config.initial_cash,
                    self._config.fee_mode,
                    self._config.fee_amount,
                    self._config.subtract_fees,
                    self._fractional_shares_enabled(),
                    self._config.position_mode,
                    self._config.max_long_positions,
                    self._config.max_short_positions,
                    self._config.return_stops,
                )
            signals = self._run_walkforward(
                portfolio=portfolio,
                df=df,
                indicator_data=indicator_data,
                tf_seconds=tf_seconds,
                between_time=between_time,
                days=day_ids,
                windows=windows,
                lookahead=lookahead,
                train_size=train_size,
                shuffle=shuffle,
                train_only=train_only,
                warmup=warmup,
            )
            if train_only:
                self._logger.walkforward_completed()
            return self._to_test_result(
                start_dt,
                end_dt,
                portfolio,
                calc_bootstrap,
                train_only,
                signals if self._config.return_signals else None,
            )
        finally:
            scope.unfreeze_data_cols()

    def _to_day_ids(
        self, days: Optional[Union[str, Day, Iterable[Union[str, Day]]]]
    ) -> Optional[tuple[int]]:
        if days is None:
            return None
        days = (
            (days,) if isinstance(days, str) or isinstance(days, Day) else days
        )
        return tuple(
            sorted(
                (day.value if isinstance(day, Day) else Day[day.upper()].value)  # type: ignore[union-attr]
                for day in set(days)  # type: ignore[arg-type]
            )
        )  # type: ignore[return-value]

    def _fractional_shares_enabled(self):
        return self._config.enable_fractional_shares or isinstance(
            self._data_source, AlpacaCrypto
        )

    def _run_walkforward(
        self,
        portfolio: Portfolio,
        df: pd.DataFrame,
        indicator_data: dict[IndicatorSymbol, pd.Series],
        tf_seconds: int,
        between_time: Optional[tuple[str, str]],
        days: Optional[tuple[int]],
        windows: int,
        lookahead: int,
        train_size: float,
        shuffle: bool,
        train_only: bool,
        warmup: Optional[int],
    ) -> dict[str, pd.DataFrame]:
        """
        执行滚动分析（Walkforward Analysis）的核心逻辑。
        
        滚动分析将数据分成多个时间窗口，每个窗口包含训练集和测试集。
        在每个窗口中：
        1. 使用训练集训练模型
        2. 使用测试集执行回测
        
        这种方法可以更真实地模拟实盘交易中模型的表现，避免未来数据泄露。
        
        参数:
            portfolio: 投资组合对象
            df: 完整的回测数据
            indicator_data: 预计算的指标数据
            tf_seconds: 时间框架（秒）
            between_time: 时间过滤范围
            days: 交易日过滤
            windows: 滚动窗口数量
            lookahead: 预测目标的前瞻期
            train_size: 训练集比例
            shuffle: 是否打乱训练数据
            train_only: 是否仅训练模式
            warmup: 预热期K线数
            
        返回:
            各股票的信号数据框字典
        """
        # 初始化会话数据（用于在K线之间保持用户自定义状态）
        sessions: dict[str, dict] = defaultdict(dict)
        
        # 设置各股票的退出日期（如果配置了在最后一根K线退出）
        exit_dates: dict[str, np.datetime64] = {}
        if self._config.exit_on_last_bar:
            for exec in self._executions:
                for sym in exec.symbols:
                    sym_dates = df[df[DataCol.SYMBOL.value] == sym][
                        DataCol.DATE.value
                    ].values
                    if len(sym_dates):
                        sym_dates.sort()
                        exit_dates[sym] = sym_dates[-1]
        
        # 存储所有窗口的信号数据
        signals: dict[str, pd.DataFrame] = {}
        
        # 遍历每个滚动窗口
        for train_idx, test_idx in self.walkforward_split(
            df=df,
            windows=windows,
            lookahead=lookahead,
            train_size=train_size,
            shuffle=shuffle,
        ):
            models: dict[ModelSymbol, TrainedModel] = {}
            train_data = df.loc[train_idx]  # 当前窗口的训练数据
            test_data = df.loc[test_idx]  # 当前窗口的测试数据
            
            # 如果有训练数据，训练模型
            if not train_data.empty:
                # 确定需要训练的模型-股票对
                model_syms = {
                    ModelSymbol(model_name, sym)
                    for sym in train_data[DataCol.SYMBOL.value].unique()
                    for execution in self._executions
                    for model_name in execution.model_names
                    if sym in execution.symbols
                }
                train_dates = get_unique_sorted_dates(
                    train_data[DataCol.DATE.value]
                )
                # 训练模型
                models = self.train_models(
                    model_syms=model_syms,
                    train_data=train_data,
                    test_data=test_data,
                    indicator_data=indicator_data,
                    cache_date_fields=CacheDateFields(
                        start_date=to_datetime(train_dates[0]),
                        end_date=to_datetime(train_dates[-1]),
                        tf_seconds=tf_seconds,
                        between_time=between_time,
                        days=days,
                    ),
                )
            
            # 如果没有测试数据，返回当前信号
            if test_data.empty:
                return signals
            
            # 执行当前窗口的回测
            split_signals = self.backtest_executions(
                config=self._config,
                executions=self._executions,
                before_exec_fn=self._before_exec_fn,
                after_exec_fn=self._after_exec_fn,
                sessions=sessions,
                models=models,
                indicator_data=indicator_data,
                test_data=test_data,
                portfolio=portfolio,
                pos_size_handler=self._pos_size_handler,
                exit_dates=exit_dates,
                train_only=train_only,
                slippage_model=self._slippage_model,
                enable_fractional_shares=self._fractional_shares_enabled(),
                round_fill_price=self._config.round_fill_price,
                warmup=warmup,
            )
            
            # 合并当前窗口的信号数据
            for sym, signals_df in split_signals.items():
                if sym in signals:
                    signals[sym] = pd.concat([signals[sym], signals_df])
                else:
                    signals[sym] = signals_df
        
        return signals

    def _filter_dates(
        self,
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        between_time: Optional[tuple[str, str]],
        days: Optional[tuple[int]],
    ) -> pd.DataFrame:
        if start_date != self._start_date or end_date != self._end_date:
            df = _between(df, start_date, end_date).reset_index(drop=True)
        if df[DataCol.DATE.value].dt.tz is not None:
            # Fixes bug on Windows.
            # https://stackoverflow.com/questions/51827582/message-exception-ignored-when-dealing-pandas-datetime-type
            df[DataCol.DATE.value] = df[DataCol.DATE.value].dt.tz_convert(None)
        is_time_range = between_time is not None or days is not None
        if is_time_range:
            df = df.reset_index(drop=True).set_index(DataCol.DATE.value)
        if days is not None:
            self._logger.info_walkforward_on_days(days)
            df = df[df.index.weekday.isin(frozenset(days))]
        if between_time is not None:
            if len(between_time) != 2:
                raise ValueError(
                    "between_time must be a tuple[str, str] of start time and"
                    f" end time, received {between_time!r}."
                )
            self._logger.info_walkforward_between_time(between_time)
            df = df.between_time(*between_time)
        if is_time_range:
            df = df.reset_index()
        return df

    def _fetch_indicators(
        self,
        df: pd.DataFrame,
        cache_date_fields: CacheDateFields,
        disable_parallel: bool,
    ) -> dict[IndicatorSymbol, pd.Series]:
        indicator_syms = set()
        for execution in self._executions:
            for sym in execution.symbols:
                for model_name in execution.model_names:
                    ind_names = self._scope.get_indicator_names(model_name)
                    for ind_name in ind_names:
                        indicator_syms.add(IndicatorSymbol(ind_name, sym))
                for ind_name in execution.indicator_names:
                    indicator_syms.add(IndicatorSymbol(ind_name, sym))
        return self.compute_indicators(
            df=df,
            indicator_syms=indicator_syms,
            cache_date_fields=cache_date_fields,
            disable_parallel=disable_parallel,
        )

    def _fetch_data(
        self, timeframe: str, adjust: Optional[Any]
    ) -> pd.DataFrame:
        unique_syms = {
            sym for execution in self._executions for sym in execution.symbols
        }
        if isinstance(self._data_source, DataSource):
            df = self._data_source.query(
                unique_syms,
                self._start_date,
                self._end_date,
                timeframe,
                adjust,
            )
        else:
            df = _between(self._data_source, self._start_date, self._end_date)
            df = df[df[DataCol.SYMBOL.value].isin(unique_syms)]
        if df.empty:
            raise ValueError("DataSource is empty.")
        return df.reset_index(drop=True)

    def _to_test_result(
        self,
        start_date: datetime,
        end_date: datetime,
        portfolio: Portfolio,
        calc_bootstrap: bool,
        train_only: bool,
        signals: Optional[dict[str, pd.DataFrame]],
    ) -> TestResult:
        """
        将回测过程中收集的数据转换为TestResult对象。
        
        该方法在回测完成后调用，负责：
        1. 将投资组合的K线记录转换为DataFrame
        2. 将持仓记录转换为DataFrame
        3. 将订单记录转换为DataFrame
        4. 将交易记录转换为DataFrame
        5. 计算评估指标（收益率、夏普比率等）
        6. 可选地计算自助法统计
        7. 处理数据精度（四舍五入）
        
        参数:
            start_date: 回测开始日期
            end_date: 回测结束日期
            portfolio: 投资组合对象，包含所有回测记录
            calc_bootstrap: 是否计算自助法统计
            train_only: 是否仅训练模式
            signals: 信号数据字典
            
        返回:
            包含完整回测结果的TestResult对象
        """
        if train_only:
            return TestResult(
                start_date=start_date,
                end_date=end_date,
                portfolio=pd.DataFrame(),
                positions=pd.DataFrame(),
                orders=pd.DataFrame(),
                trades=pd.DataFrame(),
                metrics=EvalMetrics(),
                metrics_df=pd.DataFrame(),
                bootstrap=None,
                signals=signals,
                stops=None,
            )
        pos_df = pd.DataFrame.from_records(
            portfolio.position_bars, columns=PositionBar._fields
        )
        for col in (
            "close",
            "equity",
            "market_value",
            "margin",
            "unrealized_pnl",
        ):
            pos_df[col] = quantize(pos_df, col, self._config.round_test_result)
        pos_df.set_index(["symbol", "date"], inplace=True)
        portfolio_df = pd.DataFrame.from_records(
            portfolio.bars, columns=PortfolioBar._fields, index="date"
        )
        for col in (
            "cash",
            "equity",
            "margin",
            "market_value",
            "pnl",
            "unrealized_pnl",
            "fees",
        ):
            portfolio_df[col] = quantize(
                portfolio_df, col, self._config.round_test_result
            )
        orders_df = pd.DataFrame.from_records(
            portfolio.orders, columns=Order._fields, index="id"
        )
        for col in ("limit_price", "fill_price", "fees"):
            orders_df[col] = quantize(
                orders_df, col, self._config.round_test_result
            )
        trades_df = pd.DataFrame.from_records(
            portfolio.trades, columns=Trade._fields, index="id"
        )
        trades_df["bars"] = trades_df["bars"].astype(int)
        for col in (
            "entry",
            "exit",
            "pnl",
            "return_pct",
            "agg_pnl",
            "pnl_per_bar",
            "mae",
            "mfe",
        ):
            trades_df[col] = quantize(
                trades_df, col, self._config.round_test_result
            )
        shares_type = float if self._fractional_shares_enabled() else int
        pos_df["long_shares"] = pos_df["long_shares"].astype(shares_type)
        pos_df["short_shares"] = pos_df["short_shares"].astype(shares_type)
        orders_df["shares"] = orders_df["shares"].astype(shares_type)
        trades_df["shares"] = trades_df["shares"].astype(shares_type)
        eval_result = self.evaluate(
            portfolio_df=portfolio_df,
            trades_df=trades_df,
            calc_bootstrap=calc_bootstrap,
            bootstrap_sample_size=self._config.bootstrap_sample_size,
            bootstrap_samples=self._config.bootstrap_samples,
            bars_per_year=self._config.bars_per_year,
        )
        metrics = [
            (k, v)
            for k, v in dataclasses.asdict(eval_result.metrics).items()
            if v is not None
        ]
        metrics_df = pd.DataFrame(metrics, columns=["name", "value"])
        stops_df = None
        if self._config.return_stops:
            stops_df = pd.DataFrame.from_records(
                portfolio._stop_records, columns=StopRecord._fields
            )
        self._logger.walkforward_completed()
        return TestResult(
            start_date=start_date,
            end_date=end_date,
            portfolio=portfolio_df,
            positions=pos_df,
            orders=orders_df,
            trades=trades_df,
            metrics=eval_result.metrics,
            metrics_df=metrics_df,
            bootstrap=eval_result.bootstrap,
            signals=signals,
            stops=stops_df,
        )
