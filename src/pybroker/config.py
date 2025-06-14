"""Contains configuration options."""  # 包含配置选项

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

from pybroker.common import BarData, FeeInfo, FeeMode, PositionMode, PriceType
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable, Optional, Union


@dataclass(frozen=True)
class StrategyConfig:
    """策略的配置选项
    
    属性:
        initial_cash: 策略起始现金
        fee_mode: 计算经纪费用的模式。支持以下选项:
            - ORDER_PERCENT: 费用是订单金额的百分比
            - PER_ORDER: 费用是每笔订单的固定金额
            - PER_SHARE: 费用是订单中每股的固定金额
            - Callable[[FeeInfo], Decimal]: 使用自定义函数计算费用
            - None: 禁用费用(默认)
        fee_amount: 经纪费用金额
        subtract_fees: 是否在订单成交后从现金余额中减去费用。默认为False
        enable_fractional_shares: 是否启用分数股交易。对加密货币交易设置为True。默认为False
        round_fill_price: 是否将成交价格四舍五入到最接近的分。默认为True
        position_mode: 策略的仓位模式。支持以下选项:
            - DEFAULT: 多头和空头仓位
            - LONG_ONLY: 仅多头仓位
            - SHORT_ONLY: 仅空头仓位
        max_long_positions: Portfolio中任何时候可以持有的最大多头仓位数量。
                          为None时不限制。默认为None
        max_short_positions: Portfolio中任何时候可以持有的最大空头仓位数量。
                          为None时不限制。默认为None
        buy_delay: 买入信号后下单前等待的K线数量。默认值为1表示在下一个K线下单。
                 必须大于0
        sell_delay: 卖出信号后下单前等待的K线数量。默认值为1表示在下一个K线下单。
                  必须大于0
        bootstrap_samples: 用于计算引导指标的样本数量。默认为10,000
        bootstrap_sample_size: 用于计算引导指标的随机样本大小。默认为1,000
        exit_on_last_bar: 是否在股票代码的最后一个可用K线自动退出任何未平仓仓位。
                        默认为False
        exit_cover_fill_price: 当exit_on_last_bar为True时，平仓空头仓位的成交价格。
                             默认为MIDDLE价格类型
        exit_sell_fill_price: 当exit_on_last_bar为True时，平仓多头仓位的成交价格。
                            默认为MIDDLE价格类型
        bars_per_year: 用于年化评估指标的年观察次数。例如，值为252将用于年化
                     日回报的夏普比率
        return_signals: 当为True时，K线数据、指标数据和模型预测将与测试结果一起返回。
                      默认为False
        return_stops: 当为True时，止损值将与测试结果一起返回。默认为False
        round_test_result: 当为True时，将测试结果中的值四舍五入到最接近的分。
                         默认为True
    """

    initial_cash: float = field(default=100_000)  # 初始现金
    fee_mode: Optional[Union[FeeMode, Callable[[FeeInfo], Decimal]]] = field(
        default=None
    )  # 费用模式
    fee_amount: float = field(default=0)  # 费用金额
    subtract_fees: bool = field(default=False)  # 是否减去费用
    enable_fractional_shares: bool = field(default=False)  # 是否启用分数股
    round_fill_price: bool = field(default=True)  # 是否四舍五入成交价
    position_mode: PositionMode = field(default=PositionMode.DEFAULT)  # 仓位模式
    max_long_positions: Optional[int] = field(default=None)  # 最大多头仓位数量
    max_short_positions: Optional[int] = field(default=None)  # 最大空头仓位数量
    buy_delay: int = field(default=1)  # 买入延迟
    sell_delay: int = field(default=1)  # 卖出延迟
    bootstrap_samples: int = field(default=10_000)  # 引导样本数量
    bootstrap_sample_size: int = field(default=1_000)  # 引导样本大小
    exit_on_last_bar: bool = field(default=False)  # 是否在最后K线退出
    exit_cover_fill_price: Union[
        PriceType, Callable[[str, BarData], Union[int, float, Decimal]]
    ] = field(default=PriceType.MIDDLE)  # 退出平仓空头的成交价
    exit_sell_fill_price: Union[
        PriceType, Callable[[str, BarData], Union[int, float, Decimal]]
    ] = field(default=PriceType.MIDDLE)  # 退出平仓多头的成交价
    bars_per_year: Optional[int] = field(default=None)  # 每年K线数量
    return_signals: bool = field(default=False)  # 是否返回信号
    return_stops: bool = field(default=False)  # 是否返回止损
    round_test_result: bool = field(default=True)  # 是否四舍五入测试结果

# 该模块提供了PyBroker的配置功能，主要包含了StrategyConfig类，用于配置策略的各种参数。
# 配置选项涵盖了资金管理、费用模型、仓位控制、订单执行和报告生成等方面。
# 通过这些配置，用户可以灵活调整回测环境以模拟不同的交易场景和规则。
# 大多数配置项都有合理的默认值，使新用户能够快速开始使用，同时为高级用户提供了自定义能力。
