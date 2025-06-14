"""Implements slippage models."""  # 实现滑点模型

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import random
from pybroker.context import ExecContext
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Optional


class SlippageModel(ABC):
    """滑点模型的基类"""

    @abstractmethod
    def apply_slippage(
        self,
        ctx: ExecContext,  # 执行上下文
        buy_shares: Optional[Decimal] = None,  # 买入股数
        sell_shares: Optional[Decimal] = None,  # 卖出股数
    ):
        """对ctx应用滑点"""


class RandomSlippageModel(SlippageModel):
    """实现一个简单的随机滑点模型
    
    参数:
        min_pct: 最小滑点百分比
        max_pct: 最大滑点百分比
    """

    def __init__(self, min_pct: float, max_pct: float):
        if min_pct < 0 or min_pct > 100:
            raise ValueError(r"min_pct must be between 0% and 100%.")
        if max_pct < 0 or max_pct > 100:
            raise ValueError(r"max_pct must be between 0% and 100%.")
        if min_pct >= max_pct:
            raise ValueError("min_pct must be < max_pct.")
        self.min_pct = min_pct / 100.0  # 最小滑点百分比
        self.max_pct = max_pct / 100.0  # 最大滑点百分比

    def apply_slippage(
        self,
        ctx: ExecContext,  # 执行上下文
        buy_shares: Optional[Decimal] = None,  # 买入股数
        sell_shares: Optional[Decimal] = None,  # 卖出股数
    ):
        """对买卖订单应用随机滑点
        
        通过减少实际成交的股数来模拟滑点的影响，模拟真实市场中的流动性影响
        """
        if buy_shares or sell_shares:
            slippage_pct = Decimal(random.uniform(self.min_pct, self.max_pct))  # 随机生成滑点百分比
            if buy_shares:
                ctx.buy_shares = buy_shares - slippage_pct * buy_shares  # 应用滑点到买入股数
            if sell_shares:
                ctx.sell_shares = sell_shares - slippage_pct * sell_shares  # 应用滑点到卖出股数

# 该模块提供了PyBroker的滑点模型功能，用于在回测中模拟真实市场中的滑点影响。
# SlippageModel是所有滑点模型的基类，定义了应用滑点的接口。
# RandomSlippageModel实现了一个简单的随机滑点模型，通过随机减少实际成交股数来模拟滑点。
# 滑点模型可以通过Strategy的set_slippage_model方法设置，使回测更接近真实市场环境。
