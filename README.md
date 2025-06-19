# PyBroker

## 中文注释分支

这是PyBroker的中文注释分支，所有代码文件都已添加了详细的中文注释，包括：

- 类、方法和属性的中文说明
- 函数参数和返回值的中文解释
- 重要算法和逻辑的中文注释
- 保留原始英文注释，同时添加对应的中文翻译

### 已添加中文注释的文件

1. **核心模块**
   - `__init__.py` - 模块导入和初始化
   - `strategy.py` - 策略和回测功能
   - `common.py` - 基础数据结构和工具函数
   - `config.py` - 策略配置选项

2. **数据处理模块**
   - `data.py` - 数据获取和处理
   - `cache.py` - 数据缓存机制
   - `scope.py` - 作用域管理
   - `context.py` - 执行上下文

3. **分析模块**
   - `indicator.py` - 技术指标计算
   - `model.py` - 机器学习模型
   - `eval.py` - 策略评估和指标计算
   - `vect.py` - 向量化操作

4. **交易模块**
   - `portfolio.py` - 投资组合管理
   - `slippage.py` - 滑点模型
   - `log.py` - 日志记录

### 注释内容特点

1. **代码结构说明**
   - 模块的整体功能和用途
   - 类之间的继承和依赖关系
   - 重要数据结构的定义和用途

2. **算法实现说明**
   - 技术指标的计算方法
   - 机器学习模型的训练和预测流程
   - 回测引擎的核心算法

3. **接口文档**
   - 函数参数的类型和含义
   - 返回值的格式和用途
   - 异常情况的处理说明

4. **使用示例**
   - 关键功能的使用方法
   - 常见场景的代码示例
   - 最佳实践建议

### 核心概念解析

本节将深入解析 PyBroker 中的一些核心设计理念和关键组件，帮助您更好地理解和使用该框架。

#### **仓位管理: `PosSizeContext` 与 `pos_size_handler`**

在 `pybroker` 中，仓位管理是一个核心功能，它通过 `PosSizeContext` 和 `pos_size_handler` 的协同工作来实现。这种设计将"交易信号的生成"与"仓位大小的决策"清晰地分离开来，提供了极大的灵活性。

1.  **`PosSizeContext` (仓位大小上下文)**
    -   **是什么**：一个包含决策仓位大小时所需全部信息的"数据快照"或上下文对象。
    -   **作用**：
        -   **汇集信号**：当策略为多个股票生成买卖信号后，`PosSizeContext` 会将这些信号 (`ExecSignal`) 全部收集起来。
        -   **提供全局信息**：您可以从中获取投资组合的当前现金、所有持仓、历史交易、指标数据等全局信息。
        -   **提供修改接口**：通过 `ctx.set_shares()` 方法，您可以在处理器中为每个信号动态设置最终的交易股数。

2.  **`pos_size_handler` (仓位大小处理器)**
    -   **是什么**：一个由用户自定义的、用于实现具体仓位管理逻辑的函数。您通过 `Strategy.set_pos_size_handler()` 将其注册到策略中。
    -   **作用**：接收 `PosSizeContext` 对象，并根据您的自定义逻辑（如风险平价、波动率加权等）来计算每个交易信号应分配的最终股数。

3.  **协同工作流程与用途**
    -   **解耦设计**：策略执行函数只负责在"正确的时间"对"正确的股票"发出"买/卖"的初步意图；而 `pos_size_handler` 则从全局视角，负责根据所有信号来统一进行资金分配和风险管理。
    -   **执行时机**：`handler` 在策略生成所有初步信号之后、但在真正下单之前被调用。
    -   **强大用途**：这种机制使得实现高级资金管理策略成为可能，例如：
        -   基于波动率倒数加权，为低风险股票分配更多资金。
        -   实现风险平价（Risk Parity）模型。
        -   基于信号得分进行排名加权。
        -   确保所有交易的总风险敞口在预设范围内。

通过理解并善用这一机制，您可以构建出更加复杂和稳健的量化交易策略。

#### **回测引擎工作流程**

`pybroker` 的回测引擎 (`Strategy.walkforward`) 经过精心设计，将复杂的回测过程分解为四个清晰的阶段，确保了数据的准确处理和模拟的真实性。

1.  **阶段一：数据准备与预处理**
    *   **目标**：为回测准备所有必需的数据。
    *   **流程**：
        1.  **获取原始数据**：从指定的数据源（如 `Alpaca`）拉取原始OHLCV数据。
        2.  **数据过滤**：根据时间段（`between_time`）、交易日（`days`）等条件筛选K线。
        3.  **指标计算**：分析所有策略和模型，**一次性、并行地**计算所需的全部技术指标，并进行缓存。

2.  **阶段二：模型训练与滑动窗口**
    *   **目标**：使用历史数据训练模型，并以滚动方式向前推进回测窗口。
    *   **流程**：
        1.  **划分窗口**：将整个数据集划分为多个"训练集"和"测试集"对（即Walk-forward窗口）。
        2.  **模型训练**：在每个窗口的**训练集**上训练所有机器学习模型，确保模型仅使用过去的数据。
        3.  **执行回测**：在对应的**测试集**上运行核心的交易模拟。

3.  **阶段三：核心模拟循环**
    *   **目标**：逐根K线模拟真实的交易行为。
    *   **流程**（在每个交易日循环）：
        1.  **更新上下文**：更新 `ExecContext`，为策略逻辑准备好当前K线的数据。
        2.  **仓位决策**：调用 `pos_size_handler`（如果已设置），从全局视角决定最终的交易股数。
        3.  **检查止损**：检查并执行所有被当前K线价格触发的止损/止盈单。
        4.  **执行订单**：处理所有**计划在今天执行**的买卖订单，更新投资组合状态（现金、持仓等）。
        5.  **记录快照**：记录当前K线收盘后投资组合的权益、市值等状态。
        6.  **执行策略**：调用用户定义的策略函数 `fn(ctx)`，产生新的**交易意图**。
        7.  **调度新单**：将新的交易意图根据延迟（`buy_delay`/`sell_delay`）安排到未来的执行计划中。

4.  **阶段四：结果汇总与评估**
    *   **目标**：将模拟过程中的所有数据整理成报告。
    *   **流程**：
        1.  **数据转换**：将记录的交易、订单、投资组合历史等原始数据转换为 `pandas.DataFrame`。
        2.  **指标评估**：计算夏普比率、最大回撤、索提诺比率等一系列性能评估指标。
        3.  **打包返回**：将所有结果打包成一个 `TestResult` 对象返回，方便用户进行分析。

## 文档

- [数据源入门](https://www.pybroker.com/en/latest/notebooks/1.%20Getting%20Started%20with%20Data%20Sources.html)
- [回测策略](https://www.pybroker.com/en/latest/notebooks/2.%20Backtesting%20a%20Strategy.html)
- [使用Bootstrap指标评估](https://www.pybroker.com/en/latest/notebooks/3.%20Evaluating%20with%20Bootstrap%20Metrics.html)
- [排名和仓位管理](https://www.pybroker.com/en/latest/notebooks/4.%20Ranking%20and%20Position%20Sizing.html)
- [编写指标](https://www.pybroker.com/en/latest/notebooks/5.%20Writing%20Indicators.html)
- [训练模型](https://www.pybroker.com/en/latest/notebooks/6.%20Training%20a%20Model.html)
- [创建自定义数据源](https://www.pybroker.com/en/latest/notebooks/7.%20Creating%20a%20Custom%20Data%20Source.html)
- [应用止损](https://www.pybroker.com/en/latest/notebooks/8.%20Applying%20Stops.html)
- [重新平衡仓位](https://www.pybroker.com/en/latest/notebooks/9.%20Rebalancing%20Positions.html)
- [轮动交易](https://www.pybroker.com/en/latest/notebooks/10.%20Rotational%20Trading.html)
- [常见问题](https://www.pybroker.com/en/latest/notebooks/FAQs.html)

## 在线文档

[完整参考文档托管在 **www.pybroker.com**](https://www.pybroker.com)

(中文用户：[中文文档](https://www.pybroker.com/zh_CN/latest/)，由 [Albert King](https://github.com/albertandking) 提供。)

