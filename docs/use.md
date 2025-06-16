# PyBroker 使用指南

PyBroker是一个Python算法交易库，提供回测、指标计算、模型训练和策略评估等功能。本文档提供了各种使用PyBroker的示例，帮助您快速上手。

## 目录

1. [安装](#安装)
2. [数据获取](#数据获取)
3. [创建指标](#创建指标)
4. [开发策略](#开发策略)
5. [回测策略](#回测策略)
6. [评估结果](#评估结果)
7. [模型训练](#模型训练)
8. [仓位管理](#仓位管理)
9. [实际交易应用](#实际交易应用)

## 安装

使用pip安装PyBroker：

```python
pip install pybroker
```

## 数据获取

### 使用YFinance数据源

```python
import pybroker as pb
from datetime import date

# 创建YFinance数据源
data_source = pb.YFinance(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start=date(2020, 1, 1),
    end=date(2023, 1, 1)
)

# 加载数据
data = data_source.load()
```

### 使用Alpaca数据源

```python
import pybroker as pb
from datetime import date

# 创建Alpaca数据源
data_source = pb.Alpaca(
    api_key='YOUR_API_KEY',
    api_secret='YOUR_API_SECRET',
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start=date(2020, 1, 1),
    end=date(2023, 1, 1)
)

# 加载数据
data = data_source.load()
```

### 使用自定义数据源

```python
import pybroker as pb
import pandas as pd
from datetime import date

class CustomDataSource(pb.data.DataSource):
    def _fetch_data(self, symbols, start_date, end_date, timeframe):
        # 实现从您自己的数据源获取数据的逻辑
        # 返回一个包含所需数据的DataFrame
        data_dict = {}
        for symbol in symbols:
            # 假设我们从CSV文件加载数据
            df = pd.read_csv(f'data/{symbol}.csv')
            # 转换日期并筛选数据
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            data_dict[symbol] = df
        
        return data_dict

# 使用自定义数据源
custom_source = CustomDataSource(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start=date(2020, 1, 1),
    end=date(2023, 1, 1)
)
data = custom_source.load()
```

## 创建指标

### 基本技术指标

```python
import pybroker as pb
import pandas as pd
import numpy as np

# 使用装饰器创建简单移动平均线(SMA)指标
@pb.indicator
def sma(series, window=20):
    """计算简单移动平均线"""
    return pd.Series(series).rolling(window).mean().values

# 使用装饰器创建相对强弱指数(RSI)指标
@pb.indicator
def rsi(close, window=14):
    """计算相对强弱指数"""
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    rsi_values = 100 - (100 / (1 + rs))
    return rsi_values.values
    
# 使用装饰器创建布林带指标
@pb.indicator
def bollinger_bands(close, window=20, num_std=2):
    """计算布林带(上轨、中轨、下轨)"""
    rolling_mean = pd.Series(close).rolling(window).mean()
    rolling_std = pd.Series(close).rolling(window).std()
    
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    
    return np.column_stack((upper_band.values, rolling_mean.values, lower_band.values))
```

### 组合使用多个指标

```python
import pybroker as pb

# 创建一个指标集
indicator_set = pb.IndicatorSet()

# 添加多个指标
indicator_set.add('sma_20', sma, window=20)
indicator_set.add('sma_50', sma, window=50)
indicator_set.add('rsi_14', rsi, window=14)
indicator_set.add('bb_20', bollinger_bands, window=20, num_std=2)

# 在策略中使用指标集
strategy = pb.Strategy(indicators=indicator_set)
```

## 开发策略

### 简单的均线交叉策略

```python
import pybroker as pb

# 创建指标
@pb.indicator
def sma(series, window):
    return pb.pd.Series(series).rolling(window).mean().values

# 定义策略
def init_strategy():
    strategy = pb.Strategy()
    
    # 添加指标
    strategy.add_indicator('sma_short', sma, window=20)
    strategy.add_indicator('sma_long', sma, window=50)
    
    # 定义买入条件：短期均线上穿长期均线
    def buy_signal(ctx):
        sma_short = ctx.indicator('sma_short')
        sma_long = ctx.indicator('sma_long')
        
        # 检查前一天和当天的均线关系，确定是否有上穿信号
        if ctx.previous:
            prev_short = ctx.previous.indicator('sma_short')
            prev_long = ctx.previous.indicator('sma_long')
            
            if prev_short < prev_long and sma_short > sma_long:
                # 上穿信号，买入
                return ctx.symbol_equity * 0.1  # 使用10%的资金
        
        return 0
    
    # 定义卖出条件：短期均线下穿长期均线
    def sell_signal(ctx):
        if not ctx.position:
            return False
            
        sma_short = ctx.indicator('sma_short')
        sma_long = ctx.indicator('sma_long')
        
        if ctx.previous:
            prev_short = ctx.previous.indicator('sma_short')
            prev_long = ctx.previous.indicator('sma_long')
            
            if prev_short > prev_long and sma_short < sma_long:
                # 下穿信号，卖出
                return True
        
        return False
    
    # 设置策略信号
    strategy.add_rule('buy', buy_signal)
    strategy.add_rule('sell', sell_signal)
    
    return strategy

# 创建策略实例
strategy = init_strategy()
```

### 基于RSI的交易策略

```python
import pybroker as pb

# 定义RSI指标
@pb.indicator
def rsi(close, window=14):
    delta = pb.pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    rsi_values = 100 - (100 / (1 + rs))
    return rsi_values.values

# 定义策略
def init_rsi_strategy():
    strategy = pb.Strategy()
    
    # 添加RSI指标
    strategy.add_indicator('rsi_14', rsi, window=14)
    
    # 买入信号：RSI低于30，超卖
    def buy_signal(ctx):
        rsi_value = ctx.indicator('rsi_14')
        
        if rsi_value < 30:
            return ctx.symbol_equity * 0.1  # 使用10%的资金
        
        return 0
    
    # 卖出信号：RSI高于70，超买
    def sell_signal(ctx):
        if not ctx.position:
            return False
            
        rsi_value = ctx.indicator('rsi_14')
        
        if rsi_value > 70:
            return True  # 卖出全部持仓
        
        return False
    
    # 设置策略信号
    strategy.add_rule('buy', buy_signal)
    strategy.add_rule('sell', sell_signal)
    
    return strategy

# 创建策略实例
rsi_strategy = init_rsi_strategy()
```

## 回测策略

```python
import pybroker as pb
from datetime import date

# 创建数据源
data_source = pb.YFinance(
    symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
    start=date(2018, 1, 1),
    end=date(2023, 1, 1)
)

# 初始化策略
strategy = init_strategy()  # 使用前面定义的策略函数

# 配置回测参数
config = pb.StrategyConfig(
    initial_cash=100000,  # 初始资金
    fee_mode=pb.FeeMode.PERCENTAGE,  # 手续费模式
    fee_amount=0.001,  # 手续费金额（0.1%）
    enable_fractional_shares=True,  # 允许购买零股
    buy_delay=1,  # 买入信号生效延迟
    sell_delay=1  # 卖出信号生效延迟
)

# 运行回测
result = pb.Strategy.backtest(
    strategy=strategy,
    data_source=data_source,
    config=config
)

# 打印回测结果
print(f"总收益率: {result.metrics.total_return:.2%}")
print(f"年化收益率: {result.metrics.cagr:.2%}")
print(f"最大回撤: {result.metrics.max_drawdown:.2%}")
print(f"夏普比率: {result.metrics.sharpe_ratio:.2f}")
print(f"索提诺比率: {result.metrics.sortino_ratio:.2f}")
```

## 评估结果

### 基本评估指标

```python
# 接上面的回测结果
metrics = result.metrics

# 查看详细的评估指标
print(f"总收益率: {metrics.total_return:.2%}")
print(f"年化收益率: {metrics.cagr:.2%}")
print(f"最大回撤: {metrics.max_drawdown:.2%}")
print(f"夏普比率: {metrics.sharpe_ratio:.2f}")
print(f"索提诺比率: {metrics.sortino_ratio:.2f}")
print(f"胜率: {metrics.win_rate:.2%}")
print(f"盈亏比: {metrics.profit_factor:.2f}")
print(f"平均每笔收益率: {metrics.avg_trade_return:.2%}")
print(f"交易次数: {metrics.num_trades}")
```

### 使用Bootstrap进行策略稳健性评估

```python
import pybroker as pb

# 假设已经有了回测结果
result = ...  # 前面的回测结果

# 使用Bootstrap方法进行策略稳健性评估
bootstrap_result = pb.eval.run_bootstrap(
    test_result=result,
    samples=1000,  # 样本数量
    sample_pct=0.5,  # 每个样本占总样本的比例
    random_state=42  # 随机种子
)

# 输出Bootstrap评估结果
print(f"平均总收益率: {bootstrap_result.mean('total_return'):.2%}")
print(f"总收益率95%置信区间: [{bootstrap_result.percentile('total_return', 2.5):.2%}, {bootstrap_result.percentile('total_return', 97.5):.2%}]")
print(f"平均夏普比率: {bootstrap_result.mean('sharpe_ratio'):.2f}")
print(f"夏普比率95%置信区间: [{bootstrap_result.percentile('sharpe_ratio', 2.5):.2f}, {bootstrap_result.percentile('sharpe_ratio', 97.5):.2f}]")
```

## 模型训练

### 训练简单的机器学习模型

```python
import pybroker as pb
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 自定义特征创建函数
def create_features(data, symbol, start_date, end_date):
    df = data.loc[symbol].copy()
    
    # 添加基本特征
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['rsi_14'] = compute_rsi(df['close'], 14)  # 假设我们有一个计算RSI的函数
    df['volatility'] = df['close'].rolling(20).std()
    df['returns_5'] = df['close'].pct_change(5)
    
    # 创建目标变量：未来5天的价格是否上涨
    df['target'] = (df['close'].shift(-5) > df['close']).astype(int)
    
    # 移除缺失值
    df = df.dropna()
    
    # 选择特征和目标变量
    features = ['sma_20', 'sma_50', 'rsi_14', 'volatility', 'returns_5']
    X = df[features].values
    y = df['target'].values
    
    return X, y, df.index

# 定义模型源
@pb.model
def random_forest_model(data, symbol, start_date, end_date):
    # 创建特征和目标变量
    X, y, index = create_features(data, symbol, start_date, end_date)
    
    # 创建并训练模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # 返回模型和预测函数
    def predict_fn(data, symbol, start_date, end_date):
        # 为预测创建特征
        X_pred, _, index_pred = create_features(data, symbol, start_date, end_date)
        # 获取预测概率
        proba = model.predict_proba(X_pred)[:, 1]  # 取正类的概率
        return pd.Series(proba, index=index_pred)
    
    return predict_fn

# 创建模型训练器
model_trainer = pb.ModelTrainer(
    model_source=random_forest_model,  # 我们的模型源
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start_date=date(2018, 1, 1),
    end_date=date(2022, 1, 1),
    data_source=data_source,  # 假设前面已经定义了data_source
    train_size=0.7  # 使用70%的数据作为训练集
)

# 训练模型
trained_models = model_trainer.train_models()

# 创建模型加载器
model_loader = pb.ModelLoader(trained_models)

# 在策略中使用模型
def model_strategy():
    strategy = pb.Strategy(model_loader=model_loader)
    
    # 定义买入条件：模型预测概率大于0.7就买入
    def buy_signal(ctx):
        if ctx.model_prediction and ctx.model_prediction > 0.7:
            return ctx.symbol_equity * 0.1  # 使用10%的资金
        return 0
    
    # 定义卖出条件：模型预测概率小于0.3就卖出
    def sell_signal(ctx):
        if ctx.position and ctx.model_prediction and ctx.model_prediction < 0.3:
            return True
        return False
    
    # 设置策略信号
    strategy.add_rule('buy', buy_signal)
    strategy.add_rule('sell', sell_signal)
    
    return strategy

# 创建基于模型的策略
ml_strategy = model_strategy()

# 使用不同的时间段进行回测
test_result = pb.Strategy.backtest(
    strategy=ml_strategy,
    data_source=pb.YFinance(
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        start=date(2022, 1, 1),  # 注意这是训练集之后的数据
        end=date(2023, 1, 1)
    ),
    config=pb.StrategyConfig(initial_cash=100000)
)
```

## 仓位管理

### 使用百分比仓位管理

```python
import pybroker as pb

# 创建策略
def position_sizing_strategy():
    strategy = pb.Strategy()
    
    # 添加指标...
    strategy.add_indicator('rsi_14', rsi, window=14)
    
    # 定义买入信号并使用百分比仓位管理
    def buy_signal(ctx):
        rsi_value = ctx.indicator('rsi_14')
        
        if rsi_value < 30:  # 超卖信号
            # 根据RSI值的强度动态调整仓位大小
            position_pct = (30 - rsi_value) / 30  # RSI越低，仓位越大
            return ctx.symbol_equity * position_pct  # 根据百分比分配资金
        
        return 0
    
    # 定义卖出信号
    def sell_signal(ctx):
        if not ctx.position:
            return False
            
        rsi_value = ctx.indicator('rsi_14')
        
        if rsi_value > 70:  # 超买信号
            return True
        
        return False
    
    # 设置策略信号
    strategy.add_rule('buy', buy_signal)
    strategy.add_rule('sell', sell_signal)
    
    return strategy
```

### 使用止损和止盈

```python
import pybroker as pb

def stop_loss_strategy():
    strategy = pb.Strategy()
    
    # 添加指标...
    
    # 定义买入信号
    def buy_signal(ctx):
        # 买入逻辑...
        if buy_condition:
            return ctx.symbol_equity * 0.1
        return 0
    
    # 定义卖出信号，包括止损和止盈
    def sell_signal(ctx):
        if not ctx.position:
            return False
        
        # 获取当前持仓信息
        entry_price = ctx.position.entry_price
        current_price = ctx.bar_data.close
        
        # 计算当前收益率
        pnl_pct = (current_price - entry_price) / entry_price
        
        # 止损条件：亏损超过5%
        if pnl_pct < -0.05:
            return True
        
        # 止盈条件：盈利超过20%
        if pnl_pct > 0.20:
            return True
        
        # 其他卖出条件...
        
        return False
    
    # 设置策略信号
    strategy.add_rule('buy', buy_signal)
    strategy.add_rule('sell', sell_signal)
    
    return strategy
```

### 跟踪止损

```python
import pybroker as pb

def trailing_stop_strategy():
    strategy = pb.Strategy()
    
    # 保存每个交易的最高价格
    highest_prices = {}
    
    # 定义买入信号
    def buy_signal(ctx):
        # 买入逻辑...
        if buy_condition:
            # 记录初始价格
            highest_prices[ctx.symbol] = ctx.bar_data.close
            return ctx.symbol_equity * 0.1
        return 0
    
    # 定义卖出信号，包括跟踪止损
    def sell_signal(ctx):
        if not ctx.position:
            return False
        
        # 获取当前价格
        current_price = ctx.bar_data.close
        
        # 更新该股票的最高价格
        if ctx.symbol in highest_prices:
            if current_price > highest_prices[ctx.symbol]:
                highest_prices[ctx.symbol] = current_price
        
        # 跟踪止损：如果从最高点下跌超过10%，则卖出
        if ctx.symbol in highest_prices:
            highest_price = highest_prices[ctx.symbol]
            if current_price < highest_price * 0.9:  # 10%的跟踪止损
                # 重置最高价格
                highest_prices[ctx.symbol] = None
                return True
        
        # 其他卖出条件...
        
        return False
    
    # 设置策略信号
    strategy.add_rule('buy', buy_signal)
    strategy.add_rule('sell', sell_signal)
    
    return strategy
```

## 实际交易应用

### 集成到交易系统

```python
import pybroker as pb
import time
from datetime import datetime, timedelta

# 假设已经有了训练好的模型和策略
model_loader = ...  # 之前训练的模型
strategy = ...  # 之前定义的策略

# 创建实时数据源
live_data_source = pb.Alpaca(
    api_key='YOUR_API_KEY',
    api_secret='YOUR_API_SECRET',
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start=datetime.now() - timedelta(days=100),  # 获取最近100天的数据
    end=datetime.now()
)

# 加载最新数据
latest_data = live_data_source.load()

# 执行策略，获取今日交易信号
signals = pb.Strategy.run(
    strategy=strategy,
    data=latest_data,
    model_loader=model_loader
)

# 处理交易信号
for symbol, signal in signals.items():
    if signal.buy_amount > 0:
        print(f"买入信号: 购买 {symbol}，金额: ${signal.buy_amount:.2f}")
        # 这里可以集成到您的交易API，例如：
        # alpaca_api.submit_order(symbol=symbol, qty=signal.buy_amount/current_price, side='buy')
    
    if signal.sell:
        print(f"卖出信号: 卖出 {symbol} 的全部持仓")
        # 这里可以集成到您的交易API，例如：
        # alpaca_api.submit_order(symbol=symbol, qty=current_position_size, side='sell')
```

### 定时执行策略

```python
import pybroker as pb
import schedule
import time
from datetime import datetime, timedelta

# 定义每日执行的函数
def run_daily_strategy():
    print(f"开始执行策略: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 获取最新数据
        live_data_source = pb.Alpaca(
            api_key='YOUR_API_KEY',
            api_secret='YOUR_API_SECRET',
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            start=datetime.now() - timedelta(days=100),
            end=datetime.now()
        )
        
        latest_data = live_data_source.load()
        
        # 执行策略
        signals = pb.Strategy.run(
            strategy=strategy,
            data=latest_data,
            model_loader=model_loader
        )
        
        # 执行交易
        for symbol, signal in signals.items():
            if signal.buy_amount > 0:
                print(f"买入信号: 购买 {symbol}，金额: ${signal.buy_amount:.2f}")
                # 集成到交易API
            
            if signal.sell:
                print(f"卖出信号: 卖出 {symbol} 的全部持仓")
                # 集成到交易API
    
    except Exception as e:
        print(f"策略执行错误: {str(e)}")

# 调度每天上午9:30运行策略（美国东部时间市场开盘时间）
schedule.every().day.at("09:30").do(run_daily_strategy)

# 保持程序运行
print("策略调度器已启动...")
while True:
    schedule.run_pending()
    time.sleep(60)  # 每分钟检查一次
```

### 监控和日志记录

```python
import pybroker as pb
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_log.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("trading_strategy")

# 启用PyBroker日志
pb.enable_logging()

# 创建日志装饰器
def log_execution(func):
    def wrapper(*args, **kwargs):
        logger.info(f"开始执行 {func.__name__}")
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            logger.info(f"{func.__name__} 执行成功，耗时：{datetime.now() - start_time}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} 执行失败：{str(e)}")
            raise
    return wrapper

# 使用日志装饰器记录交易执行
@log_execution
def execute_trades(signals):
    for symbol, signal in signals.items():
        if signal.buy_amount > 0:
            logger.info(f"买入信号: {symbol}, 金额: ${signal.buy_amount:.2f}")
            # 执行买入...
        
        if signal.sell:
            logger.info(f"卖出信号: {symbol}")
            # 执行卖出...
    
    return True
```

### 集成风险管理

```python
import pybroker as pb

# 定义风险管理函数
def risk_manager(strategy):
    """为策略添加风险管理规则"""
    original_buy_rule = strategy.rules['buy']
    
    # 创建新的买入规则，包含风险控制
    def risk_controlled_buy(ctx):
        # 首先检查市场条件
        if is_market_risky():  # 假设我们有这个函数来判断市场风险
            return 0  # 市场风险过高时不买入
        
        # 检查投资组合风险
        portfolio_risk = get_portfolio_risk(ctx)  # 假设的函数
        if portfolio_risk > 0.2:  # 如果组合风险超过20%
            return 0  # 不买入
        
        # 检查单个股票集中度
        if ctx.symbol_value / ctx.portfolio_value > 0.2:
            return 0  # 单一股票不超过组合的20%
        
        # 原始买入逻辑
        buy_amount = original_buy_rule(ctx)
        
        # 根据风险水平调整买入量
        adjusted_amount = buy_amount * (1 - portfolio_risk)
        
        return adjusted_amount
    
    # 替换原有的买入规则
    strategy.rules['buy'] = risk_controlled_buy
    
    return strategy

# 使用风险管理
strategy = init_strategy()  # 获取原始策略
risk_managed_strategy = risk_manager(strategy)  # 应用风险管理
```

---

本文档提供了PyBroker库的各种使用示例，从基本的数据获取到复杂的策略开发和回测。您可以根据自己的需求选择适合的示例，并根据实际情况进行修改。更多详细信息，请参考PyBroker的官方文档。
