"""包含向量化工具函数。"""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import numpy as np
from numba import njit  # 使用numba进行即时编译加速
from numpy.typing import NDArray
from typing import Literal


@njit
def _verify_input(array: NDArray[np.float64], n: int):
    """验证输入参数的有效性
    
    Args:
        array: numpy数组数据
        n: 周期长度
    """
    assert n > 0, "n需要大于等于1"
    assert n <= len(array), "n不能大于数组长度"


@njit
def lowv(array: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    """计算数组中每个n周期的最低值。

    Args:
        array: numpy数组数据
        n: 周期长度

    Returns:
        numpy数组,包含每个n周期的最低值
    """
    if not len(array):
        return np.array(tuple())
    _verify_input(array, n)
    out_len = len(array)
    out = np.array([np.nan for _ in range(out_len)])
    for i in range(n, out_len + 1):
        out[i - 1] = np.min(array[i - n : i])
    return out


@njit
def highv(array: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    """计算数组中每个n周期的最高值。

    Args:
        array: numpy数组数据
        n: 周期长度

    Returns:
        numpy数组,包含每个n周期的最高值
    """
    if not len(array):
        return np.array(tuple())
    _verify_input(array, n)
    out_len = len(array)
    out = np.array([np.nan for _ in range(out_len)])
    for i in range(n, out_len + 1):
        out[i - 1] = np.max(array[i - n : i])
    return out


@njit
def sumv(array: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    """计算数组中每个n周期的和。

    Args:
        array: numpy数组数据
        n: 周期长度

    Returns:
        numpy数组,包含每个n周期的和
    """
    if not len(array):
        return np.array(tuple())
    _verify_input(array, n)
    out_len = len(array)
    out = np.array([np.nan for _ in range(out_len)])
    for i in range(n, out_len + 1):
        out[i - 1] = np.sum(array[i - n : i])
    return out


@njit
def returnv(array: NDArray[np.float64], n: int = 1) -> NDArray[np.float64]:
    """计算收益率。

    Args:
        array: numpy数组数据
        n: 收益率周期,默认为1

    Returns:
        numpy数组,包含计算的收益率
    """
    if not len(array):
        return np.array(tuple())
    _verify_input(array, n)
    out_len = len(array)
    out = np.array([np.nan for _ in range(out_len)])
    for i in range(n, out_len):
        out[i] = (array[i] - array[i - n]) / array[i - n]
    return out


@njit
def cross(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.bool_]:
    """检查a是否上穿b。

    Args:
        a: numpy数组数据
        b: numpy数组数据

    Returns:
        numpy布尔数组,当a上穿b时值为1,否则为0
    """
    assert len(a), "a不能为空"
    assert len(b), "b不能为空"
    assert len(a) == len(b), "a和b的长度必须相同"
    assert len(a) >= 2, "a和b的长度必须大于等于2"
    crossed = np.where(a > b, 1, 0)
    return (sumv(crossed > 0, 2) == 1) * crossed


@njit
def normal_cdf(z: float) -> float:
    """计算标准正态分布的累积分布函数(CDF)。

    Args:
        z: 标准正态分布的z值

    Returns:
        对应的累积概率值
    """
    zz = np.fabs(z)
    pdf = np.exp(-0.5 * zz * zz) / np.sqrt(2 * np.pi)
    t = 1 / (1 + zz * 0.2316419)
    poly = (
        (((1.330274429 * t - 1.821255978) * t + 1.781477937) * t - 0.356563782)
        * t
        + 0.319381530
    ) * t
    return 1 - pdf * poly if z > 0 else pdf * poly


@njit
def inverse_normal_cdf(p: float) -> float:
    """计算标准正态分布的反累积分布函数。

    Args:
        p: 概率值,范围[0,1]

    Returns:
        对应的z值
    """
    pp = p if p <= 0.5 else 1 - p
    if pp == 0:
        pp = 1.0e-10
    t = np.sqrt(np.log(1 / (pp * pp)))
    numer = (0.010328 * t + 0.802853) * t + 2.515517
    denom = ((0.001308 * t + 0.189269) * t + 1.432788) * t + 1
    x = t - numer / denom
    return -x if p <= 0.5 else x


@njit
def _atr(
    last_bar: int,
    lookback: int,
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    use_log: bool = False,
) -> float:
    """计算平均真实范围(Average True Range)。

    Args:
        last_bar: ATR计算的最后一根K线索引
        lookback: 回看的K线数量
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        use_log: 是否使用对数转换,默认为False

    Returns:
        计算得到的ATR值
    """
    assert last_bar >= lookback
    if lookback == 0:
        if use_log:
            return np.log(high[last_bar] / low[last_bar])
        else:
            return high[last_bar] - low[last_bar]
    total = 0.0
    for i in range(last_bar - lookback + 1, last_bar + 1):
        if use_log:
            term = high[i] / low[i]
            if high[i] / close[i - 1] > term:
                term = high[i] / close[i - 1]
            if close[i - 1] / low[i] > term:
                term = close[i - 1] / low[i]
            total += np.log(term)
        else:
            term = high[i] - low[i]
            if high[i] - close[i - 1] > term:
                term = high[i] - close[i - 1]
            if close[i - 1] - low[i] > term:
                term = close[i - 1] - low[i]
            total += term
    return total / lookback


@njit
def _variance(
    use_change: bool, last_bar: int, length: int, prices: NDArray[np.float64]
) -> float:
    """计算价格变化的方差。

    Args:
        use_change: 是否使用价格变化率
        last_bar: 计算的最后一根K线索引
        length: 计算周期长度
        prices: 价格数组

    Returns:
        计算得到的方差值
    """
    if use_change:
        assert last_bar >= length
    else:
        assert last_bar >= length - 1
    total = 0.0
    for i in range(last_bar - length + 1, last_bar + 1):
        if use_change:
            term = np.log(prices[i] / prices[i - 1])
        else:
            term = np.log(prices[i])
        total += term
    mean = total / length
    total = 0.0
    for i in range(last_bar - length + 1, last_bar + 1):
        if use_change:
            term = np.log(prices[i] / prices[i - 1]) - mean
        else:
            term = np.log(prices[i]) - mean
        total += term * term
    return total / length


@njit
def detrended_rsi(
    values: NDArray[np.float64],
    short_length: int,
    long_length: int,
    reg_length: int,
) -> NDArray[np.float64]:
    """计算去趋势相对强弱指标(RSI)。

    Args:
        values: 输入数据数组
        short_length: 短期RSI的回看周期
        long_length: 长期RSI的回看周期
        reg_length: 用于线性回归的K线数量

    Returns:
        numpy数组,包含计算的去趋势RSI值
    """
    assert short_length > 0
    assert short_length <= long_length
    assert short_length > 1
    assert long_length > 1
    assert reg_length >= 1
    n = len(values)
    front_bad = long_length + reg_length - 1
    output = np.zeros(n)
    if front_bad >= n:
        return output
    work1 = np.zeros(n)
    for i in range(short_length):
        work1[i] = 1.0e90
    up_sum = dn_sum = 1.0e-60
    for i in range(1, short_length):
        diff = values[i] - values[i - 1]
        if diff > 0.0:
            up_sum += diff
        else:
            dn_sum -= diff
    up_sum /= short_length - 1
    dn_sum /= short_length - 1
    for i in range(short_length, n):
        diff = values[i] - values[i - 1]
        if diff > 0:
            up_sum = ((short_length - 1.0) * up_sum + diff) / short_length
            dn_sum *= (short_length - 1.0) / short_length
        else:
            dn_sum = ((short_length - 1.0) * dn_sum - diff) / short_length
            up_sum *= (short_length - 1.0) / short_length
        work1[i] = 100.0 * up_sum / (up_sum + dn_sum)
        if short_length == 2:
            work1[i] = -10.0 * np.log(
                2.0 / (1 + 0.00999 * (2 * work1[i] - 100)) - 1
            )
    work2 = np.zeros(n)
    for i in range(long_length):
        work2[i] = -1.0e90
    up_sum = dn_sum = 1.0e-60
    for i in range(1, long_length):
        diff = values[i] - values[i - 1]
        if diff > 0.0:
            up_sum += diff
        else:
            dn_sum -= diff
    up_sum /= long_length - 1
    dn_sum /= long_length - 1
    for i in range(long_length, n):
        diff = values[i] - values[i - 1]
        if diff > 0.0:
            up_sum = ((long_length - 1.0) * up_sum + diff) / long_length
            dn_sum *= (long_length - 1.0) / long_length
        else:
            dn_sum = ((long_length - 1.0) * dn_sum - diff) / long_length
            up_sum *= (long_length - 1.0) / long_length
        work2[i] = 100.0 * up_sum / (up_sum + dn_sum)
    for i in range(front_bad, n):
        x_mean = y_mean = 0.0
        for j in range(reg_length):
            k = i - j
            x_mean += work2[k]
            y_mean += work1[k]
        x_mean /= reg_length
        y_mean /= reg_length
        xss = xy = 0.0
        for j in range(reg_length):
            k = i - j
            x_diff = work2[k] - x_mean
            y_diff = work1[k] - y_mean
            xss += x_diff * x_diff
            xy += x_diff * y_diff
        coef = xy / (xss + 1.0e-60)
        x_diff = work2[i] - x_mean
        y_diff = work1[i] - y_mean
        output[i] = y_diff - coef * x_diff
    return output


@njit
def macd(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    short_length: int,
    long_length: int,
    smoothing: float = 0.0,
    scale: float = 1.0,
) -> NDArray[np.float64]:
    """计算移动平均收敛发散指标(MACD)。

    Args:
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        short_length: 短期回看周期
        long_length: 长期回看周期
        smoothing: 如果>=2则计算MACD减去平滑值
        scale: 返回值压缩比例,>1.0增加压缩,<1.0减少压缩,默认为1.0

    Returns:
        numpy数组,包含计算的MACD值,范围[-50, 50]
    """
    assert len(high) == len(low) and len(high) == len(close)
    assert short_length > 0
    assert short_length <= long_length
    assert smoothing >= 0
    assert scale > 0
    n = len(close)
    output = np.zeros(n)
    long_alpha = 2.0 / (long_length + 1.0)
    short_alpha = 2.0 / (short_length + 1.0)
    long_sum = short_sum = close[0]
    for i in range(1, n):
        long_sum = long_alpha * close[i] + (1.0 - long_alpha) * long_sum
        short_sum = short_alpha * close[i] + (1.0 - short_alpha) * short_sum
        diff = 0.5 * (long_length - 1.0)
        diff -= 0.5 * (short_length - 1.0)
        denom = np.sqrt(np.fabs(diff))
        k = long_length + smoothing
        if k > i:
            k = i
        denom *= _atr(i, k, high, low, close, False)
        output[i] = (short_sum - long_sum) / (denom + 1.0e-15)
        output[i] = 100.0 * normal_cdf(scale * output[i]) - 50.0
    if smoothing > 1:
        alpha = 2.0 / (smoothing + 1.0)
        smoothed = output[0]
        for i in range(1, n):
            smoothed = alpha * output[i] + (1.0 - alpha) * smoothed
            output[i] -= smoothed
    return output


@njit
def stochastic(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    lookback: int,
    smoothing: int = 0,
) -> NDArray[np.float64]:
    """计算随机指标(KDJ)。

    Args:
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        lookback: 回看K线数量
        smoothing: 对原始随机指标进行平滑的次数,可以是0,1或2次,默认为0

    Returns:
        numpy数组,包含计算的随机指标值
    """
    assert len(high) == len(low) and len(high) == len(close)
    assert lookback > 0
    assert smoothing == 0 or smoothing == 1 or smoothing == 2
    n = len(close)
    front_bad = lookback - 1
    if front_bad > n:
        front_bad = n
    output = np.zeros(n)
    for i in range(front_bad, n):
        min_val = 1.0e60
        max_val = -1.0e60
        for j in range(lookback):
            if high[i - j] > max_val:
                max_val = high[i - j]
            if low[i - j] < min_val:
                min_val = low[i - j]
        sto_0 = (close[i] - min_val) / (max_val - min_val + 1.0e-60)
        if smoothing == 0:
            output[i] = 100.0 * sto_0 - 50
        else:
            if i == front_bad:
                sto_1 = sto_0
                output[i] = 100.0 * sto_0 - 50
            else:
                sto_1 = 0.33333333 * sto_0 + 0.66666667 * sto_1
                if smoothing == 1:
                    output[i] = 100.0 * sto_1 - 50
                else:
                    if i == front_bad + 1:
                        sto_2 = sto_1
                        output[i] = 100.0 * sto_1 - 50
                    else:
                        sto_2 = 0.33333333 * sto_1 + 0.66666667 * sto_2
                        output[i] = 100.0 * sto_2 - 50
    return output


@njit
def stochastic_rsi(
    values: NDArray[np.float64],
    rsi_lookback: int,
    sto_lookback: int,
    smoothing: float = 0.0,
) -> NDArray[np.float64]:
    """计算随机相对强弱指标(Stochastic RSI)。

    Args:
        values: 输入数据数组
        rsi_lookback: RSI计算的回看周期
        sto_lookback: 随机指标计算的回看周期
        smoothing: 平滑系数,<=1表示不平滑,默认为0.0

    Returns:
        numpy数组,包含计算的随机RSI值
    """
    assert rsi_lookback > 0
    assert sto_lookback > 0
    assert smoothing >= 0
    n = len(values)
    front_bad = rsi_lookback + sto_lookback - 1
    if front_bad > n:
        front_bad = n
    output = np.zeros(n)
    if rsi_lookback >= n:
        return output
    for i in range(front_bad):
        output[i] = 0
    up_sum = dn_sum = 1.0e-60
    for i in range(1, rsi_lookback):
        diff = values[i] - values[i - 1]
        if diff > 0.0:
            up_sum += diff
        else:
            dn_sum -= diff
    up_sum /= rsi_lookback - 1
    dn_sum /= rsi_lookback - 1
    work1 = np.zeros(n)
    for i in range(rsi_lookback, n):
        diff = values[i] - values[i - 1]
        if diff > 0.0:
            up_sum = ((rsi_lookback - 1) * up_sum + diff) / rsi_lookback
            dn_sum *= (rsi_lookback - 1.0) / rsi_lookback
        else:
            dn_sum = ((rsi_lookback - 1) * dn_sum - diff) / rsi_lookback
            up_sum *= (rsi_lookback - 1.0) / rsi_lookback
        work1[i] = 100.0 * up_sum / (up_sum + dn_sum)
    for i in range(front_bad, n):
        min_val = 1.0e60
        max_val = -1.0e60
        for j in range(sto_lookback):
            if work1[i - j] > max_val:
                max_val = work1[i - j]
            if work1[i - j] < min_val:
                min_val = work1[i - j]
        output[i] = (
            100.0 * (work1[i] - min_val) / (max_val - min_val + 1.0e-60) - 50.0
        )
    if smoothing > 1:
        alpha = 2.0 / (smoothing + 1.0)
        smoothed = output[front_bad]
        for i in range(front_bad + 1, n):
            smoothed = alpha * output[i] + (1.0 - alpha) * smoothed
            output[i] = smoothed
    return output


@njit
def _legendre_1(n: int) -> NDArray[np.float64]:
    """计算第一个勒让德多项式。

    Args:
        n: 结果长度

    Returns:
        numpy数组,包含第一个勒让德多项式的系数
    """
    c1 = np.zeros(n)
    total = 0.0
    for i in range(n):
        c1[i] = 2.0 * i / (n - 1.0) - 1.0
        total += c1[i] * c1[i]
    total = np.sqrt(total)
    for i in range(n):
        c1[i] /= total
    return c1


@njit
def _legendre_2(n: int) -> tuple[NDArray, NDArray]:
    """计算前两个勒让德多项式。

    Args:
        n: 结果长度

    Returns:
        包含前两个勒让德多项式系数的元组
    """
    c1 = _legendre_1(n)
    c2 = np.zeros(n)
    total = 0.0
    for i in range(n):
        c2[i] = c1[i] * c1[i]
        total += c2[i]
    mean = total / n
    total = 0.0
    for i in range(n):
        c2[i] -= mean
        total += c2[i] * c2[i]
    total = np.sqrt(total)
    for i in range(n):
        c2[i] /= total
    return c1, c2


@njit
def _legendre_3(n: int) -> tuple[NDArray, NDArray, NDArray]:
    """计算前三个勒让德多项式。

    第一个多项式测量线性趋势,第二个测量二次趋势,第三个测量三次趋势。

    Args:
        n: 结果长度

    Returns:
        包含前三个勒让德多项式系数的元组
    """
    c1, c2 = _legendre_2(n)
    c3 = np.zeros(n)
    total = 0.0
    for i in range(n):
        c3[i] = c1[i] * c1[i] * c1[i]
        total += c3[i]
    mean = total / n
    total = 0.0
    for i in range(n):
        c3[i] -= mean
        total += c3[i] * c3[i]
    total = np.sqrt(total)
    for i in range(n):
        c3[i] /= total
    proj = 0.0
    for i in range(n):
        proj += c1[i] * c3[i]
    total = 0.0
    for i in range(n):
        c3[i] -= proj * c1[i]
        total += c3[i] * c3[i]
    total = np.sqrt(total)
    for i in range(n):
        c3[i] /= total
    return c1, c2, c3


@njit
def _trend(
    values: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    lookback: int,
    atr_length: int,
    scale: float,
    trend_type: Literal["linear", "quadratic", "cubic"],
) -> NDArray[np.float64]:
    """计算趋势强度。

    Args:
        values: 输入数据数组
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        lookback: 回看K线数量
        atr_length: 用于ATR归一化的回看周期
        scale: 返回值压缩比例,>1.0增加压缩,<1.0减少压缩
        trend_type: 趋势类型,"linear"(线性),"quadratic"(二次)或"cubic"(三次)

    Returns:
        numpy数组,包含计算的趋势强度值,范围[-50, 50]
    """
    assert (
        len(values) == len(high)
        and len(values) == len(low)
        and len(values) == len(close)
    )
    assert lookback > 0
    assert smoothing >= 0
    n = len(close)
    front_bad = lookback - 1
    for first_volume in range(n):
        if volume[first_volume] > 0:
            break
    front_bad += first_volume
    if front_bad > n:
        front_bad = n
    output = np.zeros(n)
    for i in range(first_volume, n):
        if high[i] > low[i]:
            output[i] = (
                100.0
                * (2.0 * close[i] - high[i] - low[i])
                / (high[i] - low[i])
                * volume[i]
            )
        else:
            output[i] = 0.0
    if lookback > 1:
        for i in range(n - 1, front_bad - 1, -1):
            total = 0.0
            for j in range(lookback):
                total += output[i - j]
            output[i] = total / lookback
    if flow_type == "money_flow":
        for i in range(front_bad, n):
            total = 0.0
            for j in range(lookback):
                total += volume[i - j]
            total /= lookback
            if total > 0.0:
                output[i] /= total
            else:
                output[i] = 0.0
    elif smoothing > 1:
        alpha = 2.0 / (smoothing + 1.0)
        smoothed = volume[first_volume]
        for i in range(first_volume, n):
            smoothed = alpha * volume[i] + (1.0 - alpha) * smoothed
            if smoothed > 0.0:
                output[i] /= smoothed
            else:
                output[i] = 0.0
    for i in range(front_bad):
        output[i] = 0.0
    return output


@njit
def _flow(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    lookback: int,
    smoothing: float,
    flow_type: Literal["intraday", "money_flow"],
) -> NDArray[np.float64]:
    """计算资金流向指标。

    Args:
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        volume: 成交量数组
        lookback: 回看K线数量
        smoothing: 平滑系数
        flow_type: 指标类型,"intraday"(日内)或"money_flow"(资金流向)

    Returns:
        numpy数组,包含计算的资金流向值
    """
    assert (
        len(high) == len(low)
        and len(high) == len(close)
        and len(high) == len(volume)
    )
    assert lookback > 0
    assert smoothing >= 0
    n = len(close)
    front_bad = lookback - 1
    for first_volume in range(n):
        if volume[first_volume] > 0:
            break
    front_bad += first_volume
    if front_bad > n:
        front_bad = n
    output = np.zeros(n)
    for i in range(first_volume, n):
        if high[i] > low[i]:
            output[i] = (
                100.0
                * (2.0 * close[i] - high[i] - low[i])
                / (high[i] - low[i])
                * volume[i]
            )
        else:
            output[i] = 0.0
    if lookback > 1:
        for i in range(n - 1, front_bad - 1, -1):
            total = 0.0
            for j in range(lookback):
                total += output[i - j]
            output[i] = total / lookback
    if flow_type == "money_flow":
        for i in range(front_bad, n):
            total = 0.0
            for j in range(lookback):
                total += volume[i - j]
            total /= lookback
            if total > 0.0:
                output[i] /= total
            else:
                output[i] = 0.0
    elif smoothing > 1:
        alpha = 2.0 / (smoothing + 1.0)
        smoothed = volume[first_volume]
        for i in range(first_volume, n):
            smoothed = alpha * volume[i] + (1.0 - alpha) * smoothed
            if smoothed > 0.0:
                output[i] /= smoothed
            else:
                output[i] = 0.0
    for i in range(front_bad):
        output[i] = 0.0
    return output


@njit
def intraday_intensity(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    lookback: int,
    smoothing: float = 0.0,
) -> NDArray[np.float64]:
    """计算日内强度指标。

    Args:
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        volume: 成交量数组
        lookback: 回看K线数量
        smoothing: 平滑系数,<=1表示不平滑,默认为0.0

    Returns:
        numpy数组,包含计算的日内强度值
    """
    return _flow(high, low, close, volume, lookback, smoothing, "intraday")


@njit
def money_flow(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    lookback: int,
    smoothing: float = 0.0,
) -> NDArray[np.float64]:
    """计算蔡金资金流量指标。

    Args:
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        volume: 成交量数组
        lookback: 回看K线数量
        smoothing: 平滑系数,<=1表示不平滑,默认为0.0

    Returns:
        numpy数组,包含计算的资金流量值
    """
    return _flow(high, low, close, volume, lookback, smoothing, "money_flow")


@njit
def reactivity(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    lookback: int,
    smoothing: float = 0.0,
    scale: float = 0.6,
) -> NDArray[np.float64]:
    """计算价格反应性指标。

    Args:
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        volume: 成交量数组
        lookback: 回看K线数量
        smoothing: 平滑系数
        scale: 返回值压缩比例,>1.0增加压缩,<1.0减少压缩,默认为0.6

    Returns:
        numpy数组,包含计算的价格反应性值,范围[-50, 50]
    """
    assert (
        len(high) == len(low)
        and len(high) == len(close)
        and len(high) == len(volume)
    )
    assert lookback > 0
    assert smoothing >= 0
    assert scale > 0
    n = len(close)
    front_bad = lookback
    for first_volume in range(n):
        if volume[first_volume] > 0:
            break
    front_bad += first_volume
    if front_bad > n:
        front_bad = n
    output = np.zeros(n)
    for i in range(front_bad):
        output[i] = 0.0
    alpha = 2.0 / (lookback * smoothing + 1)
    lowest = low[first_volume]
    highest = high[first_volume]
    smoothed_range = highest - lowest
    smoothed_volume = volume[first_volume]
    if smoothed_range == 0:
        return output
    if first_volume + 1 >= n or first_volume + lookback >= n:
        return output
    for i in range(first_volume + 1, first_volume + lookback):
        if high[i] > highest:
            highest = high[i]
        if low[i] < lowest:
            lowest = low[i]
        smoothed_range = (
            alpha * (highest - lowest) + (1.0 - alpha) * smoothed_range
        )
        smoothed_volume = alpha * volume[i] + (1.0 - alpha) * smoothed_volume
    for i in range(front_bad, n):
        lowest = low[i]
        highest = high[i]
        for j in range(1, lookback + 1):
            if high[i - j] > highest:
                highest = high[i - j]
            if low[i - j] < lowest:
                lowest = low[i - j]
        smoothed_range = (
            alpha * (highest - lowest) + (1.0 - alpha) * smoothed_range
        )
        smoothed_volume = alpha * volume[i] + (1.0 - alpha) * smoothed_volume
        aspect_ratio = (highest - lowest) / smoothed_range
        if volume[i] > 0.0 and smoothed_volume > 0.0:
            aspect_ratio /= volume[i] / smoothed_volume
        else:
            aspect_ratio = 1.0
        output[i] = aspect_ratio * (close[i] - close[i - lookback])
        output[i] /= smoothed_range
        output[i] = 100.0 * normal_cdf(scale * output[i]) - 50.0
    return output


@njit
def price_volume_fit(
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    lookback: int,
    scale: float = 9.0,
) -> NDArray[np.float64]:
    """计算价格成交量拟合指标。

    Args:
        close: 收盘价数组
        volume: 成交量数组
        lookback: 回看K线数量
        scale: 返回值压缩比例,>1.0增加压缩,<1.0减少压缩,默认为9.0

    Returns:
        numpy数组,包含计算的价格成交量拟合值,范围[-50, 50]
    """
    assert len(close) == len(volume)
    assert lookback > 0
    assert scale > 0
    n = len(close)
    front_bad = lookback - 1
    for first_volume in range(n):
        if volume[first_volume] > 0:
            break
    front_bad += first_volume
    if front_bad > n:
        front_bad = n
    output = np.zeros(n)
    for i in range(front_bad, n):
        x_mean = y_mean = 0.0
        for j in range(lookback):
            k = i - j
            x_mean += np.log(volume[k] + 1.0)
            y_mean += np.log(close[k])
        x_mean /= lookback
        y_mean /= lookback
        xss = xy = 0.0
        for j in range(lookback):
            k = i - j
            x_diff = np.log(volume[k] + 1.0) - x_mean
            y_diff = np.log(close[k]) - y_mean
            xss += x_diff * x_diff
            xy += x_diff * y_diff
        coef = xy / (xss + 1.0e-30)
        output[i] = 100.0 * normal_cdf(scale * coef) - 50.0
    return output


@njit
def volume_weighted_ma_ratio(
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    lookback: int,
    scale: float = 1.0,
) -> NDArray[np.float64]:
    """计算成交量加权移动平均比率。

    Args:
        close: 收盘价数组
        volume: 成交量数组
        lookback: 回看K线数量
        scale: 返回值压缩比例,>1.0增加压缩,<1.0减少压缩,默认为1.0

    Returns:
        numpy数组,包含计算的成交量加权移动平均比率值,范围[-50, 50]
    """
    assert len(close) == len(volume)
    assert lookback > 0
    assert scale > 0
    n = len(close)
    front_bad = lookback - 1
    for first_volume in range(n):
        if volume[first_volume] > 0:
            break
    front_bad += first_volume
    if front_bad > n:
        front_bad = n
    output = np.zeros(n)
    for i in range(front_bad, n):
        total = numer = denom = 0.0
        for j in range(i - lookback + 1, i + 1):
            numer += volume[j] * close[j]
            denom += close[j]
            total += volume[j]
        if total > 0.0:
            output[i] = (
                1000.0
                * np.log(lookback * numer / (total * denom))
                / np.sqrt(lookback)
            )
            output[i] = 100.0 * normal_cdf(scale * output[i]) - 50.0
        else:
            output[i] = 0.0
    return output


@njit
def _on_balance_volume(
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    lookback: int,
    delta_length: int,
    scale: float,
    volume_type: Literal["normalized", "delta"],
) -> NDArray[np.float64]:
    """计算能量潮指标(OBV)。

    Args:
        close: 收盘价数组
        volume: 成交量数组
        lookback: 回看K线数量
        delta_length: 差分长度
        scale: 返回值压缩比例,>1.0增加压缩,<1.0减少压缩
        volume_type: 指标类型,"normalized"(归一化)或"delta"(差分)

    Returns:
        numpy数组,包含计算的OBV值,范围[-50, 50]
    """
    assert len(close) == len(volume)
    assert lookback > 0
    assert delta_length >= 0
    assert scale > 0
    n = len(close)
    front_bad = lookback
    for first_volume in range(n):
        if volume[first_volume] > 0:
            break
    front_bad += first_volume
    if front_bad > n:
        front_bad = n
    output = np.zeros(n)
    for i in range(front_bad, n):
        signed_volume = total_volume = 0.0
        for j in range(lookback):
            if close[i - j] > close[i - j - 1]:
                signed_volume += volume[i - j]
            elif close[i - j] < close[i - j - 1]:
                signed_volume -= volume[i - j]
            total_volume += volume[i - j]
        if total_volume <= 0.0:
            output[i] = 0.0
            continue
        value = signed_volume / total_volume
        value *= np.sqrt(lookback)
        output[i] = 100.0 * normal_cdf(scale * value) - 50.0
    if volume_type == "delta":
        if delta_length < 1:
            delta_length = 1
        front_bad += delta_length
        if front_bad > n:
            front_bad = n
        for i in range(n - 1, front_bad - 1, -1):
            output[i] -= output[i - delta_length]
    return output


@njit
def normalized_on_balance_volume(
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    lookback: int,
    scale: float = 0.6,
) -> NDArray[np.float64]:
    """计算归一化能量潮指标。

    Args:
        close: 收盘价数组
        volume: 成交量数组
        lookback: 回看K线数量
        scale: 返回值压缩比例,>1.0增加压缩,<1.0减少压缩,默认为0.6

    Returns:
        numpy数组,包含计算的归一化OBV值,范围[-50, 50]
    """
    return _on_balance_volume(close, volume, lookback, 0, scale, "normalized")


@njit
def delta_on_balance_volume(
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    lookback: int,
    delta_length: int = 0,
    scale: float = 0.6,
) -> NDArray[np.float64]:
    """计算差分能量潮指标。

    Args:
        close: 收盘价数组
        volume: 成交量数组
        lookback: 回看K线数量
        delta_length: 差分长度
        scale: 返回值压缩比例,>1.0增加压缩,<1.0减少压缩,默认为0.6

    Returns:
        numpy数组,包含计算的差分OBV值,范围[-50, 50]
    """
    return _on_balance_volume(
        close, volume, lookback, delta_length, scale, "delta"
    )


@njit
def _normalized_volume_index(
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    lookback: int,
    scale: float,
    volume_type: Literal["positive", "negative"],
) -> NDArray[np.float64]:
    assert len(close) == len(volume)
    assert lookback > 0
    assert scale > 0
    n = len(close)
    volatility_length = 2 * lookback
    if volatility_length < 250:
        volatility_length = 250
    front_bad = volatility_length
    for first_volume in range(n):
        if volume[first_volume] > 0:
            break
    front_bad += first_volume
    if front_bad > n:
        front_bad = n
    output = np.zeros(n)
    for i in range(front_bad, n):
        total = 0.0
        if volume_type == "positive":
            for j in range(lookback):
                if volume[i - j] > volume[i - j - 1]:
                    total += np.log(close[i - j] / close[i - j - 1])
        else:
            for j in range(lookback):
                if volume[i - j] < volume[i - j - 1]:
                    total += np.log(close[i - j] / close[i - j - 1])
        total /= np.sqrt(lookback)
        denom = np.sqrt(_variance(True, i, volatility_length, close))
        if denom > 0.0:
            total /= denom
            output[i] = 100.0 * normal_cdf(scale * total) - 50.0
        else:
            output[i] = 0.0
    return output


@njit
def normalized_positive_volume_index(
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    lookback: int,
    scale: float = 0.5,
) -> NDArray[np.float64]:
    """计算归一化正成交量指标。

    Args:
        close: 收盘价数组
        volume: 成交量数组
        lookback: 回看K线数量
        scale: 返回值压缩比例,>1.0增加压缩,<1.0减少压缩,默认为0.5

    Returns:
        numpy数组,包含计算的归一化正成交量指标值,范围[-50, 50]
    """
    return _normalized_volume_index(close, volume, lookback, scale, "positive")


@njit
def normalized_negative_volume_index(
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    lookback: int,
    scale: float = 0.5,
) -> NDArray[np.float64]:
    """计算归一化负成交量指标。

    Args:
        close: 收盘价数组
        volume: 成交量数组
        lookback: 回看K线数量
        scale: 返回值压缩比例,>1.0增加压缩,<1.0减少压缩,默认为0.5

    Returns:
        numpy数组,包含计算的归一化负成交量指标值,范围[-50, 50]
    """
    return _normalized_volume_index(close, volume, lookback, scale, "negative")


@njit
def volume_momentum(
    volume: NDArray[np.float64],
    short_length: int,
    multiplier: int = 2,
    scale: float = 3.0,
) -> NDArray[np.float64]:
    """计算成交量动量指标。

    Args:
        volume: 成交量数组
        short_length: 短期回看K线数量
        multiplier: 回看周期乘数,默认为2
        scale: 返回值压缩比例,>1.0增加压缩,<1.0减少压缩,默认为3.0

    Returns:
        numpy数组,包含计算的成交量动量值,范围[-50, 50]
    """
    assert short_length > 0
    assert multiplier >= 1
    assert scale > 0
    n = len(volume)
    if multiplier < 2:
        multiplier = 2
    long_length = short_length * multiplier
    front_bad = long_length - 1
    for first_volume in range(n):
        if volume[first_volume] > 0:
            break
    front_bad += first_volume
    if front_bad > n:
        front_bad = n
    output = np.zeros(n)
    denom = np.exp(np.log(multiplier) / 3.0)
    for i in range(front_bad, n):
        short_sum = 0.0
        for j in range(i - short_length + 1, i + 1):
            short_sum += volume[j]
        long_sum = short_sum
        for j in range(i - long_length + 1, i - short_length + 1):
            long_sum += volume[j]
        short_sum /= short_length
        long_sum /= long_length
        if long_sum > 0.0 and short_sum > 0.0:
            output[i] = np.log(short_sum / long_sum) / denom
            output[i] = 100.0 * normal_cdf(scale * output[i]) - 50.0
        else:
            output[i] = 0.0
    return output


@njit
def laguerre_rsi(
    open: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    fe_length: int = 13,
) -> NDArray[np.float64]:
    """计算拉盖尔相对强弱指标(Laguerre RSI)。

    Args:
        open: 开盘价数组
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        fe_length: 分形能量周期长度,默认为13

    Returns:
        numpy数组,包含计算的拉盖尔RSI值
    """
    assert (
        len(open) == len(high)
        and len(open) == len(low)
        and len(open) == len(close)
    )
    assert fe_length > 0
    n = len(close)
    output = np.zeros(n)
    if n <= fe_length:
        return output
    alpha = np.zeros(n)
    L0_1, L1_1, L2_1, L3_1 = 0.0, 0.0, 0.0, 0.0
    for i in range(fe_length, n):
        OC = (open[i] + close[i - 1]) / 2.0
        HC = max(high[i], close[i - 1])
        LC = min(low[i], close[i - 1])
        fe_src = (OC + HC + LC + close[i]) / 4.0
        highest = max(high[i + 1 - fe_length : i + 1])
        lowest = min(low[i + 1 - fe_length : i + 1])
        denom = highest - lowest
        if denom == 0:
            output[i] = alpha[i] = 0
            continue
        s = 0
        for i in range(fe_length):
            diff = max(high[i - i], close[i - i - 1]) - min(
                low[i - i], close[i - i - 1]
            )
            s += diff / denom
        fe_alpha = np.log(s) / np.log(fe_length)
        alpha[i] = fe_alpha * 100
        L0 = fe_alpha * fe_src + (1 - fe_alpha) * L0_1
        L1 = -(1 - fe_alpha) * L0 + L0_1 + (1 - fe_alpha) * L1_1
        L2 = -(1 - fe_alpha) * L1 + L1_1 + (1 - fe_alpha) * L2_1
        L3 = -(1 - fe_alpha) * L2 + L2_1 + (1 - fe_alpha) * L3_1
        CU = (
            (L0 - L1 if L0 >= L1 else 0)
            + (L1 - L2 if L1 >= L2 else 0)
            + (L2 - L3 if L2 >= L3 else 0)
        )
        CD = (
            (0 if L0 >= L1 else L1 - L0)
            + (0 if L1 >= L2 else L2 - L1)
            + (0 if L2 >= L3 else L3 - L2)
        )
        lrsi = CU / (CU + CD) if CU + CD != 0 else 0
        output[i] = lrsi * 100
        L0_1, L1_1, L2_1, L3_1 = L0, L1, L2, L3
    return output


@njit
def adx(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    lookback: int,
) -> NDArray[np.float64]:
    """计算平均趋向指标(ADX)。

    Args:
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        lookback: 回看K线数量

    Returns:
        numpy数组,包含计算的ADX值
    """
    assert len(high) == len(low) and len(high) == len(close)
    assert lookback > 0
    n = len(close)
    output = np.zeros(n)
    if n <= 2 * lookback:
        return output
    output[0] = 0
    dms_plus = dms_minus = atr_ = 0.0
    for i in range(1, lookback + 1):
        dm_plus = high[i] - high[i - 1]
        dm_minus = low[i - 1] - low[i]
        if dm_plus >= dm_minus:
            dm_minus = 0.0
        else:
            dm_plus = 0.0
        if dm_plus < 0.0:
            dm_plus = 0.0
        if dm_minus < 0.0:
            dm_minus = 0.0
        dms_plus += dm_plus
        dms_minus += dm_minus
        term = high[i] - low[i]
        if high[i] - close[i - 1] > term:
            term = high[i] - close[i - 1]
        if close[i - 1] - low[i] > term:
            term = close[i - 1] - low[i]
        atr_ += term
        di_plus = dms_plus / (atr_ + 1.0e-10)
        di_minus = dms_minus / (atr_ + 1.0e-10)
        adx_ = np.fabs(di_plus - di_minus) / (di_plus + di_minus + 1.0e-10)
        output[i] = 100 * adx_
    for i in range(lookback + 1, 2 * lookback):
        dm_plus = high[i] - high[i - 1]
        dm_minus = low[i - 1] - low[i]
        if dm_plus >= dm_minus:
            dm_minus = 0.0
        else:
            dm_plus = 0.0
        if dm_plus < 0.0:
            dm_plus = 0.0
        if dm_minus < 0.0:
            dm_minus = 0.0
        dms_plus = (lookback - 1.0) / lookback * dms_plus + dm_plus
        dms_minus = (lookback - 1.0) / lookback * dms_minus + dm_minus
        term = high[i] - low[i]
        if high[i] - close[i - 1] > term:
            term = high[i] - close[i - 1]
        if close[i - 1] - low[i] > term:
            term = close[i - 1] - low[i]
        atr_ = (lookback - 1.0) / lookback * atr_ + term
        di_plus = dms_plus / (atr_ + 1.0e-10)
        di_minus = dms_minus / (atr_ + 1.0e-10)
        adx_ += np.fabs(di_plus - di_minus) / (di_plus + di_minus + 1.0e-10)
        output[i] = 100 * adx_ / (i - lookback + 1)
    adx_ /= lookback
    for i in range(2 * lookback, n):
        dm_plus = high[i] - high[i - 1]
        dm_minus = low[i - 1] - low[i]
        if dm_plus >= dm_minus:
            dm_minus = 0.0
        else:
            dm_plus = 0.0
        if dm_plus < 0.0:
            dm_plus = 0.0
        if dm_minus < 0.0:
            dm_minus = 0.0
        dms_plus = (lookback - 1.0) / lookback * dms_plus + dm_plus
        dms_minus = (lookback - 1.0) / lookback * dms_minus + dm_minus
        term = high[i] - low[i]
        if high[i] - close[i - 1] > term:
            term = high[i] - close[i - 1]
        if close[i - 1] - low[i] > term:
            term = close[i - 1] - low[i]
        atr_ = (lookback - 1.0) / lookback * atr_ + term
        di_plus = dms_plus / (atr_ + 1.0e-10)
        di_minus = dms_minus / (atr_ + 1.0e-10)
        term = np.fabs(di_plus - di_minus) / (di_plus + di_minus + 1.0e-10)
        adx_ = (lookback - 1.0) / lookback * adx_ + term / lookback
        output[i] = 100 * adx_
    return output


@njit
def _aroon(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    lookback: int,
    aroon_type: Literal["up", "down", "diff"],
) -> NDArray[np.float64]:
    """计算Aroon指标。

    Args:
        high: 最高价数组
        low: 最低价数组
        lookback: 回看K线数量
        aroon_type: 指标类型,"up"(上升),"down"(下降)或"diff"(差值)

    Returns:
        numpy数组,包含计算的Aroon指标值
    """
    assert len(high) == len(low)
    assert lookback > 0
    n = len(high)
    output = np.zeros(n)
    if aroon_type == "up" or aroon_type == "down":
        output[0] = 50
    elif aroon_type == "diff":
        output[0] = 0
    for i in range(1, n):
        if aroon_type == "up" or aroon_type == "diff":
            i_max = i
            x_max = high[i]
            for i in range(i - 1, i - lookback - 1, -1):
                if i < 0:
                    break
                if high[i] > x_max:
                    x_max = high[i]
                    i_max = i
        if aroon_type == "down" or aroon_type == "diff":
            i_min = i
            x_min = low[i]
            for i in range(i - 1, i - lookback - 1, -1):
                if i < 0:
                    break
                if low[i] < x_min:
                    x_min = low[i]
                    i_min = i
        if aroon_type == "up":
            output[i] = 100 * (lookback - (i - i_max)) / lookback
        elif aroon_type == "down":
            output[i] = 100 * (lookback - (i - i_min)) / lookback
        else:
            max_val = 100 * (lookback - (i - i_max)) / lookback
            min_val = 100 * (lookback - (i - i_min)) / lookback
            output[i] = max_val - min_val
    return output


@njit
def aroon_up(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    lookback: int,
) -> NDArray[np.float64]:
    """计算Aroon上升趋势指标。

    Args:
        high: 最高价数组
        low: 最低价数组
        lookback: 回看K线数量

    Returns:
        numpy数组,包含计算的Aroon上升趋势值
    """
    return _aroon(high, low, lookback, "up")


@njit
def aroon_down(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    lookback: int,
) -> NDArray[np.float64]:
    """计算Aroon下降趋势指标。

    Args:
        high: 最高价数组
        low: 最低价数组
        lookback: 回看K线数量

    Returns:
        numpy数组,包含计算的Aroon下降趋势值
    """
    return _aroon(high, low, lookback, "down")


@njit
def aroon_diff(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    lookback: int,
) -> NDArray[np.float64]:
    """计算Aroon上升趋势与下降趋势的差值。

    Args:
        high: 最高价数组
        low: 最低价数组
        lookback: 回看K线数量

    Returns:
        numpy数组,包含计算的Aroon趋势差值
    """
    return _aroon(high, low, lookback, "diff")


@njit
def close_minus_ma(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    lookback: int,
    atr_length: int,
    scale: float = 1.0,
) -> NDArray[np.float64]:
    """计算收盘价减去移动平均的差值。

    Args:
        close: 收盘价数组
        high: 最高价数组
        low: 最低价数组
        lookback: 回看K线数量
        atr_length: 用于ATR归一化的回看周期
        scale: 返回值压缩比例,>1.0增加压缩,<1.0减少压缩,默认为1.0

    Returns:
        numpy数组,包含计算的差值,范围[-50, 50]
    """
    assert len(high) == len(low) and len(high) == len(close)
    assert lookback > 0
    assert atr_length > 0
    assert scale > 0
    n = len(close)
    front_bad = max(lookback, atr_length)
    if front_bad > n:
        front_bad = n
    output = np.zeros(n)
    for i in range(front_bad, n):
        total = 0.0
        for j in range(i - lookback, i):
            total += np.log(close[j])
        total /= lookback
        denom = _atr(i, atr_length, high, low, close, True)
        if denom > 0.0:
            denom *= np.sqrt(lookback + 1.0)
            output[i] = (np.log(close[i]) - total) / denom
            output[i] = 100.0 * normal_cdf(scale * output[i]) - 50.0
        else:
            output[i] = 0.0
    return output


@njit
def _deviation(
    values: NDArray[np.float64],
    lookback: int,
    scale: float,
    dev_type: Literal["linear", "quadratic", "cubic"],
) -> NDArray[np.float64]:
    """计算价格偏离趋势的程度。

    Args:
        values: 输入数据数组
        lookback: 回看K线数量
        scale: 返回值压缩比例,>1.0增加压缩,<1.0减少压缩
        dev_type: 偏离类型,"linear"(线性),"quadratic"(二次)或"cubic"(三次)

    Returns:
        numpy数组,包含计算的偏离值,范围[-50, 50]
    """
    assert lookback > 0
    assert scale > 0
    n = len(values)
    if dev_type == "linear" and lookback < 3:
        lookback = 3
    if dev_type == "quadratic" and lookback < 4:
        lookback = 4
    if dev_type == "cubic" and lookback < 5:
        lookback = 5
    front_bad = lookback - 1
    if front_bad > n:
        front_bad = n
    if dev_type == "quadratic" or dev_type == "cubic":
        work1, work2, work3 = _legendre_3(lookback)
    else:
        work1 = _legendre_1(lookback)
    output = np.zeros(n)
    for i in range(front_bad, n):
        c0 = c1 = c2 = c3 = 0.0
        dptr = work1
        dptr_i = 0
        for j in range(i - lookback + 1, i + 1):
            price = np.log(values[j])
            c0 += price
            c1 += price * dptr[dptr_i]
            dptr_i += 1
        c0 /= lookback
        if dev_type == "quadratic" or dev_type == "cubic":
            dptr = work2
            dptr_i = 0
            for j in range(i - lookback + 1, i + 1):
                price = np.log(values[j])
                c2 += price * dptr[dptr_i]
                dptr_i += 1
        if dev_type == "cubic":
            dptr = work3
            dptr_i = 0
            for j in range(i - lookback + 1, i + 1):
                price = np.log(values[j])
                c3 += price * dptr[dptr_i]
                dptr_i += 1
        j = 0
        total = 0.0
        for k in range(i - lookback + 1, i + 1):
            pred = c0 + c1 * work1[j]
            if dev_type == "quadratic" or dev_type == "cubic":
                pred += c2 * work2[j]
            if dev_type == "cubic":
                pred += c3 * work3[j]
            diff = np.log(values[k]) - pred
            total += diff * diff
            j += 1
        denom = np.sqrt(total / lookback)
        if denom > 0.0:
            pred = c0 + c1 * work1[lookback - 1]
            if dev_type == "quadratic" or dev_type == "cubic":
                pred += c2 * work2[lookback - 1]
            if dev_type == "cubic":
                pred += c3 * work3[lookback - 1]
            output[i] = (np.log(values[i]) - pred) / denom
            output[i] = 100.0 * normal_cdf(scale * output[i]) - 50.0
        else:
            output[i] = 0.0
    return output


@njit
def linear_deviation(
    values: NDArray[np.float64],
    lookback: int,
    scale: float = 0.6,
) -> NDArray[np.float64]:
    """计算价格偏离线性趋势的程度。

    Args:
        values: 输入数据数组
        lookback: 回看K线数量
        scale: 返回值压缩比例,>1.0增加压缩,<1.0减少压缩,默认为0.6

    Returns:
        numpy数组,包含计算的线性偏离值,范围[-50, 50]
    """
    return _deviation(values, lookback, scale, "linear")


@njit
def quadratic_deviation(
    values: NDArray[np.float64],
    lookback: int,
    scale: float = 0.6,
) -> NDArray[np.float64]:
    """计算价格偏离二次趋势的程度。

    Args:
        values: 输入数据数组
        lookback: 回看K线数量
        scale: 返回值压缩比例,>1.0增加压缩,<1.0减少压缩,默认为0.6

    Returns:
        numpy数组,包含计算的二次偏离值,范围[-50, 50]
    """
    return _deviation(values, lookback, scale, "quadratic")


@njit
def cubic_deviation(
    values: NDArray[np.float64],
    lookback: int,
    scale: float = 0.6,
) -> NDArray[np.float64]:
    """计算价格偏离三次趋势的程度。

    Args:
        values: 输入数据数组
        lookback: 回看K线数量
        scale: 返回值压缩比例,>1.0增加压缩,<1.0减少压缩,默认为0.6

    Returns:
        numpy数组,包含计算的三次偏离值,范围[-50, 50]
    """
    return _deviation(values, lookback, scale, "cubic")


@njit
def price_intensity(
    open: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    smoothing: float = 0.0,
    scale: float = 0.8,
) -> NDArray[np.float64]:
    """计算价格强度指标。

    Args:
        open: 开盘价数组
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        smoothing: 平滑系数,默认为0
        scale: 返回值压缩比例,>1.0增加压缩,<1.0减少压缩,默认为0.8

    Returns:
        numpy数组,包含计算的价格强度值,范围[-50, 50]
    """
    assert (
        len(open) == len(high)
        and len(open) == len(low)
        and len(open) == len(close)
    )
    assert smoothing >= 0
    assert scale > 0
    n = len(close)
    if smoothing < 1:
        smoothing = 1
    output = np.zeros(n)
    denom = high[0] - low[0]
    if denom < 1.0e-60:
        denom = 1.0e-60
    output[0] = (close[0] - open[0]) / denom
    for i in range(1, n):
        denom = high[i] - low[i]
        if high[i] - close[i - 1] > denom:
            denom = high[i] - close[i - 1]
        if close[i - 1] - low[i] > denom:
            denom = close[i - 1] - low[i]
        if denom < 1.0e-60:
            denom = 1.0e-60
        output[i] = (close[i] - open[i]) / denom
    if smoothing > 1:
        alpha = 2.0 / (smoothing + 1.0)
        smoothed = output[0]
        for i in range(1, n):
            smoothed = alpha * output[i] + (1.0 - alpha) * smoothed
            output[i] = smoothed
    for i in range(n):
        output[i] = (
            100.0 * normal_cdf(scale * np.sqrt(smoothing) * output[i]) - 50.0
        )
    return output


@njit
def price_change_oscillator(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    short_length: int,
    multiplier: int,
    scale: float = 4.0,
) -> NDArray[np.float64]:
    """计算价格变化振荡器。

    Args:
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        short_length: 短期回看K线数量
        multiplier: 用于计算长期回看K线数量的乘数,长期回看数量=multiplier*short_length
        scale: 返回值压缩比例,>1.0增加压缩,<1.0减少压缩,默认为4.0

    Returns:
        numpy数组,包含计算的价格变化振荡器值,范围[-50, 50]
    """
    assert len(high) == len(low) and len(high) == len(close)
    assert short_length > 0
    assert multiplier > 0
    assert scale > 0
    n = len(close)
    if multiplier < 2:
        multiplier = 2
    long_length = short_length * multiplier
    front_bad = long_length
    if front_bad > n:
        front_bad = n
    output = np.zeros(n)
    for i in range(front_bad, n):
        short_sum = 0.0
        for j in range(i - short_length - 1, i + 1):
            short_sum += np.fabs(np.log(close[j] / close[j - 1]))

        long_sum = short_sum
        for j in range(i - long_length + 1, i - short_length + 1):
            long_sum += np.fabs(np.log(close[j] / close[j - 1]))
        short_sum /= short_length
        long_sum /= long_length
        denom = 0.36 + 1.0 / short_length
        v = np.log(0.5 * multiplier) / 1.609
        denom += 0.7 * v
        denom *= _atr(i, long_length, high, low, close, True)
        if denom > 1.0e-20:
            output[i] = (short_sum - long_sum) / denom
            output[i] = 100.0 * normal_cdf(scale * output[i]) - 50.0
        else:
            output[i] = 0.0
    return output
