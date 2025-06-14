r"""Contains :class:`.DataSource`\ s used to fetch external data."""  # 包含用于获取外部数据的DataSource类

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import itertools
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Final, Iterable, Optional, Union

import alpaca.data.historical.crypto as alpaca_crypto
import alpaca.data.historical.stock as alpaca_stock
import numpy as np
import pandas as pd
import yfinance
from alpaca.data.enums import Adjustment
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from pybroker.cache import DataSourceCacheKey
from pybroker.common import (
    DataCol,
    parse_timeframe,
    to_datetime,
    to_seconds,
    verify_data_source_columns,
    verify_date_range,
)
from pybroker.scope import StaticScope


class DataSourceCacheMixin:
    """实现获取和存储缓存DataSource数据的混入类"""

    def get_cached(
        self,
        symbols: Iterable[str],  # 获取缓存数据的股票代码
        timeframe: str,  # 时间框架
        start_date: Union[str, datetime, pd.Timestamp, np.datetime64],  # 开始日期
        end_date: Union[str, datetime, pd.Timestamp, np.datetime64],  # 结束日期
        adjust: Optional[Any],  # 调整类型
    ) -> tuple[pd.DataFrame, Iterable[str]]:
        """当通过enable_data_source_cache启用缓存时，从磁盘检索缓存数据
        
        参数:
            symbols: 要获取缓存数据的股票代码
            timeframe: 指定缓存数据的时间框架分辨率的格式化字符串
                      支持以下单位:
                      - "s"/"sec": 秒
                      - "m"/"min": 分钟
                      - "h"/"hour": 小时
                      - "d"/"day": 天
                      - "w"/"week": 周
                      示例: "1h 30m"
            start_date: 缓存数据的开始日期(包含)
            end_date: 缓存数据的结束日期(包含)
            adjust: 要进行的调整类型
            
        返回:
            包含缓存数据的DataFrame和未找到缓存数据的股票代码的元组
        """
        df = pd.DataFrame()
        scope = StaticScope.instance()
        cache = scope.data_source_cache
        if cache is None:
            return df, symbols
        start_date = to_datetime(start_date)
        end_date = to_datetime(end_date)
        tf_seconds = to_seconds(timeframe)
        uncached_syms = []
        cached_syms = []
        for sym in symbols:
            cache_key = DataSourceCacheKey(
                symbol=sym,
                tf_seconds=tf_seconds,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )
            cached = cache.get(repr(cache_key))
            scope.logger.debug_get_data_source_cache(cache_key)
            if cached is None:
                uncached_syms.append(sym)
            else:
                cached_syms.append(sym)
                df = pd.concat([df, cached])
        if not uncached_syms:
            scope.logger.loaded_bar_data()
        scope.logger.info_loaded_bar_data(
            symbols=cached_syms,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )
        return df, uncached_syms

    def set_cached(
        self,
        timeframe: str,  # 时间框架
        start_date: Union[str, datetime, pd.Timestamp, np.datetime64],  # 开始日期
        end_date: Union[str, datetime, pd.Timestamp, np.datetime64],  # 结束日期
        adjust: Optional[Any],  # 调整类型
        data: pd.DataFrame,  # 要缓存的数据
    ):
        """当通过enable_data_source_cache启用缓存时，将数据存储到磁盘缓存
        
        参数:
            timeframe: 指定要缓存的数据的时间框架分辨率的格式化字符串
                      支持以下单位:
                      - "s"/"sec": 秒
                      - "m"/"min": 分钟
                      - "h"/"hour": 小时
                      - "d"/"day": 天
                      - "w"/"week": 周
                      示例: "1h 30m"
            start_date: 要缓存的数据的开始日期(包含)
            end_date: 要缓存的数据的结束日期(包含)
            adjust: 要进行的调整类型
            data: 包含要缓存的数据的DataFrame
        """
        if data.empty:
            return
        scope = StaticScope.instance()
        cache = scope.data_source_cache
        if cache is None:
            return
        start_date = to_datetime(start_date)
        end_date = to_datetime(end_date)
        tf_seconds = to_seconds(timeframe)
        for sym in data[DataCol.SYMBOL.value].unique():
            df = data[data[DataCol.SYMBOL.value] == sym]
            cache_key = DataSourceCacheKey(
                symbol=sym,
                tf_seconds=tf_seconds,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )
            cache.set(repr(cache_key), df)
            scope.logger.debug_set_data_source_cache(cache_key)


class DataSource(ABC, DataSourceCacheMixin):
    """从外部源查询数据的基类。扩展此类并重写_fetch_data方法，以实现可与Strategy一起使用的自定义DataSource"""

    def __init__(self):
        self._scope = StaticScope.instance()
        self._logger = self._scope.logger

    def query(
        self,
        symbols: Union[str, Iterable[str]],  # 要查询的股票代码
        start_date: Union[str, datetime],  # 开始日期
        end_date: Union[str, datetime],  # 结束日期
        timeframe: Optional[str] = "",  # 时间框架
        adjust: Optional[Any] = None,  # 调整类型
    ) -> pd.DataFrame:
        """查询数据。如果通过调用enable_data_source_cache启用了缓存，则返回缓存的数据
        
        参数:
            symbols: 要查询的数据的股票代码
            start_date: 要查询的数据的开始日期(包含)
            end_date: 要查询的数据的结束日期(包含)
            timeframe: 指定要查询的时间框架分辨率的格式化字符串
                      支持以下单位:
                      - "s"/"sec": 秒
                      - "m"/"min": 分钟
                      - "h"/"hour": 小时
                      - "d"/"day": 天
                      - "w"/"week": 周
                      示例: "1h 30m"
            adjust: 要进行的调整类型
            
        返回:
            包含查询数据的DataFrame
        """
        start_date = to_datetime(start_date)
        end_date = to_datetime(end_date)
        verify_date_range(start_date, end_date)
        if isinstance(symbols, str) and not symbols:
            raise ValueError("Symbols cannot be empty.")
        unique_syms = (
            frozenset((symbols,))
            if isinstance(symbols, str)
            else frozenset(symbols)
        )
        if not unique_syms:
            raise ValueError("Symbols cannot be empty.")
        timeframe = self._format_timeframe(timeframe)
        cached_df, uncached_syms = self.get_cached(
            symbols=unique_syms,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
        if not uncached_syms:
            return cached_df
        self._logger.download_bar_data_start()
        self._logger.info_download_bar_data_start(
            symbols=uncached_syms,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )
        df = self._fetch_data(
            frozenset(uncached_syms), start_date, end_date, timeframe, adjust
        )
        if (
            self._scope.data_source_cache is not None
            and not cached_df.columns.empty
            and set(cached_df.columns) != set(df.columns)
        ):
            self._logger.info_invalidate_data_source_cache()
            self._scope.data_source_cache.clear()
            return self.query(symbols, start_date, end_date, timeframe)
        verify_data_source_columns(df)
        self.set_cached(timeframe, start_date, end_date, adjust, df)
        df = pd.concat((cached_df, df))
        if not df.empty:
            df = df.sort_values(by=[DataCol.DATE.value, DataCol.SYMBOL.value])
        self._logger.download_bar_data_completed()
        return df.reset_index(drop=True)

    @abstractmethod
    def _fetch_data(
        self,
        symbols: frozenset[str],  # 要获取的股票代码集合
        start_date: datetime,  # 开始日期
        end_date: datetime,  # 结束日期
        timeframe: Optional[str],  # 时间框架
        adjust: Optional[Any],  # 调整类型
    ) -> pd.DataFrame:
        """获取数据的抽象方法，需要由子类实现
        
        参数:
            symbols: 要获取的股票代码的冻结集合
            start_date: 开始日期(包含)
            end_date: 结束日期(包含)
            timeframe: 时间框架
            adjust: 调整类型
            
        返回:
            包含获取数据的DataFrame
        """
        pass

    def _format_timeframe(self, timeframe: Optional[str]) -> str:
        """格式化时间框架字符串"""
        return "" if timeframe is None else timeframe


def _parse_alpaca_timeframe(
    timeframe: Optional[str],
) -> tuple[int, TimeFrameUnit]:
    """解析Alpaca时间框架
    
    将PyBroker的时间框架字符串转换为Alpaca的TimeFrame对象
    """
    if timeframe is None:
        raise ValueError("Timeframe needs to be specified for Alpaca.")
    parts = parse_timeframe(timeframe)
    if len(parts) != 1:
        raise ValueError(f"Invalid Alpaca timeframe: {timeframe}")
    tf = parts[0]
    if tf[1] == "min":
        unit = TimeFrameUnit.Minute
    elif tf[1] == "hour":
        unit = TimeFrameUnit.Hour
    elif tf[1] == "day":
        unit = TimeFrameUnit.Day
    elif tf[1] == "week":
        unit = TimeFrameUnit.Week
    else:
        raise ValueError(f"Invalid Alpaca timeframe: {timeframe}")
    return tf[0], unit


class Alpaca(DataSource):
    """从Alpaca获取股票数据
    
    参数:
        api_key: Alpaca API密钥
        api_secret: Alpaca API密钥
    """

    __EST: Final = "US/Eastern"  # 东部标准时间时区

    def __init__(self, api_key: str, api_secret: str):
        """初始化Alpaca数据源
        
        参数:
            api_key: Alpaca API密钥
            api_secret: Alpaca API密钥
        """
        super().__init__()
        self._stock_client = alpaca_stock.StockHistoricalDataClient(
            api_key, api_secret
        )

    def query(
        self,
        symbols: Union[str, Iterable[str]],  # 股票代码
        start_date: Union[str, datetime],  # 开始日期
        end_date: Union[str, datetime],  # 结束日期
        timeframe: Optional[str] = "1d",  # 时间框架，默认为1天
        adjust: Optional[Any] = None,  # 调整类型
    ) -> pd.DataFrame:
        """查询Alpaca股票数据
        
        参数:
            symbols: 要查询的股票代码
            start_date: 开始日期(包含)
            end_date: 结束日期(包含)
            timeframe: 时间框架，默认为"1d"(一天)
            adjust: 调整类型，支持以下选项:
                  - all: 全部调整(分红和拆分)
                  - dividend: 仅分红调整
                  - split: 仅拆分调整
                  - None: 不调整(默认)
                  
        返回:
            包含查询股票数据的DataFrame
        """
        _parse_alpaca_timeframe(timeframe)
        return super().query(symbols, start_date, end_date, timeframe, adjust)

    def _fetch_data(
        self,
        symbols: frozenset[str],  # 股票代码集合
        start_date: datetime,  # 开始日期
        end_date: datetime,  # 结束日期
        timeframe: Optional[str],  # 时间框架
        adjust: Optional[Any],  # 调整类型
    ) -> pd.DataFrame:
        """从Alpaca获取股票数据
        
        参数:
            symbols: 要获取的股票代码冻结集合
            start_date: 开始日期(包含)
            end_date: 结束日期(包含)
            timeframe: 时间框架
            adjust: 调整类型
            
        返回:
            包含获取股票数据的DataFrame
        """
        amount, unit = _parse_alpaca_timeframe(timeframe)
        adj_enum = None
        if adjust is not None:
            for member in Adjustment:
                if member.value == adjust:
                    adj_enum = member
                    break
            if adj_enum is None:
                raise ValueError(f"Unknown adjustment: {adjust}.")
        request = StockBarsRequest(
            symbol_or_symbols=list(symbols),
            start=start_date,
            end=end_date,
            timeframe=TimeFrame(amount, unit),
            limit=None,
            adjustment=adj_enum,
            feed=None,
        )
        df = self._stock_client.get_stock_bars(request).df  # type: ignore[union-attr]
        if df.columns.empty:
            return pd.DataFrame(
                columns=[
                    DataCol.SYMBOL.value,
                    DataCol.DATE.value,
                    DataCol.OPEN.value,
                    DataCol.HIGH.value,
                    DataCol.LOW.value,
                    DataCol.CLOSE.value,
                    DataCol.VOLUME.value,
                    DataCol.VWAP.value,
                ]
            )
        if df.empty:
            return df
        df = df.reset_index()
        df.rename(columns={"timestamp": DataCol.DATE.value}, inplace=True)
        df = df[[col.value for col in DataCol]]
        df[DataCol.DATE.value] = pd.to_datetime(df[DataCol.DATE.value])
        df[DataCol.DATE.value] = df[DataCol.DATE.value].dt.tz_convert(
            self.__EST
        )
        return df


class AlpacaCrypto(DataSource):
    """从Alpaca获取加密货币数据
    
    参数:
        api_key: Alpaca API密钥
        api_secret: Alpaca API密钥
    """

    TRADE_COUNT: Final = "trade_count"  # 交易次数列名
    COLUMNS: Final = (  # 列名列表
        DataCol.SYMBOL.value,
        DataCol.DATE.value,
        DataCol.OPEN.value,
        DataCol.HIGH.value,
        DataCol.LOW.value,
        DataCol.CLOSE.value,
        DataCol.VOLUME.value,
        DataCol.VWAP.value,
        TRADE_COUNT,
    )

    __EST: Final = "US/Eastern"  # 东部标准时间时区

    def __init__(self, api_key: str, api_secret: str):
        """初始化AlpacaCrypto数据源
        
        参数:
            api_key: Alpaca API密钥
            api_secret: Alpaca API密钥
        """
        super().__init__()
        self._crypto_client = alpaca_crypto.CryptoHistoricalDataClient(
            api_key, api_secret
        )

    def query(
        self,
        symbols: Union[str, Iterable[str]],  # 加密货币代码
        start_date: Union[str, datetime],  # 开始日期
        end_date: Union[str, datetime],  # 结束日期
        timeframe: Optional[str] = "1d",  # 时间框架，默认为1天
        _adjust: Optional[str] = None,  # 不适用于加密货币
    ) -> pd.DataFrame:
        """查询Alpaca加密货币数据
        
        参数:
            symbols: 要查询的加密货币代码
            start_date: 开始日期(包含)
            end_date: 结束日期(包含)
            timeframe: 时间框架，默认为"1d"(一天)
            _adjust: 不适用于加密货币，但为了保持接口一致性而保留
            
        返回:
            包含查询加密货币数据的DataFrame
        """
        _parse_alpaca_timeframe(timeframe)
        return super().query(symbols, start_date, end_date, timeframe, _adjust)

    def _fetch_data(
        self,
        symbols: frozenset[str],  # 加密货币代码集合
        start_date: datetime,  # 开始日期
        end_date: datetime,  # 结束日期
        timeframe: Optional[str],  # 时间框架
        _adjust: Optional[str],  # 不适用于加密货币
    ) -> pd.DataFrame:
        """从Alpaca获取加密货币数据
        
        参数:
            symbols: 要获取的加密货币代码冻结集合
            start_date: 开始日期(包含)
            end_date: 结束日期(包含)
            timeframe: 时间框架
            _adjust: 不适用于加密货币
            
        返回:
            包含获取加密货币数据的DataFrame
        """
        amount, unit = _parse_alpaca_timeframe(timeframe)
        request = CryptoBarsRequest(
            symbol_or_symbols=list(symbols),
            start=start_date,
            end=end_date,
            timeframe=TimeFrame(amount, unit),
            limit=None,
        )
        df = self._crypto_client.get_crypto_bars(request).df  # type: ignore[union-attr]
        if df.columns.empty:
            return pd.DataFrame(columns=self.COLUMNS)
        if df.empty:
            return df
        df = df.reset_index()
        df.rename(columns={"timestamp": DataCol.DATE.value}, inplace=True)
        df = df[[col for col in self.COLUMNS]]
        df[DataCol.DATE.value] = pd.to_datetime(df[DataCol.DATE.value])
        df[DataCol.DATE.value] = df[DataCol.DATE.value].dt.tz_convert(
            self.__EST
        )
        return df


class YFinance(DataSource):
    r"""从Yahoo Finance获取数据
    
    参数:
        auto_adjust: 是否自动调整收盘价。如果为True，则调整后的收盘价存储在close列中。
                    默认为False
        
    属性:
        ADJ_CLOSE: 调整后收盘价的列名
    """

    ADJ_CLOSE: Final = "adj_close"  # 调整后收盘价列名
    __TIMEFRAME: Final = "1d"  # 固定的时间框架为1天

    def __init__(self, auto_adjust: bool = False):
        """初始化YFinance数据源
        
        参数:
            auto_adjust: 是否自动调整收盘价，默认为False
        """
        super().__init__()
        self._auto_adjust = auto_adjust
        self._scope.register_custom_cols(self.ADJ_CLOSE)

    def query(
        self,
        symbols: Union[str, Iterable[str]],  # 股票代码
        start_date: Union[str, datetime],  # 开始日期
        end_date: Union[str, datetime],  # 结束日期
        _timeframe: Optional[str] = "",  # 不适用于Yahoo Finance
        _adjust: Optional[Any] = None,  # 不适用于Yahoo Finance
    ) -> pd.DataFrame:
        """查询Yahoo Finance数据
        
        参数:
            symbols: 要查询的股票代码
            start_date: 开始日期(包含)
            end_date: 结束日期(包含)
            _timeframe: 不适用于Yahoo Finance，但为了保持接口一致性而保留
            _adjust: 不适用于Yahoo Finance，但为了保持接口一致性而保留
            
        返回:
            包含查询股票数据的DataFrame
        """
        return super().query(
            symbols, start_date, end_date, self.__TIMEFRAME, _adjust
        )

    def _fetch_data(
        self,
        symbols: frozenset[str],  # 股票代码集合
        start_date: datetime,  # 开始日期
        end_date: datetime,  # 结束日期
        _timeframe: Optional[str],  # 不适用于Yahoo Finance
        _adjust: Optional[Any],  # 不适用于Yahoo Finance
    ) -> pd.DataFrame:
        """从Yahoo Finance获取股票数据
        
        参数:
            symbols: 要获取的股票代码冻结集合
            start_date: 开始日期(包含)
            end_date: 结束日期(包含)
            _timeframe: 不适用于Yahoo Finance
            _adjust: 不适用于Yahoo Finance
            
        返回:
            包含获取股票数据的DataFrame
        """
        show_yf_progress_bar = (
            not self._logger._disabled
            and not self._logger._progress_bar_disabled
        )
        df = yfinance.download(
            list(symbols),
            start=start_date,
            end=end_date,
            progress=show_yf_progress_bar,
            auto_adjust=self._auto_adjust,
        )
        if df.columns.empty:
            columns = [
                DataCol.SYMBOL.value,
                DataCol.DATE.value,
                DataCol.OPEN.value,
                DataCol.HIGH.value,
                DataCol.LOW.value,
                DataCol.CLOSE.value,
                DataCol.VOLUME.value,
            ]
            if not self._auto_adjust:
                columns.append(self.ADJ_CLOSE)
            return pd.DataFrame(columns=columns)
        if df.empty:
            return df
        df = df.reset_index()
        result = pd.DataFrame()
        if len(symbols) == 1:
            result[DataCol.DATE.value] = df["Date"].values
            result[DataCol.SYMBOL.value] = tuple(
                itertools.repeat(next(iter(symbols)), len(df["Close"].values))
            )
            result[DataCol.OPEN.value] = df["Open"].values
            result[DataCol.HIGH.value] = df["High"].values
            result[DataCol.LOW.value] = df["Low"].values
            result[DataCol.CLOSE.value] = df["Close"].values
            result[DataCol.VOLUME.value] = df["Volume"].values
            if not self._auto_adjust:
                result[self.ADJ_CLOSE] = df["Adj Close"].values
        else:
            df.columns = df.columns.to_flat_index()
            for sym in symbols:
                sym_df = pd.DataFrame()
                sym_df[DataCol.DATE.value] = df[("Date", "")].values
                sym_df[DataCol.SYMBOL.value] = tuple(
                    itertools.repeat(sym, len(df[("Close", sym)].values))
                )
                sym_df[DataCol.OPEN.value] = df[("Open", sym)].values
                sym_df[DataCol.HIGH.value] = df[("High", sym)].values
                sym_df[DataCol.LOW.value] = df[("Low", sym)].values
                sym_df[DataCol.CLOSE.value] = df[("Close", sym)].values
                sym_df[DataCol.VOLUME.value] = df[("Volume", sym)].values
                if not self._auto_adjust:
                    sym_df[self.ADJ_CLOSE] = df[("Adj Close", sym)].values
                result = pd.concat((result, sym_df))
        return result

# 该模块提供了PyBroker的数据获取功能，支持从多种数据源获取行情数据。
# 核心类DataSource是所有数据源的基类，定义了数据查询的标准接口。
# 内置支持多种数据源：Alpaca股票数据、Alpaca加密货币数据和Yahoo Finance数据。
# 提供了数据缓存机制，通过DataSourceCacheMixin实现，可以避免重复获取相同的数据。
# 用户可以通过继承DataSource类实现自定义数据源，以支持更多的数据提供商。
