"""Contains scopes that store data and object references used to execute a
:class:`pybroker.strategy.Strategy`.
"""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import numpy as np
import pandas as pd
from pybroker.common import (
    BarData,
    DataCol,
    IndicatorSymbol,
    ModelSymbol,
    PriceType,
    TrainedModel,
    to_decimal,
)
from pybroker.log import Logger
from collections import defaultdict
from decimal import Decimal
from diskcache import Cache
from numpy.typing import NDArray
from typing import (
    Any,
    Callable,
    Final,
    Iterable,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Union,
)

_EMPTY_PARAM: Final = object()


class StaticScope:
    """A static registry of data and object references.
    
    静态数据和对象引用的注册表。

    Attributes:
        logger: :class:`pybroker.log.Logger`
        data_source_cache: :class:`diskcache.Cache` that stores data retrieved
            from :class:`pybroker.data.DataSource`.
            存储从数据源检索的数据的缓存。
        data_source_cache_ns: Namespace set for  :attr:`.data_source_cache`.
            数据源缓存的命名空间。
        indicator_cache: :class:`diskcache.Cache` that stores
            :class:`pybroker.indicator.Indicator` data.
            存储指标数据的缓存。
        indicator_cache_ns: Namespace set for :attr:`.indicator_cache`.
            指标缓存的命名空间。
        model_cache: :class:`diskcache.Cache` that stores trained models.
            存储训练好的模型的缓存。
        model_cache_ns: Namespace set for :attr:`.model_cache`.
            模型缓存的命名空间。
        default_data_cols: Default data columns in :class:`pandas.DataFrame`
            retrieved from a :class:`pybroker.data.DataSource`.
            从数据源检索的DataFrame中的默认数据列。
        custom_data_cols: User-defined data columns in
            :class:`pandas.DataFrame` retrieved from a
            :class:`pybroker.data.DataSource`.
            用户定义的数据列。
    """

    __instance = None

    def __init__(self):
        self.logger = Logger(self)
        self.data_source_cache: Optional[Cache] = None
        self.data_source_cache_ns: str = ""
        self.indicator_cache: Optional[Cache] = None
        self.indicator_cache_ns: str = ""
        self.model_cache: Optional[Cache] = None
        self.model_cache_ns: str = ""
        self._indicators = {}
        self._model_sources = {}
        self.default_data_cols = frozenset(
            (
                DataCol.DATE.value,
                DataCol.OPEN.value,
                DataCol.HIGH.value,
                DataCol.LOW.value,
                DataCol.CLOSE.value,
                DataCol.VOLUME.value,
                DataCol.VWAP.value,
            )
        )
        self.custom_data_cols = set()
        self._cols_frozen: bool = False
        self._params: dict[str, Any] = {}

    def set_indicator(self, indicator):
        """Stores :class:`pybroker.indicator.Indicator` in static scope.
        
        在静态作用域中存储指标。
        """
        self._indicators[indicator.name] = indicator

    def has_indicator(self, name: str) -> bool:
        """Whether :class:`pybroker.indicator.Indicator` is stored in static
        scope.
        
        检查指定名称的指标是否存储在静态作用域中。
        """
        return name in self._indicators

    def get_indicator(self, name: str):
        """Retrieves a :class:`pybroker.indicator.Indicator` from static
        scope.
        
        从静态作用域中检索指定名称的指标。
        """
        if not self.has_indicator(name):
            raise ValueError(f"Indicator {name!r} does not exist.")
        return self._indicators[name]

    def get_indicator_names(self, model_name: str) -> tuple[str]:
        """Returns a ``tuple[str]`` of all
        :class:`pybroker.indicator.Indicator` names that are registered with
        :class:`pybroker.model.ModelSource` having ``model_name``.
        
        返回注册到指定模型名称的所有指标名称的元组。
        """
        return self._model_sources[model_name].indicators

    def set_model_source(self, source):
        """Stores :class:`pybroker.model.ModelSource` in static scope.
        
        在静态作用域中存储模型源。
        """
        self._model_sources[source.name] = source

    def has_model_source(self, name: str) -> bool:
        """Whether :class:`pybroker.model.ModelSource` is stored in static
        scope.
        
        检查指定名称的模型源是否存储在静态作用域中。
        """
        return name in self._model_sources

    def get_model_source(self, name: str):
        """Retrieves a :class:`pybroker.model.ModelSource` from static
        scope.
        
        从静态作用域中检索指定名称的模型源。
        """
        if not self.has_model_source(name):
            raise ValueError(f"ModelSource {name!r} does not exist.")
        return self._model_sources[name]

    def register_custom_cols(self, names: Union[str, Iterable[str]], *args):
        """Registers user-defined column names.
        
        注册用户自定义的列名。
        """
        self._verify_unfrozen_cols()
        if isinstance(names, str):
            names = (names, *args)
        else:
            names = (*names, *args)
        names = filter(lambda col: col not in self.default_data_cols, names)
        self.custom_data_cols.update(names)

    def unregister_custom_cols(self, names: Union[str, Iterable[str]], *args):
        """Unregisters user-defined column names.
        
        注销用户自定义的列名。
        """
        self._verify_unfrozen_cols()
        if isinstance(names, str):
            names = (names, *args)
        else:
            names = (*names, *args)
        self.custom_data_cols.difference_update(names)

    @property
    def all_data_cols(self) -> frozenset[str]:
        """All registered data column names.
        
        所有已注册的数据列名。
        """
        return self.default_data_cols | self.custom_data_cols

    def _verify_unfrozen_cols(self):
        if self._cols_frozen:
            raise ValueError("Cannot modify columns when strategy is running.")

    def freeze_data_cols(self):
        """Prevents additional data columns from being registered.
        
        防止注册额外的数据列，通常在策略运行时调用。
        """
        self._cols_frozen = True

    def unfreeze_data_cols(self):
        """Allows additional data columns to be registered if
        :func:`pybroker.scope.StaticScope.freeze_data_cols` was called.
        
        如果之前调用了freeze_data_cols，允许重新注册额外的数据列。
        """
        self._cols_frozen = False

    def param(
        self, name: str, value: Optional[Any] = _EMPTY_PARAM
    ) -> Optional[Any]:
        """Get or set a global parameter.
        
        获取或设置一个全局参数。
        """
        if value is _EMPTY_PARAM:
            return self._params.get(name, None)
        self._params[name] = value
        return value

    @classmethod
    def instance(cls) -> "StaticScope":
        """Returns singleton instance.
        
        返回SingleScope的单例实例。
        """
        if cls.__instance is None:
            cls.__instance = StaticScope()
        return cls.__instance


def disable_logging():
    """Disables event logging.
    
    禁用事件日志记录。
    """
    StaticScope.instance().logger.disable()


def enable_logging():
    """Enables event logging.
    
    启用事件日志记录。
    """
    StaticScope.instance().logger.enable()


def disable_progress_bar():
    """Disables logging a progress bar.
    
    禁用进度条显示。
    """
    StaticScope.instance().logger.disable_progress_bar()


def enable_progress_bar():
    """Enables logging a progress bar.
    
    启用进度条显示。
    """
    StaticScope.instance().logger.enable_progress_bar()


def register_columns(names: Union[str, Iterable[str]], *args):
    """Registers ``names`` of user-defined data columns.
    
    注册用户自定义的数据列名。
    """
    StaticScope.instance().register_custom_cols(names, *args)


def unregister_columns(names: Union[str, Iterable[str]], *args):
    """Unregisters ``names`` of user-defined data columns.
    
    注销用户自定义的数据列名。
    """
    StaticScope.instance().unregister_custom_cols(names, *args)


def param(name: str, value: Optional[Any] = _EMPTY_PARAM) -> Optional[Any]:
    """Get or set a global parameter.
    
    获取或设置一个全局参数。
    """
    return StaticScope.instance().param(name, value)


class ColumnScope:
    """Caches and retrieves column data queried from :class:`pandas.DataFrame`.
    
    缓存和检索从DataFrame查询的列数据。

    Args:
        df: :class:`pandas.DataFrame` containing the column data.
            包含列数据的DataFrame。
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df.sort_index()
        self._symbols = frozenset(df.index.get_level_values(0).unique())
        self._sym_cols: dict[str, dict[str, Optional[NDArray]]] = defaultdict(
            dict
        )

    def fetch_dict(
        self,
        symbol: str,
        names: Iterable[str],
        end_index: Optional[int] = None,
    ) -> dict[str, Optional[NDArray]]:
        r"""Fetches a ``dict`` of column data for ``symbol``.
        
        获取指定交易品种的列数据字典。

        Args:
            symbol: Ticker symbol to query.
                要查询的交易品种代码。
            names: Names of columns to query.
                要查询的列名。
            end_index: Truncates column values (exclusive). If ``None``, then
                column values are not truncated.
                列值截断位置（不包含）。如果为None，则不截断列值。

        Returns:
            ``dict`` mapping column names to :class:`numpy.ndarray`\ s of
            column values.
            将列名映射到列值数组的字典。
        """
        result: dict[str, Optional[NDArray]] = {}
        if not names:
            return result
        sym_dfs: dict[str, pd.DataFrame] = {}
        for name in names:
            if symbol in self._sym_cols and name in self._sym_cols[symbol]:
                result[name] = self._sym_cols[symbol][name]
                if result[name] is not None:
                    result[name] = result[name][:end_index]  # type: ignore[index]
                continue
            if symbol in sym_dfs:
                sym_df = sym_dfs[symbol]
            else:
                if symbol not in self._symbols:
                    raise ValueError(f"Symbol not found: {symbol}.")
                sym_df = self._df.loc[pd.IndexSlice[symbol, :]].reset_index()
                sym_dfs[symbol] = sym_df
            if name not in sym_df.columns:
                self._sym_cols[symbol][name] = None
                result[name] = None
                continue
            array = sym_df[name].to_numpy()
            self._sym_cols[symbol][name] = array
            result[name] = array[:end_index]
        return result

    def fetch(
        self, symbol: str, name: str, end_index: Optional[int] = None
    ) -> Optional[NDArray]:
        """Fetches a :class:`numpy.ndarray` of column data for ``symbol``.
        
        获取指定交易品种的列数据数组。

        Args:
            symbol: Ticker symbol to query.
                要查询的交易品种代码。
            name: Name of column to query.
                要查询的列名。
            end_index: Truncates column values (exclusive). If ``None``, then
                column values are not truncated.
                列值截断位置（不包含）。如果为None，则不截断列值。

        Returns:
            :class:`numpy.ndarray` of column data for every bar until
            ``end_index`` (when specified).
            包含直到end_index的每个柱的列数据的数组。
        """
        result = self.fetch_dict(symbol, (name,), end_index)
        return result.get(name, None)

    def bar_data_from_data_columns(
        self, symbol: str, end_index: int
    ) -> BarData:
        """Returns a new :class:`pybroker.common.BarData` instance containing
        column data of default and custom data columns registered with
        :class:`.StaticScope`.
        
        返回一个新的BarData实例，包含在StaticScope中注册的默认和自定义数据列的列数据。

        Args:
            symbol: Ticker symbol to query.
                要查询的交易品种代码。
            end_index: Truncates column values (exclusive). If ``None``, then
                column values are not truncated.
                列值截断位置（不包含）。如果为None，则不截断列值。
        """
        static_scope = StaticScope.instance()
        default_col_data = self.fetch_dict(
            symbol, static_scope.default_data_cols, end_index
        )
        custom_col_data = self.fetch_dict(
            symbol, static_scope.custom_data_cols, end_index
        )
        return BarData(
            **default_col_data,  # type: ignore[arg-type]
            **custom_col_data,  # type: ignore[arg-type]
        )


class IndicatorScope:
    """Caches and retrieves :class:`pybroker.indicator.Indicator` data.
    
    缓存和检索指标数据。

    Args:
        indicator_data: :class:`Mapping` of
            :class:`pybroker.common.IndicatorSymbol` pairs to ``pandas.Series``
            of :class:`pybroker.indicator.Indicator` values.
            指标符号对到指标值Series的映射。
        filter_dates: Filters :class:`pybroker.indicator.Indicator` data on
            :class:`Sequence` of dates.
            用于过滤指标数据的日期序列。
    """

    def __init__(
        self,
        indicator_data: Mapping[IndicatorSymbol, pd.Series],
        filter_dates: Sequence[np.datetime64],
    ):
        self._indicator_data = indicator_data
        self._filter_dates = filter_dates
        self._sym_inds: dict[IndicatorSymbol, NDArray[np.float64]] = {}

    def fetch(
        self, symbol: str, name: str, end_index: Optional[int] = None
    ) -> NDArray[np.float64]:
        """Fetches :class:`pybroker.indicator.Indicator` data.
        
        获取指标数据。

        Args:
            symbol: Ticker symbol to query.
                要查询的交易品种代码。
            name: Name of :class:`pybroker.indicator.Indicator` to query.
                要查询的指标名称。
            end_index: Truncates the array of
                :class:`pybroker.indicator.Indicator` data returned
                (exclusive). If ``None``, then indicator data is not truncated.
                返回的指标数据数组的截断位置（不包含）。如果为None，则不截断指标数据。

        Returns:
            :class:`numpy.ndarray` of :class:`pybroker.indicator.Indicator`
            data for every bar until ``end_index`` (when specified).
            包含直到end_index的每个柱的指标数据的数组。
        """
        ind_sym = IndicatorSymbol(name, symbol)
        if ind_sym in self._sym_inds:
            return self._sym_inds[ind_sym][:end_index]
        if ind_sym not in self._indicator_data:
            raise ValueError(f"Indicator {name!r} not found for {symbol}.")
        ind_series = self._indicator_data[ind_sym]
        ind_data = ind_series[ind_series.index.isin(self._filter_dates)].values
        self._sym_inds[ind_sym] = ind_data
        return ind_data[:end_index]


class ModelInputScope:
    r"""Caches and retrieves model input data.
    
    缓存和检索模型输入数据。

    Args:
        col_scope: :class:`.ColumnScope`.
            列作用域对象。
        ind_scope: :class:`.IndicatorScope`.
            指标作用域对象。
        models: :class:`Mapping` of
            :class:`pybroker.common.ModelSymbol` pairs to
            :class:`pybroker.common.TrainedModel`\ s.
            模型符号对到训练好的模型的映射。
    """

    def __init__(
        self,
        col_scope: ColumnScope,
        ind_scope: IndicatorScope,
        models: Mapping[ModelSymbol, TrainedModel],
    ):
        self._col_scope = col_scope
        self._ind_scope = ind_scope
        self._models = models
        self._sym_inputs: dict[ModelSymbol, pd.DataFrame] = {}
        self._scope = StaticScope.instance()

    def fetch(
        self, symbol: str, name: str, end_index: Optional[int] = None
    ) -> pd.DataFrame:
        """Fetches model input data.
        
        获取模型输入数据。

        Args:
            symbol: Ticker symbol to query.
                要查询的交易品种代码。
            name: Name of :class:`pybroker.model.ModelSource` to query input
                data.
                要查询输入数据的模型源名称。
            end_index: Truncates the array of model input data returned
                (exclusive). If ``None``, then model input data is not
                truncated.
                返回的模型输入数据的截断位置（不包含）。如果为None，则不截断模型输入数据。

        Returns:
            :class:`numpy.ndarray` of model input data for every bar until
            ``end_index`` (when specified).
            包含直到end_index的每个柱的模型输入数据的DataFrame。
        """
        model_sym = ModelSymbol(name, symbol)
        if model_sym in self._sym_inputs:
            df = self._sym_inputs[model_sym]
            return df if end_index is None else df.loc[: end_index - 1]
        input_ = {}
        for col in self._scope.all_data_cols:
            data = self._col_scope.fetch(symbol, col)
            if data is not None:
                input_[col] = data
        if not self._scope.has_model_source(name):
            raise ValueError(f"Model {name!r} not found.")
        for ind_name in self._scope.get_indicator_names(name):
            input_[ind_name] = self._ind_scope.fetch(symbol, ind_name)
        df = pd.DataFrame.from_dict(input_)
        if model_sym not in self._models:
            raise ValueError(f"Model {name!r} not found for {symbol}.")
        trained_model = self._models[model_sym]
        if trained_model.input_cols is not None:
            for input_col in trained_model.input_cols:
                if input_col not in df.columns:
                    raise ValueError(
                        f"Missing column {input_col!r} for input data to "
                        f"model {model_sym.model_name!r}."
                    )
            df = df[list(trained_model.input_cols)]
        model_source = self._scope.get_model_source(name)
        if not trained_model.input_cols or model_source._input_data_fn:
            df = model_source.prepare_input_data(df)
        self._sym_inputs[model_sym] = df
        return df if end_index is None else df.loc[: end_index - 1]


class PredictionScope:
    r"""Caches and retrieves model predictions.
    
    缓存和检索模型预测。

    Args:
        models: :class:`Mapping` of
            :class:`pybroker.common.ModelSymbol` pairs to
            :class:`pybroker.common.TrainedModel`\ s.
            模型符号对到训练好的模型的映射。
        input_scope: :class:`.ModelInputScope`.
            模型输入作用域对象。
    """

    def __init__(
        self,
        models: Mapping[ModelSymbol, TrainedModel],
        input_scope: ModelInputScope,
    ):
        self._models = models
        self._input_scope = input_scope
        self._sym_preds: dict[ModelSymbol, NDArray] = {}

    def fetch(
        self, symbol: str, name: str, end_index: Optional[int] = None
    ) -> NDArray:
        """Fetches model predictions.
        
        获取模型预测。

        Args:
            symbol: Ticker symbol to query.
                要查询的交易品种代码。
            name: Name of :class:`pybroker.model.ModelSource` that made the
                predictions.
                做出预测的模型源名称。
            end_index: Truncates the array of predictions returned (exclusive).
                If ``None``, then predictions are not truncated.
                返回的预测数组的截断位置（不包含）。如果为None，则不截断预测。

        Returns:
            :class:`numpy.ndarray` of model predictions for every bar until
            ``end_index`` (when specified).
            包含直到end_index的每个柱的模型预测的数组。
        """
        model_sym = ModelSymbol(name, symbol)
        if model_sym in self._sym_preds:
            return self._sym_preds[model_sym][:end_index]
        input_ = self._input_scope.fetch(symbol, name)
        if input_.empty:
            raise ValueError(
                f"No input data found for model {name!r}. Consider "
                "passing input_data_fn to pybroker#model() if custom columns "
                "were registered."
            )
        if model_sym not in self._models:
            raise ValueError(f"Model {name!r} not found for {symbol}.")
        trained_model = self._models[model_sym]
        if trained_model.predict_fn is not None:
            pred = trained_model.predict_fn(trained_model.instance, input_)
        else:
            predict_fn = getattr(trained_model.instance, "predict", None)
            if predict_fn is not None and callable(predict_fn):
                pred = trained_model.instance.predict(input_)
            else:
                raise ValueError(
                    f"Model instance trained for {model_sym.model_name!r} "
                    "does not define a predict function. Please pass a "
                    "predict_fn to pybroker.model()."
                )
        if len(pred.shape) > 1:
            pred = np.squeeze(pred)
        self._sym_preds[model_sym] = pred
        return pred[:end_index]


class PriceScope:
    """Retrieves most recent prices.
    
    检索最近的价格。
    """

    def __init__(
        self,
        col_scope: ColumnScope,
        sym_end_index: Mapping[str, int],
        round_fill_price: bool,
    ):
        self._col_scope = col_scope
        self._sym_end_index = sym_end_index
        self._round_fill_price = round_fill_price

    def fetch(
        self,
        symbol: str,
        price: Union[
            int,
            float,
            np.floating,
            Decimal,
            PriceType,
            Callable[[str, BarData], Union[int, float, Decimal]],
        ],
    ) -> Decimal:
        """获取特定价格类型的最近价格并转换为Decimal类型返回。
        
        根据提供的价格类型(如开盘价、收盘价等)或自定义价格计算函数，获取最新的价格数据。
        """
        end_index = self._sym_end_index[symbol]
        price_type = type(price)
        fill_price = None
        if price_type == PriceType:
            if price == PriceType.OPEN:
                open_ = self._col_scope.fetch(
                    symbol, DataCol.OPEN.value, end_index
                )
                if open_ is None:
                    raise ValueError("Open price not found.")
                fill_price = open_[-1]
            elif price == PriceType.HIGH:
                high = self._col_scope.fetch(
                    symbol, DataCol.HIGH.value, end_index
                )
                if high is None:
                    raise ValueError("High price not found.")
                fill_price = high[-1]
            elif price == PriceType.LOW:
                low = self._col_scope.fetch(
                    symbol, DataCol.LOW.value, end_index
                )
                if low is None:
                    raise ValueError("Low price not found.")
                fill_price = low[-1]
            elif price == PriceType.CLOSE:
                close = self._col_scope.fetch(
                    symbol, DataCol.CLOSE.value, end_index
                )
                if close is None:
                    raise ValueError("Close price not found.")
                fill_price = close[-1]
            elif price == PriceType.MIDDLE:
                low = self._col_scope.fetch(
                    symbol, DataCol.LOW.value, end_index
                )
                if low is None:
                    raise ValueError("Low price not found.")
                high = self._col_scope.fetch(
                    symbol, DataCol.HIGH.value, end_index
                )
                if high is None:
                    raise ValueError("High price not found.")

                fill_price = low[-1] + (high[-1] - low[-1]) / 2.0
            elif price == PriceType.AVERAGE:
                open_ = self._col_scope.fetch(
                    symbol, DataCol.OPEN.value, end_index
                )
                if open_ is None:
                    raise ValueError("Open price not found.")
                high = self._col_scope.fetch(
                    symbol, DataCol.HIGH.value, end_index
                )
                if high is None:
                    raise ValueError("High price not found.")
                low = self._col_scope.fetch(
                    symbol, DataCol.LOW.value, end_index
                )
                if low is None:
                    raise ValueError("Low price not found.")
                close = self._col_scope.fetch(
                    symbol, DataCol.CLOSE.value, end_index
                )
                if close is None:
                    raise ValueError("Close price not found.")
                fill_price = (open_[-1] + low[-1] + high[-1] + close[-1]) / 4.0
            else:
                raise ValueError(f"Unknown price: {price_type}")
        elif (
            price_type is float
            or price_type is int
            or isinstance(price, np.floating)
            or isinstance(price, Decimal)
        ):
            fill_price = price
        elif callable(price):
            bar_data = self._col_scope.bar_data_from_data_columns(
                symbol, self._sym_end_index[symbol]
            )
            fill_price = price(symbol, bar_data)
        else:
            raise ValueError(f"Unknown price: {price_type}")
        if self._round_fill_price:
            fill_price = round(fill_price, 2)
        return to_decimal(fill_price)


class PendingOrder(NamedTuple):
    """Holds data for a pending order.
    
    保存挂单数据。

    Attributes:
        id: Unique ID.
            唯一标识符。
        type: Type of order, either ``buy`` or ``sell``.
            订单类型，"buy"或"sell"。
        symbol: Ticker symbol of the order.
            订单的交易品种代码。
        created: Date the order was created.
            订单创建日期。
        exec_date: Date the order will be executed.
            订单执行日期。
        shares: Number of shares to be bought or sold.
            要买入或卖出的股数。
        limit_price: Limit price to use for the order.
            订单的限价。
        fill_price: Price that the order will be filled at.
            订单将被成交的价格。
    """

    id: int
    type: Literal["buy", "sell"]
    symbol: str
    created: np.datetime64
    exec_date: np.datetime64
    shares: Decimal
    limit_price: Optional[Decimal]
    fill_price: Union[
        int,
        float,
        np.floating,
        Decimal,
        PriceType,
        Callable[[str, BarData], Union[int, float, Decimal]],
    ]


class PendingOrderScope:
    r"""Stores :class:`.PendingOrder`\ s
    
    存储挂单对象的容器。
    """

    _order_id: int = 0

    def __init__(self):
        self._orders: dict[int, PendingOrder] = {}
        self._sym_orders: dict[str, set[PendingOrder]] = defaultdict(set)

    def contains(self, order_id: int) -> bool:
        """Returns whether a :class:`.PendingOrder` exists with
        ``order_id``.
        
        检查是否存在指定订单ID的挂单。
        """
        return order_id in self._orders

    def add(
        self,
        type: Literal["buy", "sell"],
        symbol: str,
        created: np.datetime64,
        exec_date: np.datetime64,
        shares: Decimal,
        limit_price: Optional[Decimal],
        fill_price: Union[
            int,
            float,
            np.floating,
            Decimal,
            PriceType,
            Callable[[str, BarData], Union[int, float, Decimal]],
        ],
    ) -> int:
        """Creates a :class:`.PendingOrder`.
        
        创建一个挂单对象。

        Args:
            type: Type of order, either ``buy`` or ``sell``.
                订单类型，"buy"或"sell"。
            symbol: Ticker symbol of the order.
                订单的交易品种代码。
            created: Date the order was created.
                订单创建日期。
            exec_date: Date the order will be executed.
                订单执行日期。
            shares: Number of shares to be bought or sold.
                要买入或卖出的股数。
            limit_price: Limit price to use for the order.
                订单的限价。
            fill_price: Price that the order will be filled at.
                订单将被成交的价格。

        Returns:
            ID of the :class:`.PendingOrder`.
            挂单对象的ID。
        """
        self._order_id += 1
        order = PendingOrder(
            id=self._order_id,
            type=type,
            symbol=symbol,
            created=created,
            exec_date=exec_date,
            shares=shares,
            limit_price=limit_price,
            fill_price=fill_price,
        )
        self._orders[self._order_id] = order
        self._sym_orders[symbol].add(order)
        return order.id

    def remove(self, order_id: int) -> bool:
        """Removes a :class:`.PendingOrder` with ``order_id```.
        
        移除指定订单ID的挂单。
        """
        if order_id in self._orders:
            order = self._orders[order_id]
            del self._orders[order_id]
            if (
                order.symbol in self._sym_orders
                and order in self._sym_orders[order.symbol]
            ):
                self._sym_orders[order.symbol].remove(order)
            return True
        return False

    def remove_all(self, symbol: Optional[str] = None):
        r"""Removes all :class:`.PendingOrder`\ s.
        
        移除所有挂单，如果指定了交易品种，则只移除该交易品种的挂单。
        """
        if symbol is None:
            cancel_ids = tuple(self._orders.keys())
            for order_id in cancel_ids:
                self.remove(order_id)
        elif symbol in self._sym_orders:
            cancel_ids = tuple(order.id for order in self._sym_orders[symbol])
            for order_id in cancel_ids:
                self.remove(order_id)

    def orders(self, symbol: Optional[str] = None) -> Iterable[PendingOrder]:
        r"""Returns an :class:`Iterable` of :class:`.PendingOrder`\ s.
        
        返回挂单对象的可迭代集合，如果指定了交易品种，则只返回该交易品种的挂单。
        """
        if symbol is None:
            return self._orders.values()
        else:
            if symbol not in self._sym_orders:
                return []
            return self._sym_orders[symbol]


def get_signals(
    symbols: Iterable[str],
    col_scope: ColumnScope,
    ind_scope: IndicatorScope,
    pred_scope: PredictionScope,
) -> dict[str, pd.DataFrame]:
    r"""Retrieves dictionary of :class:`pandas.DataFrame`\ s
    containing bar data, indicator data, and model predictions for each symbol.
    
    检索包含每个交易品种的柱数据、指标数据和模型预测的DataFrame字典。
    
    这个函数收集交易信号数据，将原始价格数据、技术指标和模型预测整合在一起，
    便于后续的交易策略使用。
    """
    static_scope = StaticScope.instance()
    cols = static_scope.all_data_cols
    inds = static_scope._indicators.keys()
    models = static_scope._model_sources.keys()
    dates = col_scope._df.index.get_level_values(1)
    dfs: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        data = {DataCol.DATE.value: dates}
        for col in cols:
            data[col] = col_scope.fetch(sym, col)
        for ind in inds:
            try:
                data[ind] = ind_scope.fetch(sym, ind)
            except ValueError:
                continue
        for model in models:
            try:
                data[f"{model}_pred"] = pred_scope.fetch(sym, model)
            except ValueError:
                continue
        dfs[sym] = pd.DataFrame(data)
    return dfs
