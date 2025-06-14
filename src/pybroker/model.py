"""Contains model related functionality."""  # 包含模型相关功能

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import functools
import pandas as pd
from pybroker.cache import CacheDateFields, ModelCacheKey
from pybroker.common import (
    DataCol,
    IndicatorSymbol,
    ModelSymbol,
    TrainedModel,
    get_unique_sorted_dates,
    to_datetime,
)
from pybroker.indicator import Indicator
from pybroker.scope import StaticScope
from dataclasses import asdict
from datetime import datetime
from numpy.typing import NDArray
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    NamedTuple,
    Optional,
    Union,
)


class ModelSource:
    r"""模型源的基类。模型源通过训练或加载预训练模型来提供模型实例。
    
    参数:
        name: 模型名称
        indicator_names: 作为模型特征的指标名称的可迭代对象
        input_data_fn: 用于预处理传递给模型进行预测的输入数据的可调用函数
                      如果设置，input_data_fn将使用包含所有测试数据的DataFrame调用
        predict_fn: 覆盖模型默认predict函数的可调用函数
                   如果设置，predict_fn将使用训练好的模型和包含所有测试数据的DataFrame调用
        kwargs: 额外参数的字典
    """

    def __init__(
        self,
        name: str,  # 模型名称
        indicator_names: Iterable[str],  # 指标名称
        input_data_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]],  # 输入数据预处理函数
        predict_fn: Optional[Callable[[Any, pd.DataFrame], NDArray]],  # 预测函数
        kwargs: dict[str, Any],  # 额外参数
    ):
        self.name = name
        self.indicators = tuple(indicator_names)
        self._input_data_fn = input_data_fn
        self._predict_fn = predict_fn
        self._kwargs = kwargs

    def prepare_input_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备用于模型预测的输入数据DataFrame
        
        如果设置了input_data_fn，则使用它预处理输入数据
        如果未设置，则使用df中的指标列作为输入特征
        """
        if df.empty:
            return df
        if self._input_data_fn is None:
            df_cols = frozenset(df.columns)
            for ind_name in self.indicators:
                if ind_name not in df_cols:
                    raise ValueError(
                        f"Indicator {ind_name!r} not found in DataFrame."
                    )
            return df[[*self.indicators]]
        return self._input_data_fn(df)


class ModelLoader(ModelSource):
    r"""加载预训练模型的类
    
    参数:
        name: 模型名称
        load_fn: 用于加载并返回预训练模型的可调用函数
                预期返回训练好的模型实例，或包含模型实例和用作输入的列名的元组
        indicator_names: 作为模型特征的指标名称的可迭代对象
        input_data_fn: 用于预处理传递给模型进行预测的输入数据的可调用函数
        predict_fn: 覆盖模型默认predict函数的可调用函数
        kwargs: 传递给load_fn的关键字参数字典
    """

    def __init__(
        self,
        name: str,  # 模型名称
        load_fn: Callable[..., Union[Any, tuple[Any, Iterable[str]]]],  # 加载函数
        indicator_names: Iterable[str],  # 指标名称
        input_data_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]],  # 输入数据预处理函数
        predict_fn: Optional[Callable[[Any, pd.DataFrame], NDArray]],  # 预测函数
        kwargs: dict[str, Any],  # 额外参数
    ):
        super().__init__(
            name, indicator_names, input_data_fn, predict_fn, kwargs
        )
        self._load_fn = functools.partial(load_fn, **kwargs)

    def __call__(
        self, symbol: str, train_start_date: datetime, train_end_date: datetime
    ) -> Union[Any, tuple[Any, Iterable[str]]]:
        """加载预训练模型
        
        参数:
            symbol: 加载预训练模型的股票代码
            train_start_date: 训练窗口的开始日期
            train_end_date: 训练窗口的结束日期
            
        返回:
            预训练模型
        """
        return self._load_fn(symbol, train_start_date, train_end_date)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"ModelLoader({self.name!r}, {self._kwargs})"


class ModelTrainer(ModelSource):
    r"""训练模型的类
    
    参数:
        name: 模型名称
        train_fn: 用于训练并返回模型的可调用函数
                 预期返回训练好的模型实例，或包含模型实例和用作输入的列名的元组
        indicator_names: 作为模型特征的指标名称的可迭代对象
        input_data_fn: 用于预处理传递给模型进行预测的输入数据的可调用函数
        predict_fn: 覆盖模型默认predict函数的可调用函数
        kwargs: 传递给train_fn的关键字参数字典
    """

    def __init__(
        self,
        name: str,  # 模型名称
        train_fn: Callable[..., Union[Any, tuple[Any, Iterable[str]]]],  # 训练函数
        indicator_names: Iterable[str],  # 指标名称
        input_data_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]],  # 输入数据预处理函数
        predict_fn: Optional[Callable[[Any, pd.DataFrame], NDArray]],  # 预测函数
        kwargs: dict[str, Any],  # 额外参数
    ):
        super().__init__(
            name, indicator_names, input_data_fn, predict_fn, kwargs
        )
        self._train_fn = functools.partial(train_fn, **kwargs)

    def __call__(
        self, symbol: str, train_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> Union[Any, tuple[Any, Iterable[str]]]:
        """训练模型
        
        参数:
            symbol: 模型的股票代码（每个股票代码训练一个模型）
            train_data: 训练数据
            test_data: 测试数据
            
        返回:
            训练好的模型
        """
        return self._train_fn(symbol, train_data, test_data)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"ModelTrainer({self.name!r}, {self._kwargs})"


def model(
    name: str,  # 模型名称
    fn: Callable[..., Union[Any, tuple[Any, Iterable[str]]]],  # 函数
    indicators: Optional[Iterable[Indicator]] = None,  # 指标
    input_data_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,  # 输入数据预处理函数
    predict_fn: Optional[Callable[[Any, pd.DataFrame], NDArray]] = None,  # 预测函数
    pretrained: bool = False,  # 是否是预训练模型
    **kwargs,  # 额外参数
) -> ModelSource:
    r"""创建ModelSource实例并全局注册
    
    参数:
        name: 全局引用模型的名称
        fn: 用于训练或加载模型的可调用函数
        indicators: 用作模型特征的指标的可迭代对象
        input_data_fn: 用于预处理传递给模型进行预测的输入数据的可调用函数
        predict_fn: 覆盖模型默认predict函数的可调用函数
        pretrained: 如果为True，创建ModelLoader实例；否则创建ModelTrainer实例
        **kwargs: 传递给fn的额外参数
        
    返回:
        ModelSource实例
    """
    scope = StaticScope.instance()
    ind_names = []
    if indicators:
        for indicator in indicators:
            if not isinstance(indicator, Indicator):
                raise TypeError(f"Expected Indicator, got {type(indicator)}.")
            ind_names.append(indicator.name)
    src: ModelSource
    if pretrained:
        src = ModelLoader(
            name, fn, ind_names, input_data_fn, predict_fn, kwargs
        )
    else:
        src = ModelTrainer(
            name, fn, ind_names, input_data_fn, predict_fn, kwargs
        )
    scope.set_model_source(src)
    return src


class CachedModel(NamedTuple):
    """存储缓存的模型数据
    
    属性:
        model: 训练好的模型实例
        input_cols: 用作模型预测输入的列名称
    """

    model: Any  # 模型实例
    input_cols: Optional[tuple[str]]  # 输入列名称


class ModelsMixin:
    """实现模型相关功能的混入类"""

    def train_models(
        self,
        model_syms: Iterable[ModelSymbol],  # 模型符号迭代器
        train_data: pd.DataFrame,  # 训练数据
        test_data: pd.DataFrame,  # 测试数据
        indicator_data: Mapping[IndicatorSymbol, pd.Series],  # 指标数据映射
        cache_date_fields: CacheDateFields,  # 缓存日期字段
    ) -> dict[ModelSymbol, TrainedModel]:
        """训练所提供的ModelSymbol对的模型
        
        参数:
            model_syms: ModelSymbol对的可迭代对象
            train_data: 训练数据
            test_data: 测试数据
            indicator_data: 将IndicatorSymbol对映射到指标值Series的字典
            cache_date_fields: 用于键缓存数据的日期字段
            
        返回:
            将每个ModelSymbol对映射到TrainedModel的字典
        """
        if not model_syms or train_data.empty:
            return {}
        scope = StaticScope.instance()
        # 检查缓存的模型
        result, left_to_train = self._get_cached_models(
            model_syms, cache_date_fields
        )
        if not left_to_train:
            return result
        # 准备训练和测试数据
        train_dates = get_unique_sorted_dates(train_data[DataCol.DATE.value])
        for sym_model in left_to_train:
            sym, model_name = sym_model.symbol, sym_model.model_name
            sym_train_data = self._slice_by_symbol(sym, train_data)
            sym_test_data = self._slice_by_symbol(sym, test_data)
            if sym_train_data.empty:
                continue
            # 获取模型源并添加指标列
            src = scope.get_model_source(model_name)
            for ind_name in src.indicators:
                ind_sym = IndicatorSymbol(ind_name, sym)
                if ind_sym not in indicator_data:
                    raise ValueError(f"Missing indicator: {ind_sym}")
                ind_series = indicator_data[ind_sym]
                sym_train_data[ind_name] = sym_train_data[
                    DataCol.DATE.value
                ].map(ind_series)
                sym_test_data[ind_name] = sym_test_data[
                    DataCol.DATE.value
                ].map(ind_series)
            # 训练模型并处理结果
            ret_value = src(sym, sym_train_data, sym_test_data)
            model: Any
            input_cols: Optional[tuple[str]] = None
            if isinstance(ret_value, tuple):
                model, input_cols = ret_value[0], tuple(ret_value[1])
            else:
                model = ret_value
            # 缓存模型并添加到结果
            self._set_cached_model(
                model, input_cols, sym_model, cache_date_fields
            )
            result[sym_model] = TrainedModel(
                name=model_name,
                instance=model,
                predict_fn=src._predict_fn,
                input_cols=input_cols,
            )
        return result

    def _slice_by_symbol(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """按股票代码切片DataFrame
        
        参数:
            symbol: 股票代码
            df: 数据框
            
        返回:
            只包含指定股票代码数据的DataFrame
        """
        return df[df[DataCol.SYMBOL.value] == symbol].copy()

    def _get_cached_models(
        self,
        model_syms: Iterable[ModelSymbol],  # 模型符号迭代器
        cache_date_fields: CacheDateFields,  # 缓存日期字段
    ) -> tuple[dict[ModelSymbol, TrainedModel], list[ModelSymbol]]:
        """获取缓存的模型
        
        参数:
            model_syms: ModelSymbol对的可迭代对象
            cache_date_fields: 用于键缓存数据的日期字段
            
        返回:
            包含已缓存模型的字典和待训练模型符号列表的元组
        """
        scope = StaticScope.instance()
        result = {}
        left_to_train = []
        # 检查每个模型符号是否在缓存中
        for sym_model in model_syms:
            key = ModelCacheKey(
                model_name=sym_model.model_name,
                symbol=sym_model.symbol,
                train_start_date=cache_date_fields.train_start_date,
                train_end_date=cache_date_fields.train_end_date,
            )
            key_dict = asdict(key)
            cache_model = scope.get_cached_model(key_dict)
            if cache_model is None:
                left_to_train.append(sym_model)
                continue
            # 添加缓存的模型到结果
            src = scope.get_model_source(sym_model.model_name)
            result[sym_model] = TrainedModel(
                name=sym_model.model_name,
                instance=cache_model.model,
                predict_fn=src._predict_fn,
                input_cols=cache_model.input_cols,
            )
        return result, left_to_train

    def _set_cached_model(
        self,
        model: Any,  # 模型实例
        input_cols: Optional[tuple[str]],  # 输入列名称
        model_sym: ModelSymbol,  # 模型符号
        cache_date_fields: CacheDateFields,  # 缓存日期字段
    ):
        """缓存模型
        
        参数:
            model: 模型实例
            input_cols: 输入列名称
            model_sym: 模型符号
            cache_date_fields: 缓存日期字段
        """
        scope = StaticScope.instance()
        key = ModelCacheKey(
            model_name=model_sym.model_name,
            symbol=model_sym.symbol,
            train_start_date=cache_date_fields.train_start_date,
            train_end_date=cache_date_fields.train_end_date,
        )
        key_dict = asdict(key)
        scope.set_cached_model(key_dict, CachedModel(model, input_cols))

# 该模块提供了PyBroker的机器学习模型功能，支持模型的训练、加载和预测。
# 核心类ModelSource是所有模型源的基类，ModelTrainer用于训练模型，ModelLoader用于加载预训练模型。
# 提供了全局注册模型的函数model()，以便在策略中轻松使用。
# 支持模型的缓存机制，避免重复训练提高回测效率。
# ModelsMixin类实现了多个模型的批量训练和管理功能。
