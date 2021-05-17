import itertools
import pandas as pd

from .base import BaseFeatureTransformer

class LagGroupbyTransformer(BaseFeatureTransformer):
    def __init__(self, value_cols, time_cols, group_col, aggs, lags):
        super(LagGroupbyTransformer, self).__init__()
        self.value_cols = value_cols
        self.time_cols = time_cols
        self.group_col = group_col
        self.aggs = aggs
        self.lags = lags


    def transform(self, df):
        for v, lag, agg, time_col in itertools.product(self.value_cols, self.lags, self.aggs, self.time_cols):
            col_name = f'fe_lag_{lag}_{self.group_col}_{v}_{agg}'
            tmp = df[[self.group_col, time_col, v]].dropna()
            group = tmp.groupby([self.group_col, time_col])[[v]].agg(agg).groupby(level=[0]).shift(lag).reset_index()
            group = group.rename(columns={v: col_name})
            df = df.merge(group, on=[self.group_col, time_col], how='left')

        return df


class LagRollingGroupbyTransformer(BaseFeatureTransformer):
    def __init__(self, value_cols, time_cols, group_col, aggs, lags, windows):
        super(LagRollingGroupbyTransformer, self).__init__()
        self.value_cols = value_cols
        self.time_cols = time_cols
        self.group_col = group_col
        self.aggs = aggs
        self.lags = lags
        self.windows = windows


    def transform(self, df):
        for v, lag, agg, time_col, w in itertools.product(self.value_cols, self.lags, self.aggs, self.time_cols, self.windows):
            col_name = f'fe_lag_{lag}_{self.group_col}_roll_{w}_{v}_{agg}'
            tmp = df[[self.group_col, time_col, v]].dropna()
            group = tmp.groupby([self.group_col, time_col])[v].agg(agg).groupby(level=[0]).shift(lag).rolling(window=w).mean()
            group = group.reset_index()
            group = group.rename(columns={v: col_name})
            df = df.merge(group, on=[self.group_col, time_col], how='left')

        return df
