
import pandas as pd
import numpy as np
from utils import load_data
from utils import unpickle

import mojimoji
import re

from utils.preprocess import normalize_area, normalize_moyori, convert_wareki_to_seireki, convert_madori
# from features import CategoryEncoder, FrequencyEncoder, GroupbyTransformer, LagTransformer, TextVectorizer

pd.set_option('display.max_rows', None)
data_dir = './input'
id_col = 'ID'
tar_col = '取引価格（総額）_log'
df = load_data(data_dir, down_sample=1.0, seed=42, id_col=id_col, target_col=tar_col)


# TODO 不要なカラムを削除
del_cols = ['種類', '地域', '市区町村コード', '土地の形状', '間口', '延床面積（㎡）',
            '前面道路：方位', '前面道路：種類', '前面道路：幅員（ｍ）', '取引の事情等']
df = df.drop(del_cols, axis=1)

# TODO 欠損処理
for c in ['地区名', '最寄駅：名称', '用途', '今後の利用目的', '建物の構造', '間取り', '都市計画', '改装']:
    df[c].fillna('不明', inplace=True)

# TODO 地区名、最寄り
for c in ['地区名', '最寄駅：名称']:
    df[c] = df[c].apply(lambda x: mojimoji.zen_to_han(x, kana=False))
# ()を削除
df['最寄駅：名称_1'] = df['最寄駅：名称'].apply(lambda x: x.split('(')[0])


def rep(x):
    if '(' in x:
        return x.split('(')[1].split(')')[0]
    else:
        x
df['最寄駅：名称_2'] = df['最寄駅：名称'].apply(rep)

# df = df.head(100)

v = df['最寄駅：名称_2'].value_counts()

