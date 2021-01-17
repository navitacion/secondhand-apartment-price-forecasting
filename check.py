
import pandas as pd
import numpy as np
from utils import load_data
from utils import unpickle

import mojimoji

from utils.preprocess import normalize_area, normalize_moyori, convert_wareki_to_seireki, convert_madori
# from features import CategoryEncoder, FrequencyEncoder, GroupbyTransformer, LagTransformer, TextVectorizer

pd.set_option('display.max_rows', None)
data_dir = './input'
id_col = 'ID'
tar_col = '取引価格（総額）_log'
df = load_data(data_dir, down_sample=1.0, seed=42, id_col=id_col, target_col=tar_col)


# TODO 欠損処理
for c in ['最寄駅：名称', '用途', '今後の利用目的', '建物の構造', '間取り', '都市計画', '改装']:
    df[c].fillna('不明', inplace=True)


# TODO 間取り
df['間取り'] = df['間取り'].apply(convert_madori)
for s in ['L', 'D', 'K', 'S']:
    df[f'間取り_{s}'] = df['間取り'].apply(lambda x: x.count(s))

# 間取りの最初の数字をとってくる
df['間取り_suffix'] = df['間取り'].apply(lambda x: x[:1])
df['間取り_suffix'] = df['間取り_suffix'].apply(lambda x: x if x.isdigit() else 1)
df['間取り_suffix'] = df['間取り_suffix'].astype(int)

tar_cols = [c for c in df.columns if c.startswith('間取り_')]
df['fe_count_間取り'] = df[tar_cols].sum(axis=1)

for c in ['オープンフロア', 'スタジオ', 'メゾネット']:
    df[f'fe_is_間取り_{c}'] = df['間取り'].apply(lambda x: 1 if x == c else 0)


df = df.head(100)
