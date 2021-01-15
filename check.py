
import pandas as pd

from utils import load_data
from utils import unpickle

from utils.preprocess import normalize_area, normalize_moyori, convert_wareki_to_seireki
from features import CategoryEncoder, FrequencyEncoder, GroupbyTransformer, LagTransformer, TextVectorizer

pd.set_option('display.max_rows', None)
data_dir = './input'
id_col = 'ID'
tar_col = '取引価格（総額）_log'
df = load_data(data_dir, down_sample=1.0, seed=42, id_col=id_col, target_col=tar_col)

del_cols = ['種類', '地域', '市区町村コード', '土地の形状', '間口', '延床面積（㎡）',
            '前面道路：方位', '前面道路：種類', '前面道路：幅員（ｍ）', '取引の事情等']
df = df.drop(del_cols, axis=1)

# TODO 面積
df = normalize_area(df)

# TODO 最寄駅：距離（分）
df = normalize_moyori(df)

# TODO 和暦→西暦
df['建築年'] = df['建築年'].apply(lambda x: convert_wareki_to_seireki(x))

# TODO 取引時点→`year`, `quarter`
df['year'] = df['取引時点'].apply(lambda x: x[:4]).astype(int)
df['quarter'] = df['取引時点'].apply(lambda x: x[6:7]).astype(int)
del df['取引時点']

# TODO 用途をOnehot
tmp = df['用途'].str.get_dummies(sep='、')
tmp.columns = [f'用途_{c}' for c in tmp.columns]
df = pd.concat([df, tmp], axis=1)
del tmp, df['用途']

print(df.dtypes)

print(df['建物の構造'].value_counts())