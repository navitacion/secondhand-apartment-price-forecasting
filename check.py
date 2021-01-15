
import pandas as pd

from utils import load_data
from utils import unpickle


from features import CategoryEncoder, FrequencyEncoder, GroupbyTransformer, LagTransformer, TextVectorizer

pd.set_option('display.max_rows', None)

df = pd.read_csv('./input/train.csv')

print(df.shape)
print(df.head())

print(df.columns)