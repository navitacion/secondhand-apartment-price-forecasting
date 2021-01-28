import re
import math
import numpy as np
import mojimoji

def normalize_moyori(df):
    rep_dict = {
        '30分?60分': 45,
        '1H?1H30': 75,
        '1H30?2H': 105,
        '2H?': 120
    }

    df['最寄駅：距離（分）'] = df['最寄駅：距離（分）'].replace(rep_dict)
    df['最寄駅：距離（分）'] = df['最寄駅：距離（分）'].astype(float)

    return df


def normalize_area(df):
    df['面積（㎡）'] = df['面積（㎡）'].apply(lambda x: int(re.sub('m\^2未満|㎡以上', '', str(x))))
    df['面積（㎡）'] = df['面積（㎡）'].astype(int)
    return df


def convert_wareki_to_seireki(wareki):
    seireki = None
    if wareki is np.nan:
        return wareki
    if wareki == '戦前':
        return 1945
    value = wareki[2:-1]
    if value == '元':
        value = 1
    else:
        value = int(value)
    if '昭和' in wareki:
        seireki = 1925+value
    elif '平成' in wareki:
        seireki = 1988+value
    elif '令和' in wareki:
        seireki = 2018+value
    return seireki


def convert_madori(x):
    if x is np.nan:
        return x
    else:
        return mojimoji.zen_to_han(x, kana=False)



def preprocess_madori(df):
    df['間取り'] = df['間取り'].apply(convert_madori)
    for s in ['L', 'D', 'K', 'S']:
        df[f'fe_count_間取り_{s}'] = df['間取り'].apply(lambda x: x.count(s))

    # 間取りの最初の数字をとってくる
    df['fe_count_間取り_prefix'] = df['間取り'].apply(lambda x: x[:1])
    df['fe_count_間取り_prefix'] = df['fe_count_間取り_prefix'].apply(lambda x: x if x.isdigit() else 1)
    df['fe_count_間取り_prefix'] = df['fe_count_間取り_prefix'].astype(int)

    # 間取りから部屋の数を計算　+Sがあるものは別途算出
    df['fe_count_room'] = df['fe_count_間取り_prefix'] + 1
    df['flag'] = df['間取り'].apply(lambda x: 1 if '+' in x else 0)
    df['fe_count_room'] = df['fe_count_room'] + df['flag']
    del df['flag']
    # 特殊な表記
    for c in ['オープンフロア', 'スタジオ', 'メゾネット']:
        df[f'fe_is_間取り_{c}'] = df['間取り'].apply(lambda x: 1 if x == c else 0)

    del df['間取り']

    return df