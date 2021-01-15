import re
import math
import numpy as np

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