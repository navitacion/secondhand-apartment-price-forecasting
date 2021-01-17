import os
import numpy as np
from comet_ml import Experiment
import hydra
from omegaconf import DictConfig
import mojimoji

import pandas as pd

from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_log_error, mean_absolute_error

from utils.utils import unpickle, to_pickle, seed_everything
from utils.preprocess import convert_wareki_to_seireki, normalize_area, normalize_moyori, convert_madori
from utils.data import load_data
from features import CategoryEncoder, FrequencyEncoder, GroupbyTransformer, TextVectorizer, PivotCountEncoder

from models.trainer import Trainer
from models.model import LGBMModel, CatBoostModel

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', None)


def MAE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    score = mean_absolute_error(y_true, y_pred)
    return score


def preprocessing(df, cfg):

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

    # TODO 面積
    df = normalize_area(df)

    # TODO 最寄駅：距離（分）
    df = normalize_moyori(df)

    # TODO 和暦→西暦
    df['建築年'] = df['建築年'].apply(convert_wareki_to_seireki)

    # TODO 取引時点→`year`, `quarter`
    df['year'] = df['取引時点'].apply(lambda x: x[:4]).astype(int)
    df['quarter'] = df['取引時点'].apply(lambda x: x[6:7]).astype(int)
    del df['取引時点']

    # TODO 用途, 建物の構造をOnehot
    for col in ['用途', '建物の構造']:
        tmp = df[col].str.get_dummies(sep='、')
        tmp.columns = [f'fe_is_{col}_{c}' for c in tmp.columns]
        df = pd.concat([df, tmp], axis=1)
        del tmp, df[col]

    # TODO 間取り
    df['間取り'] = df['間取り'].apply(convert_madori)
    for s in ['L', 'D', 'K', 'S']:
        df[f'fe_count_間取り_{s}'] = df['間取り'].apply(lambda x: x.count(s))

    # 間取りの最初の数字をとってくる
    df['fe_count_間取り_suffix'] = df['間取り'].apply(lambda x: x[:1])
    df['fe_count_間取り_suffix'] = df['fe_count_間取り_suffix'].apply(lambda x: x if x.isdigit() else 1)
    df['fe_count_間取り_suffix'] = df['fe_count_間取り_suffix'].astype(int)

    tar_cols = [c for c in df.columns if c.startswith('fe_count_間取り_')]
    df['fe_count_間取り_all'] = df[tar_cols].sum(axis=1)

    for c in ['オープンフロア', 'スタジオ', 'メゾネット']:
        df[f'fe_is_間取り_{c}'] = df['間取り'].apply(lambda x: 1 if x == c else 0)

    del df['間取り']

    # 地域
    df['fe_concat_都道府県_市区町村'] = df['都道府県名'] + df['市区町村名']
    df['fe_concat_都道府県_市区町村_地区'] = df['都道府県名'] + df['市区町村名'] + df['地区名']


    # ---------- 下記はfeature_enginneringを一通りやったあとの処理 -------------------------------------------
    not_use_col = ['ID', '取引価格（総額）_log', 'is_train']

    # カテゴリ変数に
    cols = [c for c in df.select_dtypes(include=['object', 'category']).columns if c not in not_use_col]
    cols += [c for c in df.columns if c.startswith('fe_is_')]
    cols += [c for c in df.columns if c.startswith('fe_cat_')]
    # 重複を削除
    cols = list(set(cols))
    cat_enc = CategoryEncoder(cols=cols)
    df = cat_enc.transform(df)

    return df


@hydra.main('config.yml')
def main(cfg: DictConfig):
    print('Nishika Second-hand Apartment Price Training')
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)
    data_dir = './input'

    seed_everything(cfg.data.seed)

    experiment = Experiment(api_key=cfg.exp.api_key,
                            project_name=cfg.exp.project_name,
                            auto_output_logging='simple')

    experiment.log_parameters(dict(cfg.data))

    # Config  ####################################################################################
    del_tar_col = []
    id_col = 'ID'
    tar_col = '取引価格（総額）_log'
    criterion = MAE
    cv = KFold(n_splits=cfg.data.n_splits, shuffle=True, random_state=cfg.data.seed)


    # Load Data  ####################################################################################
    if cfg.exp.use_pickle:
        # pickleから読み込み
        df = unpickle('./input/data.pkl')

    else:
        df = load_data(data_dir, down_sample=0.1, seed=cfg.data.seed, id_col=id_col, target_col=tar_col)
        # Preprocessing
        print('Preprocessing')
        df = preprocessing(df, cfg)

        # pickle形式で保存
        to_pickle('./input/data.pkl', df)
        try:
            experiment.log_asset(file_data='./input/data.pkl', file_name='data.pkl')
        except:
            pass

    features = [c for c in df.columns if c not in del_tar_col]

    # Model  ####################################################################################
    model = None
    if cfg.exp.model == 'lgb':
        model = LGBMModel(dict(cfg.lgb))
    elif cfg.exp.model == 'cat':
        model = CatBoostModel(dict(cfg.cat))

    # Train & Predict  ##############################################################################
    trainer = Trainer(model, id_col, tar_col, features, cv, criterion, experiment)
    trainer.fit(df)
    trainer.predict(df)
    trainer.get_feature_importance()


if __name__ == '__main__':
    main()
