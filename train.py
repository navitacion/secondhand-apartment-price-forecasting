import os
import numpy as np
from comet_ml import Experiment
import hydra
from omegaconf import DictConfig

import pandas as pd

from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_log_error, mean_absolute_error

from utils.utils import unpickle, to_pickle, seed_everything
from utils.preprocess import convert_wareki_to_seireki, normalize_area, normalize_moyori
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

    # TODO 用途, 建物の構造をOnehot
    for col in ['用途', '建物の構造']:
        tmp = df[col].str.get_dummies(sep='、')
        tmp.columns = [f'{col}_{c}' for c in tmp.columns]
        df = pd.concat([df, tmp], axis=1)
        del tmp, df[col]


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


    df.head(200).to_csv('ttt.csv', index=False)

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
