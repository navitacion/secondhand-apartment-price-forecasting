import os
import numpy as np
from comet_ml import Experiment
import hydra
from omegaconf import DictConfig
import mojimoji
import itertools

import pandas as pd

from sklearn.model_selection import KFold, GroupKFold
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_log_error, mean_absolute_error

from utils.utils import unpickle, to_pickle, seed_everything
from utils.preprocess import convert_wareki_to_seireki, normalize_area, normalize_moyori, preprocess_madori
from utils.data import load_data
from features import CategoryEncoder, FrequencyEncoder, GroupbyTransformer, \
    TextVectorizer, PivotCountEncoder, LagGroupbyTransformer, LagRollingGroupbyTransformer

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
    # ()を削除
    df['最寄駅：名称'] = df['最寄駅：名称'].apply(lambda x: x.split('(')[0])

    # TODO 面積
    df = normalize_area(df)

    # TODO 最寄駅：距離（分）
    df = normalize_moyori(df)

    # TODO 和暦→西暦
    df['建築年'] = df['建築年'].apply(convert_wareki_to_seireki)

    # 建築年をビニング
    df['df_cat_建築年'] = pd.cut(df['建築年'], 10, labels=False)

    # TODO 2021年時点からの経過時間
    for c in ['建築年']:
        df[f'fe_diff_{c}_from_2021'] = 2021 - df[c]

    # TODO 建ぺい率と容積率と面積
    cols = ['建ぺい率（％）', '容積率（％）', '面積（㎡）']
    for c1, c2 in itertools.combinations(cols, 2):
        df[f'fe_mul_{c1}_{c2}'] = df[c1] * df[c2]
        df[f'fe_div_{c1}_{c2}'] = df[c1] / df[c2]
        df[f'fe_div_{c2}_{c1}'] = df[c2] / df[c1]

    # TODO 取引時点→`year`, `quarter`
    df['year'] = df['取引時点'].apply(lambda x: x[:4]).astype(int)
    df['quarter'] = df['取引時点'].apply(lambda x: x[6:7]).astype(int)


    # TODO 取引時と建築年の差分
    # quarterの情報も付ける
    rep = {1: 0, 2: 0.25, 3: 0.5, 4: 0.75}
    df['quarter_rep'] = df['quarter'].map(rep)
    df['year_rep'] = df['year'] + df['quarter_rep']
    df['fe_diff_取引時点_建築年'] = df['year_rep'] - df['建築年']

    df.drop(['quarter_rep', 'year_rep'], axis=1, inplace=True)


    # TODO 用途, 建物の構造をOnehot
    for col in ['用途', '建物の構造']:
        tmp = df[col].str.get_dummies(sep='、')
        tmp.columns = [f'fe_is_{col}_{c}' for c in tmp.columns]
        df = pd.concat([df, tmp], axis=1)
        del tmp

    # TODO 間取り
    df = preprocess_madori(df)

    # TODO 1部屋あたりの面積
    df['fe_area_per_room'] = df['面積（㎡）'] / df['fe_count_room']

    # 地域
    df['fe_concat_都道府県_市区町村'] = df['都道府県名'] + df['市区町村名']
    df['fe_concat_都道府県_市区町村_地区'] = df['都道府県名'] + df['市区町村名'] + df['地区名']
    df['fe_concat_都道府県_市区町村_地区_最寄'] = df['都道府県名'] + df['市区町村名'] + df['地区名'] + df['最寄駅：名称'].apply(lambda x: x.split('(')[0])


    # TODO Groupbyでいろいろ集約
    print('groupby')
    group_cols = ['都道府県名', '市区町村名', '地区名', '最寄駅：名称',
                  'fe_concat_都道府県_市区町村_地区', 'fe_concat_都道府県_市区町村_地区_最寄']
    value_cols = ['建ぺい率（％）', '容積率（％）', '面積（㎡）', '最寄駅：距離（分）', 'fe_count_room', 'fe_area_per_room']
    value_cols += [c for c in df.columns if c.startswith('fe_mul_')]
    value_cols += [c for c in df.columns if c.startswith('fe_div_')]
    value_cols += [c for c in df.columns if c.startswith('fe_diff_')]
    aggs = ['mean', 'sum', 'std', 'max', 'min']

    for conbi in [1]:
        transformer = GroupbyTransformer(group_cols, value_cols, aggs, conbination=conbi, use_cudf=True)
        df = transformer.transform(df)

    # TODO PivotEmbedding
    index_col = '市区町村名'
    count_cols = ['建築年', '用途', '建物の構造']
    transformer = PivotCountEncoder(index_col=index_col, count_cols=count_cols, value_col='ID')
    df = transformer.transform(df)


    # TODO
    aggs = ['mean', 'std']
    group_cols = ['市区町村名', '地区名', '最寄駅：名称']
    for g in group_cols:
        transformer = LagGroupbyTransformer(value_cols=['取引価格（総額）_log'], time_cols=['year'], aggs=aggs, lags=[1, 2], group_col=g)
        df = transformer.transform(df)

        transformer = LagRollingGroupbyTransformer(value_cols=['取引価格（総額）_log'], time_cols=['year'], aggs=aggs, lags=[1], group_col=g, windows=[2, 3])
        df = transformer.transform(df)


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

    print(df.isnull().sum())

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
                            auto_output_logging='simple',
                            auto_metric_logging=False)

    experiment.log_parameters(dict(cfg.data))

    # Config  ####################################################################################
    del_tar_col = ['取引時点']
    id_col = 'ID'
    tar_col = '取引価格（総額）_log'
    g_col = 'year'
    criterion = MAE
    cv = KFold(n_splits=cfg.data.n_splits, shuffle=True, random_state=cfg.data.seed)
    # cv = GroupKFold(n_splits=5)


    # Load Data  ####################################################################################
    if cfg.exp.use_pickle:
        # pickleから読み込み
        df = unpickle('./input/data.pkl')

    else:
        df = load_data(data_dir, sampling=cfg.data.sampling, seed=cfg.data.seed, id_col=id_col, target_col=tar_col)
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
    trainer = Trainer(model, id_col, tar_col, g_col, features, cv, criterion, experiment)
    trainer.fit(df)
    trainer.predict(df)
    trainer.get_feature_importance()


if __name__ == '__main__':
    main()
