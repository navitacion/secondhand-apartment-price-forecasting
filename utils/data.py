import os
import glob
import numpy as np
import pandas as pd


def concat_data(data_dir):
    """
    Concat Multi csv files
    ---------------------------------------------
    Parameter
    data_dir: str
        directory of dataset

    ---------------------------------------------
    Returns
    train: dataframe
        Concatenated train dataset (concatenated train, test)
    """
    csv_path_list = glob.glob(os.path.join(data_dir, '*.csv'))
    train = pd.DataFrame()
    for path in csv_path_list:
        d = pd.read_csv(path)
        train = pd.concat([train, d], axis=0)

    return train



def load_data(data_dir, down_sample=1.0, seed=0, id_col=None, target_col='target'):
    """
    Load Raw Data
    Assumpt 2 Dataset 'train.csv' and 'test.csv'
    ---------------------------------------------
    Parameter
    data_dir: str
        directory of dataset
    down_sample: float
        sampling train data
        for Debug
    seed: int
        Random Seed
    id_col: str
        column of Unique Id
        if id_col is None, 'id' columns add original dataset
    target_col: str
        column of target

    ---------------------------------------------
    Returns
    df: dataframe
        dataset (concatenated train, test)
    """
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    if id_col is None:
        train['id'] = np.arange(len(train))
        test['id'] = np.arange(len(test))
    test[target_col] = 0

    train['is_train'] = 1
    test['is_train'] = 0

    if down_sample < 1:
        train = train.sample(frac=down_sample, random_state=seed).reset_index(drop=True)

    df = pd.concat([train, test], axis=0, ignore_index=True)

    return df