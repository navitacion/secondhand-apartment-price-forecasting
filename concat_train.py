from utils.data import concat_data

data_dir = './input/train'
train = concat_data(data_dir)

train.to_csv('./input/train.csv', index=False)