import pandas as pd
import sweetviz as sv

data_dir = './input'
train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')

col_name = ['ID', 'Type', 'Region', 'City Code', 'Prefecture Name', 'City Name', 'District Name', 'Nearest Station: Name',
            'Nearest station: distance', 'Floor plan', 'Area', 'Land shape', 'Frontage', 'Total floor area', 'Year built', 'Building structure',
            'Use', 'Future use', 'Frontal road: Direction', 'Frontal road: Type', 'Frontal road: Width', 'City planning', 'Building coverage',
            'Floor-area ratio', 'Date of transaction', 'Renovation', 'Circumstances of the transaction', 'Transaction price_log']

train.columns = col_name
test.columns = col_name[:-1]

train['Area'] = train['Area'].replace('2000㎡以上', 2000)
train['Area'] = train['Area'].astype(int)



skip_cols = ["ID"]
target_col = 'Transaction price_log'

feature_config = sv.FeatureConfig(skip=skip_cols)
my_report = sv.compare([train, "Training Data"], [test, "Test Data"], target_col, feature_config)

my_report.show_html()