from pathlib import Path
import pandas as pd
from model import DelayModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

##
file_dir = Path('G:\My Drive\GitHub\proj-challenge-LATAM\data\data.csv')
data = pd.read_csv(file_dir)

## 
top_feat = [
        "OPERA_Latin American Wings", 
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air"
    ]

target_col = ['delay']

# try_data = {'OPERA': 'Aerolineas Argentinas',
#             'TIPOVUELO': 'N',
#             'MES': 3}
# try_data = pd.DataFrame([try_data])

##
model = DelayModel()

## test: prepr for training
feat, target = model.preprocess(data=data, target_column='delay')

_, feat_val, _, targ_val = train_test_split(feat, target, test_size = 0.33, random_state = 42)

model.fit(features=feat, target=target)

pred_target = model.predict(features=feat_val)

print(classification_report(targ_val, pred_target, output_dict=True))

model.save_model(version='0.1.0')

print('OK')