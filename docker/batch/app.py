import pandas as pd
import mysql.connector as sql
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import joblib
import json
import wget
import zipfile
from os.path import exists
from datetime import datetime

config = {
        'user': 'root',
        'password': 'root',
        'host': 'db',
        'port': '3306',
        'database': 'creditcardfraud'
    }
conn = sql.connect(**config)

cur = conn.cursor()

def get_data():
    file_name = 'data2.zip'
    file_exists = exists(file_name)

    if not file_exists:

        url = 'https://www.dropbox.com/s/r7ivk23ht56zmkv/data2.zip?dl=1'

        wget.download(url)

        with zipfile.ZipFile(f'./{file_name}', 'r') as zip_ref:
            zip_ref.extractall('./temp/')

def error_log(msg, error, file_dir='./logs.txt'):
    log_string = f'{str(datetime.now(tz=None))}: {str(error)}'
    print(msg)
    with open(file_dir, 'a') as file:
        file.write(log_string)

with open('config.json') as json_file:
    config = json.load(json_file)

try: 
    for df_chunk in pd.read_csv(
            './temp/data.csv',
            sep=';',
            usecols=config['predict_features'],
            dtype=config['dtype'],
            chunksize=config['chunksize']
            ):

        scaler = joblib.load('./models/scaler.joblib')
        model = joblib.load('models/model.joblib')
        df_chunk['Amount'] = scaler.fit_transform(df_chunk[['Amount']].values)

        pred= model.predict_proba(df_chunk)
        df_chunk['proba'] = float(f'{pred[:,1][0]:.2f}')

        df_chunk['preddicted_class'] = np.where(df_chunk['proba'] >= config['predict_thereshold'], 1, 0)
        df_chunk['predict_thereshold'] = config['predict_thereshold']

        data = (list(zip(df_chunk['preddicted_class'], df_chunk['proba'], df_chunk['Amount'], df_chunk['predict_thereshold'])))

        sql_insert_query = "INSERT INTO transaction_score (predicted_class, proba, amount, threshold) VALUES (%s, %s, %s, %s)"
        
        try:
            cur.executemany(sql_insert_query,data)
            conn.commit()
            print('success insert')
        
        except (sql.Error,sql.Warning) as e:
            conn.close()
            error_log('MySQL Error -> please check logs.txt file', error=e)
        

except Exception as e:
    error_log('Error -> please check logs.txt file', error=e)



    