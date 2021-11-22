from os import sep
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, classification_report, roc_curve, auc
import joblib
import wget
import json
import zipfile
from os.path import exists
from random import seed
seed(42)

def get_data():
    file_name = 'data.zip'
    file_exists = exists(file_name)

    if not file_exists:

        url = 'https://www.dropbox.com/s/zio9jdlfczenkps/data.zip?dl=1'

        wget.download(url)

        with zipfile.ZipFile(f'./{file_name}', 'r') as zip_ref:
            zip_ref.extractall('./temp/')

def pre_processing(df, train=False, scaler_file='./temp/scaler.joblib'):
    df = df.dropna(how='any',axis=0) 
    df = df.drop_duplicates(keep='last')

    if train:
        scaler = MinMaxScaler()
        df['Amount'] = scaler.fit_transform(df[['Amount']].values)

        joblib.dump(scaler, scaler_file)
    else:
        scaler = joblib.load(scaler_file)
        df['Amount'] = scaler.transform(df[['Amount']].values)

    X = df
    y = X.pop('Class')

    return (X, y)

def get_class_weight(y):
    class_counts = y.value_counts()

    w_array = np.array([1]*y.shape[0])
    w_array[y==1] = class_counts[0]/class_counts[1]
    w_array[y==0] = class_counts[0]/class_counts[0]

def plot_xgb_learning_curve(model, metric, label, title, file_name):
  results = model.evals_result()
  epochs = len(results['validation_0'][metric])
  x_axis = range(0, epochs)

  fig, ax = plt.subplots()
  ax.plot(x_axis, results['validation_0'][metric], label='Train')
  ax.plot(x_axis, results['validation_1'][metric], label='Val')
  ax.legend()
  plt.ylabel(label)
  plt.title(title)
  sns.despine(left=True, right=True)
  plt.savefig(f"{file_name}.png",dpi=80)

def save_model_metrics(model, X, y, threshold=0.5):
    y_proba = model.predict_proba(X)
    y_proba = y_proba[:,1]

    y_pred = list(map(lambda x: int(x > threshold), y_proba))

    target_names = np.unique(y)
    target_names = list(map(str, target_names))
    
    metrics_dict = {}
    
    acc_bal = balanced_accuracy_score(y, y_pred)
    metrics_dict['balanced_accuracy'] = acc_bal

    cm = confusion_matrix(y, y_pred, labels=np.unique(y))
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    acc_classes = cmn.diagonal()

    for i, acc in enumerate(acc_classes):
        dic_key = f'accuracy_class_{target_names[i]}'
        metrics_dict[dic_key] = float(f'{acc:.2f}')

    fpr, tpr, _ = roc_curve(y, y_pred)
    roc_auc = auc(fpr, tpr)
    metrics_dict['auc'] = float('%0.2f' % roc_auc)

    with open("metrics.json", 'w') as outfile:
        json.dump(metrics_dict, outfile)




get_data()

with open('config.json') as json_file:
    config = json.load(json_file)

df_train = pd.read_csv('./temp/creditcard_train.csv', sep=';', usecols=config['features'])
df_val = pd.read_csv('./temp/creditcard_val.csv', sep=';', usecols=config['features'])

X_train, y_train = pre_processing(df_train, True)
X_val, y_val = pre_processing(df_val)


w_array = get_class_weight(y_train)

xgb_model_best = xgb.XGBClassifier(
    colsample_bylevel = 0.8999999999999999,
    colsample_bytree = 0.7,
    learning_rate = 0.1,
    max_depth = 6,
    n_estimators = 180,
    reg_alpha = 0.75,
    reg_lambda = 0.25
)

eval_set = [(X_train, y_train), (X_val, y_val)]
xgb_model_best.fit(X_train, y_train, eval_metric=["auc","logloss"], eval_set=eval_set, verbose=True, sample_weight=w_array)

joblib.dump(xgb_model_best, './temp/model.joblib')

plot_xgb_learning_curve(xgb_model_best, 'auc', 'AUC', 'XGBoost AUC', './temp/auc_learning_curve')
plot_xgb_learning_curve(xgb_model_best, 'logloss', 'Log Loss', 'XGBoost Log Loss', './temp/logloss_learning_curve')
save_model_metrics(xgb_model_best, X_val, y_val, threshold=0.5)