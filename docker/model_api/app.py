from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import joblib
import json
app = Flask(__name__)

with open('config.json') as json_file:
    config = json.load(json_file)

@app.route('/', methods=['POST'])
def model_predict():
    content = request.json

    if not all(key in content for key in config['predict_features']):
        return 'The request could not be completed due to missing parameters'
    
    try:
        scaler = joblib.load('./models/scaler.joblib')
        model = joblib.load('models/model.joblib')

        df = pd.DataFrame(content, index=[0])
        df = df[config['predict_features']]
        df['Amount'] = scaler.fit_transform(df[['Amount']].values)

        pred = model.predict_proba(df)
        pred = float(f'{pred[:,1][0]:.2f}')

        preddicted_class = 1 if pred >= config['predict_thereshold'] else 0

        return jsonify({
            "probability": pred,
            "class": preddicted_class
        })

    except Exception as e:
        return {"error": str(e)}


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')