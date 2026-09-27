from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the saved model and thresholds once, when the server starts
iso_forest = joblib.load('isolation_forest_model.pkl')
thresholds = joblib.load('thresholds.pkl')
lower_bound = thresholds['lower_bound']
upper_bound = thresholds['upper_bound']
feature_cols = thresholds['features']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    amount = data['amount']

    row = {
        'amount': amount,
        'hour': data.get('hour', 12),
        'day_of_week': data.get('day_of_week', 0),
        'is_night': 1 if data.get('hour', 12) < 5 or data.get('hour', 12) > 22 else 0,
        'sender_txn_count': data.get('sender_txn_count', 1),
        'amount_deviation': data.get('amount_deviation', 0),
        'time_since_last_txn': data.get('time_since_last_txn', 9999)
    }

    X = pd.DataFrame([row])[feature_cols]

    iqr_flag = int(amount < lower_bound or amount > upper_bound)
    iso_pred = iso_forest.predict(X)[0]
    iso_flag = int(iso_pred == -1)

    is_fraud = bool(iqr_flag or iso_flag)
    risk_score = iqr_flag + iso_flag

    return jsonify({
        'is_fraud': is_fraud,
        'risk_score': risk_score,
        'iqr_flag': iqr_flag,
        'iso_flag': iso_flag
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)