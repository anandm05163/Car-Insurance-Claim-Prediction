from flask import Flask, jsonify, render_template_string
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)

# Load the saved models
rf_model = joblib.load('best_rf_model.pkl')
xgb_model = joblib.load('best_xgb_model.pkl')
lgb_model = joblib.load('best_lgbm_model.pkl')

# Load the saved X_test and y_test data
X_test = joblib.load('X_test.pkl')
y_test = joblib.load('y_test.pkl')

# HTML template for the front-end UI
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Prediction API</title>
</head>
<body>
    <h1>Welcome to the Model Prediction API</h1>
    <p>Click on a model to see predictions and metrics:</p>
    <ul>
        <li><a href="/predict/random_forest">Random Forest</a></li>
        <li><a href="/predict/xgboost">XGBoost</a></li>
        <li><a href="/predict/lightgbm">LightGBM</a></li>
    </ul>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(html_template)

@app.route('/predict/random_forest', methods=['GET'])
def predict_random_forest():
    y_pred_rf = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    rf_precision = precision_score(y_test, y_pred_rf)
    rf_recall = recall_score(y_test, y_pred_rf)
    rf_f1 = f1_score(y_test, y_pred_rf)
    
    return jsonify({
        'model': 'Random Forest',
        'metrics': {
            'accuracy': rf_accuracy,
            'precision': rf_precision,
            'recall': rf_recall,
            'f1_score': rf_f1
        }
    })

@app.route('/predict/xgboost', methods=['GET'])
def predict_xgboost():
    y_pred_xgb = xgb_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
    xgb_precision = precision_score(y_test, y_pred_xgb)
    xgb_recall = recall_score(y_test, y_pred_xgb)
    xgb_f1 = f1_score(y_test, y_pred_xgb)
    
    return jsonify({
        'model': 'XGBoost',
        'metrics': {
            'accuracy': xgb_accuracy,
            'precision': xgb_precision,
            'recall': xgb_recall,
            'f1_score': xgb_f1
        }
    })

@app.route('/predict/lightgbm', methods=['GET'])
def predict_lightgbm():
    y_pred_lgb = lgb_model.predict(X_test)
    lgb_accuracy = accuracy_score(y_test, y_pred_lgb)
    lgb_precision = precision_score(y_test, y_pred_lgb)
    lgb_recall = recall_score(y_test, y_pred_lgb)
    lgb_f1 = f1_score(y_test, y_pred_lgb)
    
    return jsonify({
        'model': 'LightGBM',
        'metrics': {
            'accuracy': lgb_accuracy,
            'precision': lgb_precision,
            'recall': lgb_recall,
            'f1_score': lgb_f1
        }
    })

if __name__ == '__main__':
    app.run(debug=True)
