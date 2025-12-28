from flask import Flask, jsonify
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

@app.route('/predict/<model_name>', methods=['GET'])
def predict(model_name):
    if model_name == 'random_forest':
        model = rf_model
    elif model_name == 'xgboost':
        model = xgb_model
    elif model_name == 'lightgbm':
        model = lgb_model
    else:
        return jsonify({'error': 'Model not found'}), 404

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return jsonify({
        'model': model_name,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
