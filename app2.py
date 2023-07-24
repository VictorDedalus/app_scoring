import numpy as np
import pandas as pd
import pickle
from flask import Flask, jsonify, request
from sklearn.ensemble import RandomForestClassifier
app = Flask(__name__)

@app.route('/api/users')
def get_list_of_users():
    app_test = pd.read_csv('data/application_test_cleaned.csv').drop(columns='Unnamed: 0').iloc[0:100]
    list_users = list(app_test['SK_ID_CURR'])

    return jsonify({'users': list_users})

@app.route('/api/user/<user_id>')
def get_specific_data(user_id):
    app_test = pd.read_csv('data/application_test_cleaned.csv').drop(columns='Unnamed: 0').iloc[0:100]
    app_test_scaled = pd.read_csv('data/application_test_scaled.csv').drop(columns='Unnamed: 0').iloc[0:100]
    filename = "scoring_model.pkl"
    model = pickle.load(open("modele/" + filename, 'rb'))
    result = model.predict_proba(app_test_scaled)[:,0]
    app_test_scaled['proba'] = result
    app_test_final = pd.merge(app_test, app_test_scaled[['proba']], left_index=True, right_index=True)
    app_test_specific = app_test_final.loc[app_test_final['SK_ID_CURR'] == int(user_id)].drop(columns='SK_ID_CURR').to_json(orient="columns")

    return app_test_specific

@app.route('/api/shap_features')
def get_shap_features():
    shap_features = pd.read_csv('data/shap_features.csv').drop(columns='Unnamed: 0').to_json(orient="columns")
    return shap_features


if __name__ == '__main__':
    app.run(debug=True)