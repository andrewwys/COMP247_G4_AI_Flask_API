# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 16:59:37 2022

@author: Group 4 - COMP247
"""

from flask import Flask, request, jsonify
import traceback
import pandas as pd
#from sklearn import preprocessing
# import pickle
import joblib
import sys
from os import path
from sklearn import metrics
from flask_cors import CORS

project_folder = r'C:\Projects\COMP247\Final_Project\_deploy'
models = {
          "Random_Forest": "group4_rf_fullpipe_rajiv.pkl"
         ,"Neuro_Network": "group4_nn_fullpipe_v7_andrew.pkl"
         ,"Decision_Tree": "group4_dt_fullpipeline_manvir.pkl"
         ,"Logistic_Regression": "LR_Model_Chung.pkl"
         ,"SVM": "SVM_model_parth.pkl"
         }

cols_pkl = 'group4_model_columns.pkl'

X_train_df = pd.read_csv(path.join(project_folder,"x_train_data.csv"))
y_train_df = pd.read_csv(path.join(project_folder,"y_train_data.csv"))
X_test_df = pd.read_csv(path.join(project_folder,"x_test_data.csv"))
y_test_df = pd.read_csv(path.join(project_folder,"y_test_data.csv"))

# Your API definition
app = Flask(__name__)
CORS(app)

@app.route("/predict/<model_name>", methods=['GET','POST']) #use decorator pattern for the route
def predict(model_name):
    if loaded_model:
        try:
            json_ = request.json
            print('JSON: \n', json_)
            query = pd.DataFrame(json_, columns=model_columns)
            prediction = list(loaded_model[model_name].predict(query))
            print(f'Returning prediction with {model_name} model:')
            print('prediction=', prediction)
            res = jsonify({"prediction": str(prediction)})
            res.headers.add('Access-Control-Allow-Origin', '*')
            return res
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        return ('No model available.')
    
@app.route("/scores/<model_name>", methods=['GET','POST']) #use decorator pattern for the route
def scores(model_name):
    if loaded_model:
        try:
            y_pred = loaded_model[model_name].predict(X_test_df)
            print(f'Returning scores for {model_name}:')
            accuracy = metrics.accuracy_score(y_test_df, y_pred)
            precision = metrics.precision_score(y_test_df, y_pred)
            recall = metrics.recall_score(y_test_df, y_pred)
            f1 = metrics.f1_score(y_test_df, y_pred)
            print(f'accuracy={accuracy}  precision={precision}  recall={recall}  f1={f1}')
            res = jsonify({"accuracy": accuracy,
                            "precision": precision,
                            "recall":recall,
                            "f1": f1
                           })
            res.headers.add('Access-Control-Allow-Origin', '*')
            return res
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        return ('No model available.')
        

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345
        
    # load all models:
    loaded_model = {}
    for model_name in (models):
        loaded_model[model_name] = joblib.load(path.join(project_folder, models[model_name]))
        print(f'Model {model_name} loaded')
        
    model_columns = ['Elapsed_Days_Before_Reported', 'Primary_Offence', 'Occurrence_Year',
           'Occurrence_DayOfWeek', 'Occurrence_DayOfYear', 'Occurrence_Hour',
           'Division', 'City', 'Hood_ID', 'Premises_Type', 'Bike_Make',
           'Bike_Model', 'Bike_Type']
    # model_columns = joblib.load(path.join(project_folder, cols_pkl))
    # print ('Model columns loaded')
    
    app.run(port=port, debug=True)
    
