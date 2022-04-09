# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 16:59:37 2022

@author: Group 4 - COMP247
"""

from flask import Flask, request, jsonify
import traceback
import pandas as pd
#from sklearn import preprocessing
import pickle
import sys
from os import path

project_folder = r'C:\Projects\COMP247\Final_Project\_deploy'
models = {
        "Random Forest": "group4_nn_fullpipe_v7_andrew.pkl",
        "Neuro Network": "group4_nn_fullpipe_v7_andrew.pkl",
        "Decision Tree": "group4_nn_fullpipe_v7_andrew.pkl",
        "Logistic Regression": "group4_nn_fullpipe_v7_andrew.pkl",
        "SVM": "group4_nn_fullpipe_v7_andrew.pkl"
    }

cols_pkl = 'group4_model_columns.pkl'

# Your API definition
app = Flask(__name__)

@app.route("/predict/<model_name>", methods=['GET','POST']) #use decorator pattern for the route
def predict(model_name):
    if loaded_model:
        try:
            json_ = request.json
            print('JSON: \n', json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)
            print('Query: \n', query)

            # query = pd.DataFrame(query, columns=model_columns)
            # print(query)
            prediction = list(loaded_model[model_name].predict(query))
            print({'prediction': str(prediction)})
            return jsonify({'prediction': str(prediction)})
            return "Welcome to COMP247 - Group 4 APIs!"

        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345
        
    # load all models:
    loaded_model = {}
    for model_name in (models):
        loaded_model[model_name] = pickle.load(open(path.join(project_folder, models[model_name]),"rb"))
        print(f'Model {model_name} loaded')
        
    model_columns = pickle.load(open(path.join(project_folder, cols_pkl),"rb"))
    print ('Model columns loaded')
    
    app.run(port=port, debug=True)
