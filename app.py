# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 11:45:56 2020

@author: Niteesh
"""

from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

pickle_in = open('cvd-prediction-svc-model.pkl', 'rb')
classifier = pickle.load(pickle_in)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    for rendering result on HTML GUI
    """

    age = int(request.form['age'])
    height = int(request.form['height'])
    weight = int(request.form['weight'])
    systolic_bp = int(request.form['systolic_bp'])
    diastolic_bp = int(request.form['diastolic_bp'])
    cholesterol = int(request.form['cholesterol'])
    glucose = int(request.form['glucose'])
    smoke = int(request.form['smoke'])
    alcoholic = int(request.form['alcoholic'])
    active = int(request.form['active'])
    gender = int(request.form['gender'])

    final_features = [[age,gender,height,weight,systolic_bp,diastolic_bp,cholesterol,glucose,smoke,alcoholic,active]]
    my_prediction = classifier.predict(final_features)

    if my_prediction:
        return render_template('index.html', prediction_text="Oops! You must have CVD, consult a doctor.")
    else:
        return render_template('index.html', prediction_text="Yay! negative, no CVD.")


if __name__ == "__main__":
    app.run(debug=True)
