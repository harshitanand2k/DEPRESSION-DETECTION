# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 14:59:40 2020

@author: HP
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__,template_folder='template')
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = (prediction[0])
    if output=='1':
        output='The results are positive'
    else:
        output='The results are negative'
        
    return render_template('index.html', prediction_text=output)


if __name__ == "__main__":
    app.run(debug=True)
