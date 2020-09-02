import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import tensorflow as tf
import json
import model
import os

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/learn", methods=['POST'])
def learn():
    if request.method == "POST":
        username = str(request.form['username'])
        data = pd.read_json(request.json, orient='split', convert_dates=['time'])
        print(data.head)
        if not os.path.isdir(username):
            os.mkdir(username)

        # data.set_index('time', inplace=True)
        train, test = model.datasplit(data)

        x_train, x_test, num_features = model.preprocess(train, test, username)
        model.train_net(x_train, train, num_features, username)
        return "Model successfully trained!"

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == "POST":
        username = str(request.form['username'])
        if not os.path.isdir(username):
            os.mkdir(username)

        data = pd.read_json(request.json, orient='split', convert_dates=['time'])
        # data.set_index('time', inplace=True)
        data_out = model.predict_anom(data, username)
        return jsonify(data_out)

@app.route("/test", methods=['POST'])
def test():
    if request.method == "POST":
        username = request.form['username']
        stuff = request.json
        print(stuff)
        return username

if __name__ == '__main__':
    app.run(host= '0.0.0.0', debug=True)