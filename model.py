import os
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import regularizers, callbacks

def getData(username):
    data_dir = username + "/data"
    combined_data = pd.DataFrame()

    for filename in os.listdir(data_dir):
        dataset = pd.read_json(os.path.join(data_dir, filename), orient='split', convert_dates=True)
        dataset = dataset.set_index('time')
        combined_data = combined_data.append(dataset)

    return combined_data

def datasplit(data):
    train_size = int(len(data) * 0.90)
    train, test = data.iloc[0:train_size], data.iloc[train_size:len(data)]
    return train, test

def preprocess(train, test, username):
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(train)
    x_test = scaler.transform(test)
    scaler_filename = username + "/scaler_data"
    joblib.dump(scaler, scaler_filename)
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
    num_features = x_train.shape[2]
    return x_train, x_test, num_features

def network_model(x, num):
    inputs = Input(shape=(x.shape[1], x.shape[2]))
    l1 = LSTM(num*num, activation='relu', return_sequences=True,
             kernel_regularizer=regularizers.l2(0.00))(inputs)
    l2 = LSTM(num, activation='relu', return_sequences=False)(l1)
    l3 = RepeatVector(x.shape[1])(l2)
    l4 = LSTM(num, activation='relu', return_sequences=True)(l3)
    l5 = LSTM(num*num, activation='relu', return_sequences=True)(l4)
    output = TimeDistributed(Dense(x.shape[2]))(l5)
    model = Model(inputs=inputs, outputs=output)
    return model

def train_net(x_train, train, num_features, username):
    model = network_model(x_train, num_features)
    model.compile(optimizer='adam', loss='mae')
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', restore_best_weights=True, min_delta=0.0005, patience=5)
    nb_epochs = 100
    batch_size = 10
    history = model.fit(x_train, x_train, epochs=nb_epochs, batch_size=batch_size, validation_split=0.1, callbacks=[callback]).history

    x_pred = model.predict(x_train)
    x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[2])
    x_pred = pd.DataFrame(x_pred, columns=train.columns)
    x_pred.index = train.index

    scored = pd.DataFrame(index=train.index)
    xtrain = x_train.reshape(x_train.shape[0], x_train.shape[2])
    scored['Loss_mae'] = np.mean(np.abs(x_pred - xtrain), axis=1)
    scoreSorted = np.sort(scored['Loss_mae'])
    threshold = scoreSorted[int(0.97 * len(scoreSorted))]

    threshold_name = username + "/threshold"
    model_loc = username + "/model.h5"
    model.save(model_loc)
    joblib.dump(threshold, threshold_name)

def predict_anom(data, username):
    data_out = {}
    model_name = username + "/model.h5"
    threshold_name = username + "/threshold"
    threshold = joblib.load(threshold_name)
    scaler_name = username + "/scaler_data"
    scaler = joblib.load(scaler_name)
    x = scaler.transform(data)
    x = x.reshape(x.shape[0], 1, x.shape[1])

    with tf.Graph().as_default():
        model = load_model(model_name)
        data_out["Analysis"] = []

        preds = model.predict(x)
        preds = preds.reshape(preds.shape[0], preds.shape[2])
        preds = pd.DataFrame(preds, columns = data.columns)
        preds.index = data.index

        scored = pd.DataFrame(index = data.index)
        yhat = x.reshape(x.shape[0], x.shape[2])
        scored['Loss_mae'] = np.mean(np.abs(yhat - preds), axis =1)
        scored['Threshold'] = threshold
        scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']

        triggered = []
        for i in range(len(scored)):
            temp = scored.iloc[i]
            if temp.iloc[2]:
                triggered.append(temp)
        print(len(triggered))
        if len(triggered) > 0:
            for j in range(len(triggered)):
                out = triggered[j]
                result = {"Anomaly": True, "date": f"{out.name.year:02d}" + "-" + f"{out.name.month:02d}" + "-" + f"{out.name.day:02d}", "time": f"{out.name.hour:02d}" + ":" + f"{out.name.minute:02d}"}
                data_out["Analysis"].append(result)

    return data_out