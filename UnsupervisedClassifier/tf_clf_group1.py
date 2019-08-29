#!/usr/bin/env python
# coding: utf-8


import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from scipy import interpolate
import json



def interp_1darray(data):
    assert len(data.shape)==1, "data is not 1d"
    num_index = [i for i in range(len(data)) if not np.isnan(data[i])]
    f= interpolate.interp1d(num_index, data[num_index])
    data_new = f(range(len(data)))
    return data_new
    
def interp_2darray(data):
    assert len(data.shape)==2," data is not 2d "
    new_data = data.copy().T
    for i in range(len(new_data)):
        if(np.sum(np.isnan(new_data[i]))>0):
            new_data[i] = interp_1darray(new_data[i])
    new_data = new_data.T
    return new_data
    
def normalize_1d(data):
    new_data = data.copy()
    min_ = min(new_data)
    max_ = max(new_data)
    if(max_-min_==0):
        new_data = np.ones(new_data.shape,dtype="double") 
    #print(data)
    else:
        new_data= (new_data-min_)/(max_-min_)
    #print(data)
    return new_data

def normalize_2d(data, axis = 1):
    new_data = data.copy()
    if(axis == 0):
        new_data = new_data.T
    #print(data.shape)
    for i in range(len(new_data)):
        new_data[i] = normalize_1d(new_data[i])
    if(axis == 0):
        new_data = new_data.T
        
    return new_data



def get_eval(ground_truth, pred_val):
    assert ground_truth.shape == pred_val.shape
    TP, FP, TN, FN = 0,0,0,0
    gth = ground_truth.ravel()
    pred = pred_val.ravel()
    for i in range(len(gth)):
        if(gth[i] >0.5):
            if(pred[i]>0.5):
                TP+=1
            elif(pred[i]<=0.5):
                FN +=1
        elif(gth[i]<=0.5):
            if(pred[i]>0.5):
                FP +=1
            elif(pred[i]<=0.5):
                TN+=1
    if(TP==0 or FP==0 or FN==0):
        #print("someone=0, TP=", TP, " FP=", FP, ", FN=",FN)
        return (0,0,0)
    pre = TP/(TP+FP)
    rec = TP/(TP+FN)
    F1 = 2*pre*rec/(pre+rec)
    #print("pre: ", pre, ";  rec: ", rec, "; F1: ", F1)
    return (pre, rec, F1)    
def get_pred_errors(model, test_x, test_y):
    test_pred = model.predict(test_x)
    return np.array([mean_squared_error(test_y[i], test_pred[i]) for i in range(len(test_y))])


if(__name__ == "__main__"):
    normal_pc = np.load("../../data/WADI/normal_pc4.npy")
    anomaly_pc = np.load("../../data/WADI/anomaly_pc4.npy")


    normal_len = len(normal_pc)
    anomaly_len = len(anomaly_pc)
    dimension = normal_pc.shape[1]
    sample_size = 30

    # Train
    train_sample_step = 10
    train_size = (normal_len-sample_size)//train_sample_step
    train_index = np.arange(normal_len, train_sample_step)[:train_size]

    print("train_sample_step: ", train_sample_step, ", train_size: ", train_size)

    train_x = np.zeros((train_size, sample_size, dimension), dtype="double")
    for i in range(train_size):
        train_x[i, :, :] = normal_pc[i*train_sample_step: (i*train_sample_step+sample_size), :]


    test_sample_step = 1
    test_size = (anomaly_len-sample_size)//test_sample_step
    print("test_sample_step: ", test_sample_step)
    test_index = np.array([i*test_sample_step for i in range(test_size)])
    test_x = np.zeros((test_size, sample_size, dimension), dtype = "double")
    for i in range(test_size):
        test_x[i,:,:] = anomaly_pc[test_index[i]:(test_index[i]+sample_size), :-1]

    test_attack_level = np.array([np.mean(anomaly_pc[i:(i+sample_size), -1]) for i in test_index])
    X_train = np.concatenate((train_x, test_x), axis=0)
    y_train = np.concatenate((np.zeros(train_size), np.ones(test_size)))


    model_deep_dropout_sigmoid = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(sample_size,dimension)),
        tf.keras.layers.Dense(1000, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1000, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(200, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(40, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    #adam = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
    model_deep_dropout_sigmoid.compile(optimizer="Adam",
                  loss=tf.keras.losses.BinaryCrossentropy(),
                   metrics=["accuracy"])

    for epoch in range(50):
        print("***********epoch: ", epoch)
        model_deep_dropout_sigmoid.fit(X_train, y_train, epochs=1)
        # eval model
        y_pred = model_deep_dropout_sigmoid.predict(X_train).ravel()
        print("clf mean on 14days", 1 - np.mean(y_pred[:train_size]))
        print("corr on 2days", np.corrcoef(y_pred[train_size:], anomaly_pc[test_index,:][:,-1]))
        res = {}
        res["F1"] = []
        res["rec"] = []
        res["pre"] = []
        for th in np.arange(0.01, 1, 0.01):
            y_pred_th = np.array([1 if y_pred[i] > th else 0 for i in range(len(y_pred))])
            #print("On 14days:", 1 - np.mean(y_pred_th[:train_size]))
            pre, rec, f1 = get_eval(np.array(y_pred_th[train_size:]), anomaly_pc[test_index,:][:, -1])
            res["F1"].append(f1)
            #print("F1 score: ", f1, "rec:", rec, "pre:", pre)
            res["rec"].append(rec)
            res["pre"].append(pre)

        print(max(res["F1"]))
        pd.DataFrame(res).to_csv("res_epoch"+str(epoch)+".csv")
        model_deep_dropout_sigmoid.save("model_deep_dropout_sigmoid_epoch"+str(epoch)+".h5")

