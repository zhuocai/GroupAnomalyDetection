import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from scipy import interpolate
import json
import os

def eval_measure(test, pred, test_th = 0.02, pred_th = 0.24):
    TP, FP, TN, FN = 0,0,0,0
    for i in range(len(test)):
        if(test[i] >test_th):
            if(pred[i]>pred_th):
                TP+=1
            elif(pred[i]<=pred_th):
                FN +=1
        elif(test[i]<=test_th):
            if(pred[i]>pred_th):
                FP +=1
            elif(pred[i]<=pred_th):
                TN+=1
    if(TP+FP==0):
        print("TP+FP==0")
        return (0,0,0)
    if(TP+FN==0):
        print("TP+FN==0")
        return (0,0,0)

    pre = TP/(TP+FP)   
    rec = TP/(TP+FN)
    F1 = 2*pre*rec/(pre+rec)
    #print("pre: ", pre, ";  rec: ", rec, "; F1: ", F1)
    return (pre, rec, F1)    

normal_pc = np.load("../../data/SWAT/normal_pc.npy")
anomaly_pc = np.load("../../data/SWAT/anomaly_pc.npy")
if("cai_checkpoints" not in os.listdir()):
    os.mkdir("cai_checkpoints")


normal_len = len(normal_pc)
anomaly_len = len(anomaly_pc)
dimension = normal_pc.shape[1]
sample_size = 50

# Train
train_sample_step = 2
train_size = (normal_len-sample_size)//train_sample_step
train_index = np.arange(normal_len, train_sample_step)[:train_size]

print("train_sample_step: ", train_sample_step, ", train_size: ", train_size)

train_x = np.zeros((train_size, sample_size, dimension), dtype="double")
for i in range(train_size):
    train_x[i, :, :] = normal_pc[i*train_sample_step: (i*train_sample_step+sample_size), :]

train_y = np.zeros(train_size)

test_sample_step = 1
test_size = (anomaly_len-sample_size)//test_sample_step
print("test_sample_step: ", test_sample_step, "test size:", test_size)
test_index = np.array([i*test_sample_step for i in range(test_size)])
test_x = np.zeros((test_size, sample_size, dimension), dtype = "double")
for i in range(test_size):
    test_x[i,:,:] = anomaly_pc[test_index[i]:(test_index[i]+sample_size), :-1]
test_attack_level = np.array([np.mean(anomaly_pc[i:(i+sample_size), -1]) for i in test_index])
test_y = test_attack_level

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
prev_epoch=2
model_deep_dropout_sigmoid.load_weights("cai_checkpoints/model_deep_dropout_sigmoid_weights_epoch"+str(prev_epoch)+".h5")

print("previous:")
test_pred = model_deep_dropout_sigmoid.predict(test_x).ravel()
test_pred = -np.log(1+1e-10-test_pred)
print("After log:", pd.Series(test_pred).describe())

for epoch in range(prev_epoch+1,prev_epoch+6):
    print("***********epoch: ", epoch)
    model_deep_dropout_sigmoid.fit(X_train, y_train, epochs=3)
    # eval model
    train_pred = model_deep_dropout_sigmoid.predict(train_x).ravel()
    #train_pred_mse = np.array([mean_squared_error(train_y[i], train_pred[i]) for i in range(len(train_y))])
    #pd.Series(train_pred_mse).describe()
    test_pred = model_deep_dropout_sigmoid.predict(test_x).ravel()
    #test_pred_mse = np.array([mean_squared_error(test_y[i], test_pred[i])for i in range(len(test_y))])
    #pd.Series(test_pred_mse).describe()
    print("len test_pred: ", test_pred.shape, "len test_attack_level:", test_attack_level.shape)
    print("corr in test: ", np.corrcoef(test_pred, test_attack_level))
    res = {}
    check_num = 100
    if(epoch == 0):
        check_num = 2
    test_pred = -np.log(1+1e-10-test_pred)
    print("after log:", pd.Series(test_pred).describe())
    pred_mean = np.mean(test_pred)
    pred_std = np.std(test_pred)
    lower_bound = np.quantile(test_pred, 0.5)
    upper_bound = np.quantile(test_pred, 0.99)
    
    for i in np.linspace(lower_bound,upper_bound, check_num):
        res[str(i)] = eval_measure(test_attack_level, test_pred, test_th=1/(sample_size+1.0), pred_th=i)
        print("pred_th = ",i, ", pre, rec, f1 =",res[str(i)])
    np.save("cai_checkpoints/pred_epoch"+str(epoch)+".npy", test_pred)
    with open("cai_checkpoints/res_epoch"+str(epoch)+".txt", "w") as f:
        f.write(str(res))
    #model_deep_dropout_sigmoid.fit(X_train, y_train, epochs=2)
    model_deep_dropout_sigmoid.save_weights("cai_checkpoints/model_deep_dropout_sigmoid_weights_epoch"+str(epoch)+".h5")
    #model_deep_dropout_sigmoid.fit(X_train, y_train,epochs=2)


