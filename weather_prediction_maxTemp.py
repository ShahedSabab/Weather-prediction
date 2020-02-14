# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:49:28 2020

@author: sabab
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import os
from sklearn.model_selection import train_test_split
from pandas.plotting import register_matplotlib_converters
from sklearn import metrics
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

def create_dataset(X, y, time_steps=1):
  Xs, ys = [], []
  for i in range(len(X)-time_steps):
    v= X.iloc[i: (i+time_steps)].to_numpy()
    Xs.append(v)
    ys.append(y.iloc[i + time_steps])
  return np.array(Xs), np.array(ys)

def create_model():
    model = tf.keras.Sequential([
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, input_shape = (X_train.shape[1], X_train.shape[2]))),
            tf.keras.layers.Dropout(rate = dropout_prob),
            tf.keras.layers.Dense(1)
            ])
    return model

def plot_prediction(f, y_test, y_pred):
    plt.plot(y_test, marker='.', label='true')
    plt.plot(y_pred,'r',marker='.', label='predicted')
    plt.legend()
    plt.title('Max Temperature')


    
    
#register_matplotlib_converters()
#sns.set(style='whitegrid', palette='muted', font_scale=1.5)
#rcParams['figure.figsize']=22, 10

RANDOM_SEED = 42
dropout_prob = 0.2
train_size_frac = 0.8
TIME_STEPS = 7     #divide data into month basis 30 days
epochs = 190
batch_size = 16
lr = 0.0001
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

#read data
df = pd.read_csv(
    "toronto_weather.csv",
    parse_dates=['date'],
    index_col = "date"
)


train_size = int(len(df)*train_size_frac)
test_size = len(df) - train_size


#split train and test
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]


##scale and convert data only using taining 
base_columns = ['wind_speed', 'pressure', 'humidity']
label_column_max = ['max_temp']

base_transformer = RobustScaler()
base_transformer = base_transformer.fit(train[base_columns].to_numpy())
train.loc[:, base_columns] = base_transformer.transform(train[base_columns].to_numpy())
test.loc[:, base_columns] = base_transformer.transform(test[base_columns].to_numpy())


#for max temp
label_column_max_transformer = RobustScaler()
label_column_max_transformer = label_column_max_transformer.fit(train[label_column_max])
train[label_column_max] = label_column_max_transformer.transform(train[label_column_max])
test[label_column_max] = label_column_max_transformer.transform(test[label_column_max])


## Calculation for max_temp

#create data using time  steps 
X_train, y_train = create_dataset(train[train.columns[:-3]], train.max_temp, time_steps=TIME_STEPS)
X_test, y_test = create_dataset(test[test.columns[:-3]], test.max_temp, TIME_STEPS)

##[samples, time_steps, n_features]

#create model
model = create_model()

earlystop_callback = tf.keras.callbacks.EarlyStopping(
  monitor='mean_squared_error', patience=6)
opt = tf.optimizers.Adam(lr)
#compile model
model.compile(loss='mean_squared_error' ,optimizer=opt)

#fit model
history = model.fit(
    X_train, y_train,
    epochs = epochs,
    batch_size = batch_size,
    validation_split=0.1,
    shuffle= False,
    callbacks=[earlystop_callback],
    verbose = 2
) 

#save model
#my_model_path = os.path.dirname('saved_model/my_model')
#model.save(my_model_path) 
#
#plt.plot(history.history['loss'], label='train')
#plt.plot(history.history['val_loss'], label = 'validation')
#plt.legend()
##
#evaluate model on testing data
y_pred = model.predict(X_test)
y_train_inv = label_column_max_transformer.inverse_transform(y_train.reshape(1,-1))
y_test_inv = label_column_max_transformer.inverse_transform(y_test.reshape(1,-1))
y_pred_inv = label_column_max_transformer.inverse_transform(y_pred)

f2 = plt.figure()
plot_prediction(f2,y_test_inv.flatten(), y_pred_inv.flatten())


#print r2
print(metrics.r2_score(y_test, y_pred))
print(metrics.mean_absolute_error(y_test, y_pred))
print(metrics.mean_squared_error(y_test, y_pred))
