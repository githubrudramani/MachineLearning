# -*- coding: utf-8 -*-
"""TensorFlowRegression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17lnMGT1bVlsJMHo9za9v5OYg_8s3KhPZ
"""

!pip install tensorflow-gpu

"""# House price prediction using ANN regressin model

"""

# mounting drive
from google.colab import drive
drive.mount('/content/drive')

dir = '/content/drive/MyDrive/UdemyRayan/'

"""Importing libraries"""

import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
sns.set()

## importing data
house_df = pd.read_csv(dir+"Data/kc_house_data.csv")
house_df.info()

house_df.describe()

## to check null elements
sns.heatmap(house_df.isnull())

# correlation

f.ax = plt.subplots(figsize = (20,20))
sns.heatmap(house_df.corr(), annot=True)

house_df.columns

sns.scatterplot(x = "sqft_living", y= "price", hue ="bedrooms", data = house_df)

house_df.hist(bins=10, figsize=(20,20), color = "purple")

sns.pairplot(house_df)

"""Data cleaning"""

selected_features = ['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'sqft_above', 'sqft_basement']
house_df.columns

X = house_df[selected_features]

y = house_df["price"]
y

X.shape

y.shape

# Normalize data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled.shape

# reshape our out put as (no of samples,    1)
y = y.values.reshape(-1,1)
y.shape

y_scaled = scaler.fit_transform(y)
y_scaled

## split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.25)

X_train.shape

X_test.shape

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units = 100, activation= 'relu', input_shape = (7,)))
model.add(tf.keras.layers.Dense(units = 100, activation = 'relu'))
model.add(tf.keras.layers.Dense(units = 100, activation= 'relu'))
model.add(tf.keras.layers.Dense(units = 1, activation = "linear"))

model.summary()

## compile model
model.compile(optimizer = 'Adam', loss = 'mean_squared_error')

## Train model
epoch_hist = model.fit(X_train, y_train, epochs = 100, batch_size = 50, validation_split= 0.2)

epoch_hist.history.keys()

plt.plot(epoch_hist.history["loss"])
plt.plot(epoch_hist.history["val_loss"])

## prediction
y_predict = model.predict(X_test)
plt.plot(y_test, y_predict, '^')

## other evaluation parameters
# plot original data 
y_predict_raw = scaler.inverse_transform(y_predict)
y_test_raw = scaler.inverse_transform(y_test)
plt.plot(y_test_raw, y_predict_raw, "^", color = 'r')
plt.xlabel("Model Prediction")
plt.ylabel("True Values")

k = X_test.shape[1] # features
n = len(X_test) # data point
## adjusted R2
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt

RMSE = float(format(np.sqrt(mean_squared_error(y_test_raw, y_predict_raw)), '0.3f'))

MSE = mean_squared_error(y_test_raw, y_predict_raw)

MAE = mean_absolute_error(y_test_raw, y_predict_raw)
MAE

r2 = r2_score(y_test_raw, y_predict_raw)
adj_r2 = 1- (1-r2)*(n-1)/(n-k-1)

r2

adj_r2

## very low r2 
## try including other models