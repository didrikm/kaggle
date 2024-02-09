#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 18:58:56 2024

@author: didrikmolinder
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

dataset = pd.read_csv("train.csv")
dataset_test = pd.read_csv("test.csv")
X = dataset.iloc[:, [2,4,5,6,7,11]].values
y = dataset.iloc[: , 1].values
X_test = dataset_test.iloc[:, [1,3,4,5,6,10]].values

#Label encoding gender
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])
X_test[:, 1] = le.transform(X_test[:, 1])

#OHE of ticket class
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
X_test = np.array(ct.transform(X_test))

#OHE of port of embarcation
ct2 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(categories=[['S','C','Q']],handle_unknown='ignore'), [7])], remainder='passthrough')
X = np.array(ct2.fit_transform(X))
X_test = np.array(ct2.transform(X_test))

#Imputing of missing age
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 7:8])
X[:, 7:8] = imputer.transform(X[:, 7:8])
X_test[:, 7:8] = imputer.transform(X_test[:, 7:8])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(16, activation = 'relu'))
ann.add(tf.keras.layers.Dense(16, activation = 'relu'))
ann.add(tf.keras.layers.Dense(16, activation = 'relu'))

ann.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier()
ann.fit(X, y, batch_size = 32, epochs = 100)

#KFCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kf = KFold(shuffle=True, random_state=42)
score = cross_val_score(ann, X, y, cv=kf)


# y_pred = ann.predict(X_test)
# rounded_array = np.floor(y_pred + 0.5).astype(int)

# submission = pd.read_csv("gender_submission.csv")
# df = pd.DataFrame(submission)
# df["Survived"] = rounded_array
# file_name = 'submission.csv'
# df.to_csv(file_name, index=False)
# print(f'{file_name} created successfully.')

