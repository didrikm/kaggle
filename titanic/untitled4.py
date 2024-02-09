#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:02:05 2024

@author: didrikmolinder
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from scikeras.wrappers import KerasClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import GradientBoostingClassifier

# =============================================================================
# DATA PRE-PROCESSING
# =============================================================================

dataset = pd.read_csv("train.csv")
dataset_test = pd.read_csv("test.csv")

#Normalization of honorifics (needed to be done before being converted from dataframe to numpy array)
dataset['Title'] = dataset.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
dataset_test['Title'] = dataset_test.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
normalized_titles = {
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"
}
dataset.Title = dataset.Title.map(normalized_titles)
dataset_test.Title = dataset_test.Title.map(normalized_titles)

X = dataset.iloc[:, [2,4,5,6,7,11,12]].values
y = dataset.iloc[: , 1].values
X_test = dataset_test.iloc[:, [1,3,4,5,6,10,11]].values

#LE gender
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])
X_test[:, 1] = le.transform(X_test[:, 1])

#OHE ticket class
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
X_test = np.array(ct.transform(X_test))

#OHE port of embarcation
ct2 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(categories=[['S','C','Q']],handle_unknown='ignore'), [7])], remainder='passthrough')
X = np.array(ct2.fit_transform(X))
X_test = np.array(ct2.transform(X_test))

#OHE honorific
ct3 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(categories=[['Mr', 'Miss', 'Mrs', 'Master', 'Officer', 'Royalty']],handle_unknown='ignore'), [10])], remainder='passthrough')
X = np.array(ct3.fit_transform(X))
X_test = np.array(ct3.transform(X_test))

#Imputation of age
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 13:14])
X[:, 13:14] = imputer.transform(X[:, 13:14])
X_test[:, 13:14] = imputer.transform(X_test[:, 13:14])

#Standardization
sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)

# =============================================================================
# MODELS
# =============================================================================

# model1 = LogisticRegression(random_state=0)
# model1.fit(X, y)

model2 = RandomForestClassifier(n_estimators=50)
# model2.fit(X, y)

model3 = XGBClassifier()
# model3.fit(X, y)

model4 = tf.keras.models.Sequential()
model4.add(tf.keras.layers.Dense(16, activation = 'relu'))
model4.add(tf.keras.layers.Dense(16, activation = 'relu'))
model4.add(tf.keras.layers.Dense(16, activation = 'relu'))
model4.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
model4.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model4.fit(X, y, batch_size = 32, epochs = 100)
model4b = KerasClassifier(model=model4, optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# =============================================================================
# CROSS-VALIDATION
# =============================================================================

# score1 = cross_val_score(estimator=model1, X=X, y=y, cv=5)
# score2 = cross_val_score(estimator=model2, X=X, y=y, cv=5)
# score3 = cross_val_score(estimator=model3, X=X, y=y, cv=5)
# score4 = cross_val_score(estimator=model4b, X=X, y=y, cv=5)

# print("Mean score 1: {:.2f}%".format(score1.mean()*100))
# print("Mean score 2: {:.2f}%".format(score2.mean()*100))
# print("Mean score 3: {:.2f}%".format(score3.mean()*100))
# print("Mean score 4: {:.2f}%".format(score4.mean()*100))

# =============================================================================
# STACKING
# =============================================================================

estimators = [('forest', model2), ('XG', model3), ('FCANN', model4b)] #('log', model1), 
final_estimator = GradientBoostingClassifier() 
final_classifier = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
final_classifier.fit(X, y)
score5 = cross_val_score(estimator=final_classifier, X=X, y=y, cv=5)

# =============================================================================
# PREDICTION & CSV EXPORT
# =============================================================================

prediction = final_classifier.predict(X_test)
rounded_array = np.floor(prediction + 0.5).astype(int)

submission = pd.read_csv("gender_submission.csv")
df = pd.DataFrame(submission)
df["Survived"] = rounded_array
file_name = 'submission6.csv'
df.to_csv(file_name, index=False)
print(f'{file_name} created successfully.')









