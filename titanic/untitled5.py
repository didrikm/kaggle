#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 16:10:53 2024

@author: didrikmolinder
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from autogluon.tabular import TabularDataset, TabularPredictor

dataset = TabularDataset("train.csv")
dataset_test = TabularDataset("test.csv")

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

y = dataset.loc[:,['Survived']]
X = dataset.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket', 'Fare', 'Cabin', ])
X_test = dataset_test.drop(columns=['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', ])

X = pd.get_dummies(X, columns=['Sex', 'Pclass', 'Embarked', 'Title'])
X_test = pd.get_dummies(X_test, columns=['Sex', 'Pclass', 'Embarked', 'Title'])

X['Age'] = X['Age'].fillna(X['Age'].mean())
X_test['Age'] = X_test['Age'].fillna(X_test['Age'].mean())

sc = StandardScaler()
X_scaled = sc.fit_transform(X)
X_test_scaled = sc.transform(X_test)
X = TabularDataset(X_scaled, columns=X.columns)
X_test = TabularDataset(X_test_scaled, columns=X_test.columns)

X['Survived'] = y

model = TabularPredictor(label='Survived').fit(X, presets='best_quality')
prediction = model.predict(X_test)

rounded_array = np.floor(prediction + 0.5).astype(int)

submission = pd.read_csv("gender_submission.csv")
df = pd.DataFrame(submission)
df["Survived"] = rounded_array
file_name = 'submission8.csv'
df.to_csv(file_name, index=False)
print(f'{file_name} created successfully.')

ld = model.leaderboard(X)









