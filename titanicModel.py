#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv('train.csv')


data.head()



data['Sex'] = data['Sex'].replace('female',0)
data['Sex'] = data['Sex'].replace('male',1)
data.head()


train, test = train_test_split(data, test_size=0.2, random_state=1)


x_train = train[['Pclass', 'Sex','SibSp','Parch', 'Fare']]
x_test = test[['Pclass', 'Sex','SibSp','Parch', 'Fare']]
y_train = train['Survived']
y_test = test['Survived']


clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
clf.fit(x_train, y_train)

print(clf.feature_importances_)



predictions = clf.predict(x_test)
predictions


from sklearn.metrics import accuracy_score
print("Accuracy: " + str(accuracy_score(y_test,predictions)))

