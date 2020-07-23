import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("creditcard.csv")

print(data.head())
print(data.info())

cols = data.columns.tolist()

cols = [ x for x in cols if x not in ['Class']]

target = 'Class'

X = data[cols]
Y = data[target]

print(X.shape)
print(Y.shape)

xData = X.values
yData = Y.values

x_train,x_test,y_train,y_test = train_test_split(xData,yData,test_size = 0.2, random_state=42)

rf = RandomForestClassifier()
rf.fit(x_train,y_train)

y_pred = rf.predict(x_test)

acc_rc = accuracy_score(y_test, y_pred)
print("Accuracy is {}".format(acc_rc))
