import csv as csv
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train['day'] = np.ones(len(train['Open Date']))

for i in range(len(train)):
	train['day'][i] = (datetime.strptime(train['Open Date'].values[i], "%m/%d/%Y") - datetime.strptime('03/23/2015', "%m/%d/%Y")).days

train['date'] = train['day']/7
train['month'] = train['day']/30
train['revenue1'] = train['revenue']
print(train)

test['day'] = np.ones(len(test['Open Date']))

for i in range(len(test)):
	test['day'][i] = (datetime.strptime(test['Open Date'].values[i], "%m/%d/%Y") - datetime.strptime('03/23/2015', "%m/%d/%Y")).days

test['date'] = test['day']/7
test['month'] = test['day']/30
print(test)



train = train.drop('Open Date', axis=1)
train = train.drop('City', axis=1)
train = train.drop('City Group', axis=1)
train = train.drop('Type', axis=1)
train = train.drop('revenue', axis=1)
train = train.drop('P14', axis=1)
train = train.drop('P15', axis=1)
train = train.drop('P21', axis=1)
train = train.drop('P37', axis=1)
test = test.drop('P21', axis=1)
test = test.drop('Open Date', axis=1)
test = test.drop('City', axis=1)
test = test.drop('City Group', axis=1)
test = test.drop('Type', axis=1)
test = test.drop('P14', axis=1)
test = test.drop('P15', axis=1)
test = test.drop('P37', axis=1)

train = train[train.revenue1 < 16000000]
print(train)

train.to_csv('preprocess.csv')

mean = train.revenue1.mean(axis=1)
std = train.revenue1.std(axis=1)
print(mean)
print(std)
train = train[train.revenue1 < (mean + 3 * std)]
train = train[train.revenue1 > (mean - 3 * std)]
X = train.loc[:,:'month']
print (X)
#regr=KNeighborsClassifier(n_neighbors=6) # default value for n_neighbors is 5
#rng = np.random.RandomState(1)
#regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=10), n_estimators=1000, random_state=rng)

rng = np.random.RandomState(1)
regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=10), n_estimators=1000, random_state=rng)
regr.fit(X, train['revenue1'])

# Predict
y = regr.predict(test)


with open("output1.csv", "w") as res:
	new = csv.writer(res)
	new.writerow(('Id','Prediction'))
	x=0;
	for i in y:
		new.writerow((x,i))
		x=x+1
		


