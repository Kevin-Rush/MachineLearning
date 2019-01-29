#regression simulating stock prices

import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, model_selection, svm # this is for scaling features, it is normally good to have feature between 1 and -1 
from sklearn.linear_model import LinearRegression

df = quandl.get("WSE/TBULL", authtoken="xykwYQEpNJg8xpdTbLGH")

df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

df['HL_PCT'] = (df['High'] - df['Close'])/df['Close']*100
df['PCT_Change'] = (df['Close'] - df['Open'])/df['Open']*100

df = df[['Close','HL_PCT','PCT_Change','Volume']]   #these will be features

#what is the label? we are trying to predict the price

forcast_col = 'Close'
df.fillna(-99999, inplace=True)                     #can't let there be unfilled data therefore fill empty space with True
forcast_out = int(math.ceil(0.01*len(df)))          #we are going to try to predict out 10% of the Data Frame

print(forcast_out)

df['label'] = df[forcast_col].shift(-forcast_out)   #we are looking at the features to try to predict the Close in '10 days' (actually 10%)
df.dropna(inplace=True)

X = np.array(df.drop(['label'],1))                  # will define features
y = np.array(df['label'])                           # will define leabels

X = preprocessing.scale(X)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print(accuracy)