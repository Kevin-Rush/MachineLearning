#regression simulating stock prices

import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm # this is for scaling features, it is normally good to have feature between 1 and -1 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
from matplotlib import style
import pickle 

style.use('ggplot')

df = quandl.get("WIKI/GOOGL", authtoken="xykwYQEpNJg8xpdTbLGH")

#print(df)

df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

df['HL_PCT'] = (df['High'] - df['Close'])/df['Close']*100       #percent change between the high and the close
df['PCT_Change'] = (df['Close'] - df['Open'])/df['Open']*100    #percent change between open and close

#objective is to idenfiy what featuers will directly affect price?
#           price   x           x           x
df = df[['Close','HL_PCT','PCT_Change','Volume']]               #these will be features

#what is the label? we are trying to predict the price

forcast_col = 'Close'                               #forcast column = the close price
df.fillna(-99999, inplace=True)                     #can't let there be unfilled data therefore fill empty space with True
forcast_out = int(math.ceil(0.1*len(df)))          #we are going to try to predict out 10% of the Data Frame

#print(forcast_out)

df['label'] = df[forcast_col].shift(-forcast_out)   #we are looking at the features to try to predict the Close in '10 days' (actually 10%)

X = np.array(df.drop(['label'],1))                  # will define features
X = preprocessing.scale(X)
X_lately = X[-forcast_out:]
X = X[:-forcast_out]                                # will define leabels


#print(X)
#print(X_lately) 

df.dropna(inplace=True)
y = np.array(df['label'])    

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)  #only sending 20% of data??

# clf = LinearRegression(n_jobs=-1)
# clf.fit(X_train, y_train)
# with open('linearregression.pickle', 'wb') as f: #we are now saving the classifier so we can avoid the training step b/c we can save the classifier after it's been trained
#     pickle.dump(clf, f)                            #should still retrain it every now and again, but this way we won't retrain everytime we run the program

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)



accuracy = clf.score(X_test, y_test)


forcast_set = clf.predict(X_lately)         #this is where the prediction is happening 

#print(forcast_set, accuracy, forcast_out)

df['Forcast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forcast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i] #this is a list of values that are np.nan  + i (i is the forcast) 

#print(df.tail())

df ['Close'].plot() 
df['Forcast'].plot() 
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show() 