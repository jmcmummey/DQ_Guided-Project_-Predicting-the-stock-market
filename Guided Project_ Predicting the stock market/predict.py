import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LinearRegression
from datetime import datetime

sp500 = pd.read_csv('sphist.csv')

#columns = [Date,Open,High,Low,Close,Volume,Adj Close]

sp500['Date'] = pd.to_datetime(sp500['Date'])
sp500.sort_values('Date',ascending=True,inplace=True)
print('SP 500 Data Set Info:')
print(sp500.info())
print()
print('First 3 Data Points')
print(sp500.head(3))

#tools to calculate
#average price for the last 5 days / average price for the last year
#average std for the last 5 days / average std for the last year
#std/avg price for the last 30 days

sp500i = sp500.set_index('Date')

d5 = sp500i['Close'].rolling('5d') #five days
d30 = sp500i['Close'].rolling('30d') #thirty days
d365 = sp500i['Close'].rolling('365d') #one year
v5 = sp500i['Volume'].rolling('5d')
v60 = sp500i['Volume'].rolling('60d')
sp500i['a5'] = d5.mean()
sp500i['a5_a365'] = d5.mean()/d365.mean()
sp500i['s5_s365'] = d5.std()/d365.std()
sp500i['s30_a30'] = d30.std()/d30.mean()
sp500i['v5_v60'] = v5.mean()/v60.mean()
sp500i['hl_365'] = d365.max()/d365.min()



#shift the data forward 1
sp500i.iloc[:,-6:] = sp500i.iloc[:,-6:].shift(1)
#remove first year
filt = sp500i.index > datetime(year=1951, month=1, day=2)
sp500i = sp500i[filt]
print(sp500i.shape[0])
#remove any NAs
sp500i.dropna(axis=0,inplace=True)


print('Running model...')

#generate test and training data
train = sp500i[sp500i.index < datetime(year=2013, month=1, day=1)]
test = sp500i[sp500i.index >= datetime(year=2013, month=1, day=1)]

features = ['a5','a5_a365','s5_s365','s30_a30','v5_v60','hl_365']
lr = LinearRegression()
lr.fit(train[features],train['Close'])
pred = lr.predict(test[features])

print(pred[-5:])
print(test['Close'].tail(5))
print('Mean Absolute Error: %0.2f' % mae(pred,test['Close']))
sind = sp500i.index.get_loc(datetime(year=2013, month=1, day=3))
#make the prediction 5 days ahead
pred5 = []
for i in range(0,100):
    lr = LinearRegression()
    lr.fit(sp500i.iloc[:(sind+i)][features],sp500i.iloc[:(sind+i)]['Close'])
    print(lr.predict(sp500i.iloc[(sind+i)+4][features].reshape(1,-1)),sp500i.iloc[(sind+i)+4]['Close'].reshape(1,-1))