import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

etf = pd.read_csv(r'.\zz500etf.csv',index_col=0,parse_dates=True)

etf['one_month_return_past'] = etf['close'].shift(22)/etf['close'] - 1
etf['one_year_return_past'] = etf['close'].shift(252)/etf['close'] - 1
etf['one_month_return_future'] = etf['close'].shift(-22)/etf['close'] - 1
etf['vol'] = etf['close'].rolling(22).var()

etf_train = etf.dropna().iloc[0:-252,:]
etf_train['one_year_le_one_month'] = (np.sign(etf_train['one_month_return_past'])*np.sign(etf_train['one_month_return_future']) >= 
                                    np.sign(etf_train['one_year_return_past']*np.sign(etf_train['one_month_return_future']))).apply(int)
etf_train['one_year_le_one_month'] = etf_train['one_year_le_one_month'].where(etf_train['one_year_le_one_month']==1,-1)

from sklearn import tree 
clf = tree.DecisionTreeClassifier(max_depth=3) 
clf = clf.fit(etf_train[['vol']],etf_train['one_year_le_one_month']) 

etf_test = etf.dropna().iloc[-252:-1,:]
etf_test['position'] = clf.predict(etf_test[['vol']]) * np.sign(etf_test['one_month_return_past']).shift(1)
etf_test['everyday_position'] = etf_test['position'].rolling(22).sum()
etf_test['pnl'] = etf_test['everyday_position']*etf_test['close'] - etf_test['everyday_position'].shift(1)*etf_test['close'].shift(1)
etf_test['pnl'].cumsum().plot()
