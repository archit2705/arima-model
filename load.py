import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

file = pd.read_csv('data.csv')
fl = file.groupby('item_id')
# for i in range(1,11):
grp_1 = fl.get_group(1)
w=np.array(grp_1['week'])
p=np.array(grp_1['profit'])
print(type(w))
wp = np.zeros((w.shape[0], 2))
wp[:, 0] = w
wp[:,1] = p

# wp = pd.concat([w,p],axis=1)
print(wp)
# plt.scatter(p, w)
# plt.show()
p = range(0, 2)
d = range(0, 2)
q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

warnings.filterwarnings("ignore") # specify to ignore warning messages

for param in pdq:
    for param_seasonal in seasonal_pdq:
        # wp = pd.dataframe(wp)
        mod = sm.tsa.statespace.SARIMAX(wp,
                                        order=param,
                                        seasonal_order=param_seasonal,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)

        results = mod.fit()
        print('a')
        print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
