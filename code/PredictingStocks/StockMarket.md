# Jupyter Notebook for Creating a Deep Learning Model to Predict Stocks

### 3/20/2020

Predicting time-series data can be challenging depending on the priors. Here we are going to look at two cases, both using ARIMA models to see how predictions are affected.

Data taken from https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python/



```python
# import packages
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

df = pd.read_csv('data.csv')

df.head() # prints the head. Note that df.head gives a nonformatted output. df.head() is nice and pretty
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Last</th>
      <th>Close</th>
      <th>Total Trade Quantity</th>
      <th>Turnover (Lacs)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-10-08</td>
      <td>208.00</td>
      <td>222.25</td>
      <td>206.85</td>
      <td>216.00</td>
      <td>215.15</td>
      <td>4642146.0</td>
      <td>10062.83</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-10-05</td>
      <td>217.00</td>
      <td>218.60</td>
      <td>205.90</td>
      <td>210.25</td>
      <td>209.20</td>
      <td>3519515.0</td>
      <td>7407.06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-10-04</td>
      <td>223.50</td>
      <td>227.80</td>
      <td>216.15</td>
      <td>217.25</td>
      <td>218.20</td>
      <td>1728786.0</td>
      <td>3815.79</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-10-03</td>
      <td>230.00</td>
      <td>237.50</td>
      <td>225.75</td>
      <td>226.45</td>
      <td>227.60</td>
      <td>1708590.0</td>
      <td>3960.27</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-10-01</td>
      <td>234.55</td>
      <td>234.60</td>
      <td>221.05</td>
      <td>230.30</td>
      <td>230.90</td>
      <td>1534749.0</td>
      <td>3486.05</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

plt.style.use('dark_background')
plt.figure(figsize=(8,4))
plt.plot(df.Close, label='Close price history')
```




    [<matplotlib.lines.Line2D at 0x212db65a460>]




![svg](StockMarket_files/StockMarket_2_1.svg)



```python
# ARIMA Model

import pmdarima as pm
from pmdarima.model_selection import train_test_split

# split the data
train, test = train_test_split(df.Close, train_size=500)

# train the model
model = pm.auto_arima(np.flip(train), seasonal = False)

# make forecasts
forecasts = model.predict(test.shape[0])


# Visualize the forecasts (blue=train, green=forecasts)
x = np.arange(df.Close.shape[0])
plt.plot(x[:500], np.flip(train))
plt.plot(x[500:], forecasts, c='green')
plt.show()

```


![svg](StockMarket_files/StockMarket_3_0.svg)

