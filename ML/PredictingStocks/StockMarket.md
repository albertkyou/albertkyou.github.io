# ARIMA Models to Predict Stocks

### 3/20/2020

Predicting time-series data can be challenging depending on the priors. Here we are going to look at two cases, both using ARIMA models to see how predictions are affected.


## ARIMA On Non-Periodic Data

We begin by importing important packages such as numpy and pandas. 
We'll analyze data found [here](https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python/).

Note that the data has already been downloaded locally and renamed as 'data.csv'. 



```python
# import packages
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

df = pd.read_csv('data.csv')

# prints the head. Note that df.head gives a nonformatted output. df.head() is nice and pretty
df.head()
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



Now that we see what the data looks like, we need to reformat the data slightly. As is common with time-series data, we first convert the index into dates using `to_datetime()`.

Finally, we'll plot the `Close` prices for each datapoint.


```python
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

plt.figure(figsize=(8,4))
plt.plot(df.Close, label='Close price history')
```




    [<matplotlib.lines.Line2D at 0x1ba9d042a00>]




![svg](StockMarket_files/StockMarket_4_1.svg)


Now it's time to train the ARIMA model! `pmdarima` (formerly pyramid) is based on `statsmodels` and offers more functionality for training ARIMA models.



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
plt.figure(figsize=(8,4))
plt.plot(x[:500], np.flip(train))
plt.plot(x[500:], forecasts, c='green')
plt.show()

```


![svg](StockMarket_files/StockMarket_6_0.svg)


We now see that the forecast is quite poor. However, because the data does not show an obvious periodic trend, a horizontal line is a better prediction than other predictions you might expect.

Let's see next what happens when we use ARIMA on a dataset with more periodic trends.


## ARIMA On Periodic Data

Now that we've seen how an ARIMA model (doesn't) work on data with weird trends, let's see how it performs given data that we know to be periodic.


I did a quick search for `christmas` in the US on Google Trends and downloaded the data for the past 16 years.

Let's load that in and see what it plots!


```python
df = pd.read_csv('christmas.csv')

df.index = df['Month']

plt.figure(figsize=(8,4))
plt.plot(df['christmas: (United States)'])
plt.xlabel('Time')
```




    Text(0.5, 0, 'Time')




![svg](StockMarket_files/StockMarket_9_1.svg)


So we see that the data is periodic with no obvious trend over time. Let's see what our ARIMA model does now!


```python
# split the data as before
train, test = train_test_split(df['christmas: (United States)'], train_size = 50)

# train the model
model = pm.auto_arima(train, seasonal = True, m=12)

# make forecasts
forecasts = model.predict(test.shape[0])


# Visualize
x = np.arange(len(df))
plt.figure(figsize=(8,4))
plt.plot(x,df['christmas: (United States)'], c='orange')
plt.plot(x[:50], train)
plt.plot(x[50:], forecasts, c='green')

plt.legend(['Actual', 'Training Data', 'Forecast'])
plt.show()
```

    C:\Users\youal\AppData\Local\Programs\Python\Python38\lib\site-packages\statsmodels\base\model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      warn("Maximum Likelihood optimization failed to converge. "
    


![svg](StockMarket_files/StockMarket_11_1.svg)


From this we can see that our model did a great job at predicting what happens on a 12 month cycle. However, it failed to predict any changes in peak search count over time. That's one of the shortcomings of ARIMA models. While they can be powerful tools to find periodic trends, they work less well at capturing subtle trends (though, most models would struggle with this particular data).
