import pandas as pd

train = pd.read_csv("train.csv")


train["Datetime"] = pd.to_datetime(train.Datetime, format = "%d-%m-%Y %H:%M")
train.index = train.Datetime


train.drop(["ID", "Datetime"], axis = 1, inplace = True)


daily_train = train.resample("D").sum()


daily_train["ds"] = daily_train.index
daily_train["y"] = daily_train.Count


daily_train.drop(["Count"], axis = 1, inplace = True)


from fbprophet import Prophet

m = Prophet(yearly_seasonality = True, seasonality_prior_scale = 0.1)
m.fit(daily_train)
future = m.make_future_dataframe(periods = 213)

forecast = m.predict(future)
print(forecast)

m.plot(forecast)
m.plot_components(forecast)