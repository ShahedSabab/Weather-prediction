# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 00:54:52 2020

@author: sabab
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


#read data
df_temp = pd.read_csv(
    "temperature.csv",
    parse_dates=['datetime'],
    index_col = "datetime"
)

df_pressure = pd.read_csv(
    "pressure.csv",
    parse_dates=['datetime'],
    index_col = "datetime"
)

df_wind_dir = pd.read_csv(
    "wind_direction.csv",
    parse_dates=['datetime'],
    index_col = "datetime"
)

df_wind_speed = pd.read_csv(
    "wind_speed.csv",
    parse_dates=['datetime'],
    index_col = "datetime"
)

df_humidity = pd.read_csv(
    "humidity.csv",
    parse_dates=['datetime'],
    index_col = "datetime"
)

df_weather_des = pd.read_csv(
    "weather_description.csv",
    parse_dates=['datetime'],
    index_col = "datetime"
)


#merge datafrmae and collect data on Toronto
df_temp = df_temp['Toronto']
df_pressure = df_pressure['Toronto']
df_wind_dir = df_wind_dir['Toronto']
df_wind_speed = df_wind_speed['Toronto']
df_humidity = df_humidity['Toronto']
df_weather_des = df_weather_des['Toronto']

merge = pd.merge(df_humidity, df_pressure, suffixes=('_humidity', '_pressure'),left_index=True, right_index=True, how='outer')
merge2 = pd.merge(df_wind_dir, df_wind_speed, suffixes=('_wind_direcion', '_wind_speed'),left_index=True, right_index=True, how='outer')
merge3 = pd.merge(df_temp, df_weather_des, suffixes=('_temperature', '_weather_description'),left_index=True, right_index=True, how='outer')

merge4 = pd.merge(merge, merge2, left_index=True, right_index=True, how='outer')
merge5 = pd.merge(merge4, merge3, left_index=True, right_index=True, how='outer')

merge5['hours'] = merge5.index.hour
merge5['date'] = merge5.index.date
merge5['day_of_week'] = merge5.index.dayofweek
merge5['day_of_month'] = merge5.index.day
merge5['month'] = merge5.index.month

#calculate mean, median, max, mix day basis
mean_temp = merge5.groupby(['date'])['Toronto_temperature'].mean()
convertedtemp_mean = mean_temp-273.15
max_temp = merge5.groupby(['date'])['Toronto_temperature'].max()
convertedtemp_max = max_temp-273.15
min_temp = merge5.groupby(['date'])['Toronto_temperature'].min()
convertedtemp_min = min_temp-273.15
median_pressure = merge5.groupby(['date'])['Toronto_pressure'].median()
median_wind = merge5.groupby(['date'])['Toronto_wind_speed'].median()
median_humidity = merge5.groupby(['date'])['Toronto_humidity'].median()
month = merge5.groupby(['date'])['month'].median()
day_of_week = merge5.groupby(['date'])['day_of_week'].median()
day_of_month = merge5.groupby(['date'])['day_of_month'].median()

#final dataframe
data = pd.DataFrame()
data['month'] = month
data['day_of_week'] = day_of_week
data['day_of_month'] = day_of_month
data['wind_speed'] = median_wind
data['pressure'] = median_pressure
data['humidity'] = median_humidity
data['average_temp'] = convertedtemp_mean
data['max_temp'] = convertedtemp_max
data['min_temp'] = convertedtemp_min

file_name = "toronto_weather.csv"
data.to_csv(file_name, encoding='utf-8', index=True)
#plot to get insight
#sns.lineplot(x=data.index, y='average_temp', data=data)
#sns.pointplot(data=data, x = "month", y = "average_temp")

