# Run after 'handling date and time.py' script
# This Script clears the ['s_lat','s_long','e_lat','e_long'] from outlier values.
# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("bus_data_final.csv")
df['s_lat'].value_counts()
# Clearing s_lat (Automatically clears s_long of irrelevant values)
df = df.set_index("s_lat")
df = df.drop(99.99999999, axis = 0)
df = df.drop(0.000000, axis = 0)
df = df.reset_index()
# Clearing e_lat(Automatically clears e_long of irrelevant values)
df = df.set_index("e_lat")
df = df.drop(0.00, axis = 0)
#df = df.drop(99.99999999, axis = 0)
df = df.reset_index()

plt.show(sns.jointplot(x = "e_lat", y = "e_long",data = df))

plt.show(sns.jointplot(x = "s_lat", y = "s_long",data = df))



# Deleting negative values and outlier values from the 'time_diff' column

print(df["time_diff"].min())
df = df[df.time_diff > 0] # Rows where end_time - start_time < 0 have been deleted.

df = df.dropna()
df.info()
sns.distplot(df['time_diff'])
df['time_diff'].mean() # Any values above 100 are probably outliers and wrong
df["time_diff"].describe(percentiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.8,0.9,0.95],include = 'all')
# More than 95 percentile of data is around 51 so taking only the points below 55 is safe.
df[df.time_diff > 55]
# Only 16k points out of some 1.4 million points are above 55. these can be dropped
df = df[df.time_diff <= 55]
sns.distplot(df['time_diff'])
# We can still see that the curve isnt gaussian.
df.to_csv("final_ver1.2.csv")
#
#
#
