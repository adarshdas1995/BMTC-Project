# This runs after final.py script
# This script is run to handle the timestamp data columns ['s_time','e_time']
# We label encode the date parameter and convert the time parameter into seconds. The seconds value is counted from 00:00:00 time period.

import pandas as pd

df = pd.read_csv('final_bus_data.csv')

# Deal with dates
df['s_date'] = pd.to_datetime(df['s_time']).dt.date
df['e_date'] = pd.to_datetime(df['e_time']).dt.date

# Deal with time.
df['s_time'] = df['s_time'].apply(pd.Timestamp)
df['e_time'] = df['e_time'].apply(pd.Timestamp)

# Create s_clock and e_clock parameters which are in the string form HH:MM:SS
df['s_clock'] = df['s_time'].dt.strftime('%H:%M:%S')
df['e_clock'] = df['e_time'].dt.strftime('%H:%M:%S')

# Split it into hours:minutes:seconds columns
df[['H1','M1','S1']] = df['s_clock'].str.split(':', expand = True)
df[['H2','M2','S2']] = df['e_clock'].str.split(':', expand = True)


df.head()

# Convert the data from string to numeric
df['H1'] = pd.to_numeric(df['H1'], errors='coerce')
df['M1'] = pd.to_numeric(df['M1'], errors='coerce')
df['S1'] = pd.to_numeric(df['S1'], errors='coerce')
df['H2'] = pd.to_numeric(df['H2'], errors='coerce')
df['M2'] = pd.to_numeric(df['M2'], errors='coerce')
df['S2'] = pd.to_numeric(df['S2'], errors='coerce')

# Convert Hours and Minutes into seconds
df['H1'] = 3600*df['H1']
df['M1'] = 60*df['M1']
df['M2'] = 60*df['M2']
df['H2'] = 3600*df['H2']

df.head(100)

# Convert s_clock and e_clock to seconds
df['s_clock'] = df['H1'] + df['M1'] + df['S1']
df['e_clock'] = df['H2'] + df['M2'] + df['S2']

df.head(100)

# Drop un necessary columns
df = df.drop(columns = ["s_time","e_time","H1","H2","M1","M2","S1","S2"])
df.head()

# Use Label Encoding on the dates
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df['s_date'] = le.fit_transform(df['s_date'])
df['e_date'] = le.fit_transform(df['e_date'])

# Define The 'time_diff' column
df["time_diff"] = df["e_clock"] - df["s_clock"]

# Write to a new csv
df.to_csv("bus_data_final.csv",index = False)

#


