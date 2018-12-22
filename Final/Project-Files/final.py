# 3rd Script to be run.
# We now have a dataset of 400 buses.
# We will create a new dataset. Each row of the dataset will correspond to a single bus id start and stop location and timestamp.
# Final dataset columns are ['s_lat','s_long','s_time','e_lat','e_long','e_time'].

import pandas as pd
import random
import gc
df_final = pd.DataFrame(columns = ['s_lat','s_long','s_time','e_lat','e_long','e_time'])
for i in range(100,500):
    try:
        k = 0
        df = pd.read_csv('bus{}.csv'.format(i))
        print('for bus{}'.format(i))
        print(i)
        df = df.sort_values(by = 'timestamp')
        df_buffer = pd.DataFrame(index = range(0,100000),columns = ['s_lat','s_long','s_time','e_lat','e_long','e_time'])
        try:
            for l in range(0,len(df['timestamp'].unique()),random.randint(7,10)):
            # Start columns
                df_buffer['s_lat'].iloc[k] = df['lat'].iloc[l]
                df_buffer['s_long'].iloc[k] = df['long'].iloc[l]
                df_buffer['s_time'].iloc[k] = df['timestamp'].iloc[l]
            # End columns
                df_buffer['e_lat'].iloc[k] = df['lat'].iloc[l+5]
                df_buffer['e_long'].iloc[k] = df['long'].iloc[l+5]
                df_buffer['e_time'].iloc[k] = df['timestamp'].iloc[l+5]
                k = k+1
                if k > 5000:
                    break
            df_buffer = df_buffer.dropna()
            df_final = df_final.append(df_buffer)
            del df
            gc.collect()
        except IndexError:
            print('IndexError')
    except FileNotFoundError:
        i = i+1
        print('Inside exception')
        
df_final.to_csv('final_bus_data.csv', index = False)

