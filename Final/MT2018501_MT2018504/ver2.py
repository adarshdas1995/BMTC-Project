# 2nd Script to be run.
# Script to extract a set of buses from chunked dataset.
# The 'bus_id_3.csv' is a manually cleaned bus id dataset
# The script selects 400 buses from the bus_id_3.csv files and creates their own dataset consisting of bus_id,lat,long and timestamp columns
import pandas as pd
import gc
bus_df = pd.read_csv('bus_id_3.csv')
for k in range(100,500):
    bus_id = bus_df.iloc[k]
    count = 1
    print("For bus {}".format(int(bus_df.iloc[k])))
    for i in range(0,22):
        c = i
        df = pd.read_csv('chunk{}.csv'.format(i))
        df.set_index("bus_id", inplace=True)
        #df.loc[150218715].to_csv('150218715.csv')
        try:
            df = df.loc[bus_id]
            if i is 0:
                df1 = df.loc[bus_id]
                print("Inside i = 0")
            else:
                df2 = df.loc[bus_id]
                if i is 1:
                    df3 = df2.append(df1)
                else:
                    df3 = df2.append(df3)
                print("inside i not 0")
            print(i)
            del df
            gc.collect()
        except KeyError:
            print("KeyError")
            count = 0
            break
        except TypeError:
            print("TypeError")
            count = 0
            break
    if (count > 0) or (c > 10):
        print("Writing to csv")
        df3 = df3.drop(columns = ['Unnamed: 0'])
        df3.to_csv('/home/shatterstar/BMTC/sorted_according_to_bus_id/bus{}.csv'.format(k), index = False)
        del df1,df2,df3
#
#
#
