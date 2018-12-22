# 1st Script to be run to separate w1.csv file.
# This code does the following things
# Reads chunks of '10000000' rows from the w1 dataset given to us. We manage to create 22 chunks #each of size '610 MB' from the 14GB data given to us
# Append the 'index','bus_id','lat','long','p1','p2','timestamp' to top of each csv
# Create a set of unique bus ID's in all the chunks 
# Store it in 'bus_id.csv' file. But it stores the set as a string and not a pandas series. #Manual cleaning needs to be done

import pandas as pd
import fileinput
import pandas as pd
import csv
B = {}
for i,chunk in enumerate(pd.read_csv('w1.csv',chunksize = 10000000)):
	chunk.to_csv('chunk{}.csv'.format(i))
	for line in fileinput.input(files=['chunk{}.csv'.format(i)],inplace=True):
		if fileinput.isfirstline():
			print('index,bus_id,lat,long,p1,p2,timestamp')
		print(line)
	df = pd.read_csv('chunk{}.csv'.format(i))
	A = set(df['bus_id'].unique())
	B = set(B).union(set(A))
	print(len(B))
	df2 = df.drop(columns=['index','p1','p2'],axis=1)
	df2.to_csv('chunk{}.csv'.format(i))
with open('bus_id.csv','w') as file:
	file.write((str(B)))
