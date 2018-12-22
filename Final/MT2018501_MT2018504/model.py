# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
import pickle
import gc
min_max_scaler = MinMaxScaler()

class preprocessing_train():
    def __init__(self,TrainData):
        self.TrainData = TrainData

    def fe_training_data(self):
        print(self.TrainData.head())

        # Drop useless columns
        self.TrainData = self.TrainData.drop(columns = ['Unnamed: 0'])
        #plt.show(sns.jointplot(x="s_lat", y="s_long",height = 50, ratio = 1, data=self.TrainData))
        #plt.show(sns.jointplot(x="e_lat", y="e_long",height = 50, ratio = 1, data=self.TrainData))

        # Shuffle Dataset to avoid biasing
        self.TrainData = self.TrainData.sample(frac=1).reset_index(drop=True)

        # Distribution Plots Before Scaling
        #self.TrainData.hist(bins=500, figsize=(20,15))
        #plt.show()

        df2 = self.TrainData # Creating a copy

        # Normalizing the ['s_lat','s_long','e_lat','e_long'] columns between (0,1)
        min_max_scaler = MinMaxScaler()
        df2 = pd.DataFrame(df2,columns = ['s_lat','s_long','e_lat','e_long']) 
        df2 = min_max_scaler.fit_transform(df2) # apply transformation but output is a np.array
        df2 = pd.DataFrame(df2,columns = ['s_lat','s_long','e_lat','e_long']) # convert np.array to df
        df2['s_date'] = self.TrainData['s_date']
        df2['s_clock'] = self.TrainData['s_clock']
        df2['e_date'] = self.TrainData['e_date']
        df2['e_clock'] = self.TrainData['e_clock'] # DO NOT SCALE THE TARGET FEATURE
        df2['time_diff'] = self.TrainData['time_diff']

        # Distribution after Scaling
        #df2.hist(bins=500, figsize=(20,15))
        #plt.show()
        print(df2.head())
        # Check Correlation
        corr = df2.corr()
        print(corr)
        # Dividing into train and test
        x_train = df2.drop(columns = ["e_clock",'time_diff','s_date','e_date'])
        y_train = df2["e_clock"]
        return x_train,y_train
class training_models():
    def __init__(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train
    
    
    def train_linear_regression(self):
        lr = LinearRegression()
        lr.fit(self.x_train,self.y_train)
        scores = cross_val_score(lr, self.x_train, self.y_train, scoring='neg_mean_absolute_error',cv = 3)
        print(scores)
        cvs_lr = scores.mean()
        print(scores.mean())
        return cvs_lr,lr
    
    
    def train_decision_tree_regressor(self):
        dtr = DecisionTreeRegressor()
        dtr.fit(self.x_train,self.y_train)
        scores = cross_val_score(dtr, self.x_train, self.y_train, scoring='neg_mean_absolute_error',cv = 3)
        print(scores)
        cvs_dtr = scores.mean()
        print(scores.mean())
        return cvs_dtr,dtr
    
    
    def train_random_forest_regressor(self):
        rfr = RandomForestRegressor()
        rfr.fit(self.x_train,self.y_train)
        scores = cross_val_score(rfr, self.x_train, self.y_train, scoring='neg_mean_absolute_error',cv = 3)
        print(scores)
        cvs_rfr = scores.mean()
        print(scores.mean())
        return cvs_rfr,rfr
    
    
    def train_xgb_regressor(self):
        xgb = XGBRegressor()
        xgb.fit(self.x_train,self.y_train)
        scores = cross_val_score(xgb, self.x_train, self.y_train, scoring='neg_mean_absolute_error',cv = 3)
        print(scores)
        cvs_xgb = scores.mean()
        print(scores.mean())
        return cvs_xgb,xgb


class preprocessing_test():
    def __init__(self,TestData):
        self.TestData = TestData
    def preprocessing(self):
        print(self.TestData.head())
        # Breaking down timestamps


        self.TestData['s_date'] = pd.to_datetime(self.TestData['TimeStamp']).dt.date
        self.TestData['TimeStamp'] = self.TestData['TimeStamp'].apply(pd.Timestamp)
        self.TestData['s_clock'] = self.TestData['TimeStamp'].dt.strftime('%H:%M:%S')
        self.TestData[['H1','M1','S1']] = self.TestData['s_clock'].str.split(':', expand = True)

        self.TestData['H1'] = pd.to_numeric(self.TestData['H1'], errors='coerce')
        self.TestData['M1'] = pd.to_numeric(self.TestData['M1'], errors='coerce')
        self.TestData['S1'] = pd.to_numeric(self.TestData['S1'], errors='coerce')

        self.TestData['H1'] = 3600*self.TestData['H1']
        self.TestData['M1'] = 60*self.TestData['M1']

        self.TestData['s_clock'] = self.TestData['H1'] + self.TestData['M1'] + self.TestData['S1']

        print(self.TestData.head())

        self.TestData = self.TestData.drop(columns = ['H1','M1','S1','TimeStamp'])
        
        #pd.set_option('display.max_rows', 500) 
        self.TestData = self.TestData.drop(columns = 's_date')
        df_test_3 = self.TestData
        self.TestData = self.TestData.drop(columns = 'Id')

        print(self.TestData.head())

        for i in range(1,101):
            self.TestData[['lat{}'.format(i),'long{}'.format(i)]] = self.TestData['LATLONG{}'.format(i)].str.split(':', expand = True)

        print(self.TestData.head())
        for i in range(1,101):
            self.TestData= self.TestData.drop(columns = 'LATLONG{}'.format(i))
        print(self.TestData.head())
        return self.TestData,df_test_3

class fitting_models():
    def __init__(self,Test,model):
        self.Test = Test
        self.model = model
    def any_model(self):
        for i in range(1,100):
            # Initial lat long time
            self.Test['s_lat'] = self.Test['lat{}'.format(i)]
            self.Test['s_long'] = self.Test['long{}'.format(i)]
            self.Test['e_lat'] = self.Test['lat{}'.format(i+1)]
            self.Test['e_long'] = self.Test['long{}'.format(i+1)]
    
            x_test = pd.DataFrame(self.Test[['s_lat','s_long','e_lat','e_long','s_clock']],columns = ['s_lat','s_long','e_lat','e_long','s_clock'])

            x_test_2 = min_max_scaler.fit_transform(x_test.drop(columns = 's_clock'))
            x_test_2 = pd.DataFrame(x_test_2,columns = ['s_lat','s_long','e_lat','e_long'])
            if i is 1:
                x_test_2['s_clock'] = x_test['s_clock']
            else:
                x_test_2['s_clock'] = y_pred_real
            y_pred_real = model.predict(x_test_2)
        return y_pred_real




if __name__ == "__main__":
    df = pd.read_csv('final_ver1.2.csv')
    pp = preprocessing_train(df)
    x_train, y_train = pp.fe_training_data()
    #print(x_train.head())
    #print(y_train.head())
    #print()
    model = training_models(x_train,y_train)
    cvs_model,model = model.train_linear_regression()
    #cvs_model,model = model.train_decision_tree_regressor()
    #cvs_model,model = model.train_random_forest_regressor
    #cvs_model,model = model.train_xgb_regressor()
    df_test = pd.read_csv('test.csv')
    pp_test = preprocessing_test(df_test)
    df_preprocessed_test,df_test_3 = pp_test.preprocessing()
    
    fitting = fitting_models(df_preprocessed_test,model)
    y_pred = fitting.any_model()
    df_test_3['Predicted_e_clock'] = y_pred
    df_test_3['TimeDiff'] = df_test_3['Predicted_e_clock'] - df_test_3['s_clock']
    submission = pd.DataFrame({
        "Id": df_test_3["Id"],
        "Duration": df_test_3['TimeDiff']
    })
    submission.to_csv('submission.csv',index = False)
    # Saving LinearRegression Model
    filename = 'lr_pickle.sav'
    pickle.dump(model,open(filename,'wb'))

    
