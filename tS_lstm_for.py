#-----------import the needed libraries to do the forecasting -------------

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

#---------load the data using pandas

#df=pd.read_csv("C:/ML_workshop/data/2.323.24h_p.ptq",sep=" ",header=None)
df=pd.read_csv("H:/HV/Bruker/byha/ML/New_data/combd24h.ptq",sep=" ",header=None)

#print(df.index.freq)

df.shape

df.head(n=10)

df.tail(n=10)

# drop columns for dates (year, month, day, time)

df = df.drop(df.columns[[0,1,2,3]], axis = 1)

# delete rows with numbers -10000

df=df[df.iloc[:,2] >= 0]

# --------- Split train test - Remember here we can’t shuffle the data, 
# -----------because in time series we must follow the order

test_split = round(len(df)*0.20)

df_for_training = df[:-test_split]

df_for_testing = df[-test_split:]


print(df_for_training.shape)
print(df_for_testing.shape)

# to avoid prediction errors let’s scale the data first using MinMaxScaler or StandardScaler

scaler = MinMaxScaler(feature_range=(0,1))

df_for_training_scaled = scaler.fit_transform(df_for_training)

df_for_testing_scaled=scaler.transform(df_for_testing)

df_for_training_scaled.shape

df_for_testing_scaled.shape

# ----- split the data in X and Y ---- follow the steps below

# we are trying to predict the future value of “flow” column, so “flow” is the target column (2)here

def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i,2])  # 3rd column (2) contains flow 
    return np.array(dataX),np.array(dataY)        

trainX,trainY=createXY(df_for_training_scaled,30) 

# [I will discuss about trainX and trainY for the above code, testX and testY follow the same idea]
# n_past is the number of step we will look in the past to predict the next target value.
# So here I used 30, means we will use past 30 values (which have all the features including the target column) to predict the 31st target value.
# So, in trainX we will have all the feature values and in trainY we will have only target value.
# Let’s breakdown each and every part of the for loop —
# for training, dataset = df_for_training_scaled , n_past=30
# When i=30:
# data_X.append(df_for_training_scaled[i — n_past:i, 0:df_for_training.shape[1]])
# As you can see the range started from n_past means 30 so for the first time the data range will be- [30 – 30 : 30, 0 : 5] means [0:30 ,0 : 5]
# So for the first time in the dataX list the df_for_training_scaled[0:30,0:5] array will go.
# Now , dataY.append(df_for_training_scaled[i,0])
# As you know right now i = 30 , so it will take only the open value (because in prediction we only want open column, so column range is only 0 
# which represents open column) from the 30th row.
# So, for the first time in the dataY list the df_for_training_scaled[30,0] value will get stored.
# So, first 30 rows with 5 columns got stored in dataX and 31st row with only open column got stored in dataY. Then we converted the dataX and dataY 
# list to array, because we need them in array format to train in LSTM.
# Like this,the data will get saved in trainX and trainY till the length of the dataset.

trainX.shape

testX,testY=createXY(df_for_testing_scaled,30)

trainX[0]

print("trainX Shape-- ",trainX.shape)

print("trainY Shape-- ",trainY.shape)

print("testX Shape-- ",testX.shape)

print("testY Shape-- ",testY.shape)

# 4132 is the total number of array available in trainX in every array there is total 30 rows and 5 columns and
# in trainY for each array we have the next target value to train the model.
# Let’s have a look on one of the array containing (30,5) data from trainX
# and trainY value for that trainX array


print("trainX[0]-- \n",trainX[0])

print("\ntrainY[0]-- ",trainY[0])

trainY[0]

trainY.shape

# Now if you check trainX[1] value you will notice that it’s the same data from trainX[0] except the first column, as we are going to see the first 30 to predict the 31st column, automatically after the first prediction it will shift to the 2nd column and take the next 30 value to predict the next target value.
# lets explain this all in a simple format —
# trainX — — →trainY
# [0 : 30,0:5] → [30,0]
# [1:31, 0:5] → [31,0]
# [2:32,0:5] →[32,0]
# like this every data will get saved in trainX and trainY.

# --------Train the model, used girdsearchCV to do some hyperparameter tuning to find the based model

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

def build_model(optimizer):
    grid_model = Sequential()
    grid_model.add(LSTM(50,return_sequences=True,input_shape=(30,3)))
    grid_model.add(LSTM(50))
    grid_model.add(Dropout(0.2))
    grid_model.add(Dense(1))

    grid_model.compile(loss = 'mse',optimizer = optimizer)
    return grid_model

grid_model = KerasRegressor(build_fn=build_model,verbose=1,validation_data=(testX,testY))
parameters = {'batch_size' : [32,20],
              'epochs' : [8,10],
              'optimizer' : ['adam','Adadelta'] }

grid_search  = GridSearchCV(estimator = grid_model,
                            param_grid = parameters,
                            cv = 2)
                            
# If the dataset is very large I suggest you to increase the epochs and units in LSTM model.
# You can see in the first LSTM layer the input shape is (30,5). It’s came from trainX shape.
# (trainX.shape[1],trainX.shape[2]) → (30,5)

# --- fit the model in out trainX and trainY data

grid_search = grid_search.fit(trainX,trainY)

# check the best parameters of our model

grid_search.best_params_

# save the best model in my_model variable

my_model=grid_search.best_estimator_.model

# do the predictions for time series
# test the model with our test data set

prediction=my_model.predict(testX)

print("prediction\n", prediction)
print("\nPrediction Shape-",prediction.shape)

prediction.shape

# Inverse scaling 
# inverse scaling is little bit tricky, let’s see everything by code

#scaler.inverse_transform(prediction)  # fails error

# If you see the error you can understand the problem. 
# While scaling the data we had 5 column for each row, right now we are having only 1 column 
# which is the target column.

#So we have to change shape to use inverse_transform

prediction_copies_array = np.repeat(prediction,3, axis=-1)

# Now we have 5 columns [Same column 5 times] 

prediction_copies_array.shape

prediction_copies_array

#Now can easily use inverse_transform function

pred=scaler.inverse_transform(np.reshape(prediction_copies_array,(len(prediction),3)))[:,0]

# compare this pred values with testY. But our testY is also scaled. So, let’s use inverse transform with the same above codes

original_copies_array = np.repeat(testY,3, axis=-1)

original_copies_array.shape

original=scaler.inverse_transform(np.reshape(original_copies_array,(len(testY),3)))[:,0]

print("Pred Values-- " ,pred)
print("\nOriginal Values-- ",original)


# make a plot to check our pred and original data

import matplotlib.pyplot as plt

plt.plot(original, color = 'blue', label = 'Observed flow')
plt.plot(pred, color = 'red', label = 'Simulated  flow')
plt.title(' 2.323 Flows ')
plt.xlabel('Time')
plt.ylabel(' Flow - m^3/s')
plt.legend()
plt.show()


# ----Till now we trained our model , checked that model with test values. 
#   Now let’s predict some future values.

# Now let’s take the last 30 values from the main df dataset what we loaded at the beginning
# [Why 30? because that’s the number of past values we want, to predict the 31st value]

df_30_days_past=df.iloc[-30:,:]

df_30_days_past.tail()


# in multivariate time series forecasting if we want to predict single column by using different features,
# while doing the prediction we need the feature values(except the target column) to do the upcoming predictions


df_future=pd.read_csv("C:/ML_workshop/data/2.323.24h_t.ptq",sep=" ",header=None)

# drop columns for dates (year, month, day, time)

df_future = df_future.drop(df_future.columns[[0,1,2,3]], axis = 1)

f_flows = df_future.columns[[2]]

df_30_days_future = df_future.drop(df_future.columns[[2]], axis=1)

#df_30_days_future = df_future.loc[0:30]

df_30_days_future.shape

df_30_days_future


# As we can see we have all the columns except “flow” column

# ---  Now we have to do some steps before doing the prediction using our model →

# 1. We have to scale the past and future data. As you can see in our future data we don’t have “flow” column , 
#   so before scaling it , just add a “flow” column in the future data with all “0” values.
# 2. After scaling replace the “flow” column value with “nan” in the future data
# 3. Now attach the 30 days old value with 30 days new value (where last 30 “flow” values are nan)

df_30_days_future["flow"]=0

#df_30_days_future=df_30_days_future[["rain","temp","flow"]]

df_30_days_future.columns=df_30_days_past.columns

old_scaled_array=scaler.transform(df_30_days_past)

new_scaled_array=scaler.transform(df_30_days_future)

new_scaled_df=pd.DataFrame(new_scaled_array)

new_scaled_df.iloc[:,2]=np.nan

full_df=pd.concat([pd.DataFrame(old_scaled_array),new_scaled_df]).reset_index().drop(["index"],axis=1)

full_df.shape

full_df.tail(n=10)

# Now to do the prediction we have to use the for loop again, what we made while spliting the data in 
# trainX and trainY. But this time we have only X, no Y value is there


full_df_scaled_array=full_df.values

full_df_scaled_array.shape

all_data=[]

time_step=30

for i in range(time_step,len(full_df_scaled_array)):
    
    data_x=[]
    
    data_x.append(full_df_scaled_array[i-time_step:i,0:full_df_scaled_array.shape[1]])
   
    data_x=np.array(data_x)
   
    prediction=my_model.predict(data_x)
  
    all_data.append(prediction)
   
    full_df.iloc[i,2]=prediction
 
 # for the first prediction we have previous 30 values. Means, when the for loop run for the first time, 
 # it checked the previous 30 values and predict the 31st “Open” data.

# But when the 2nd for loop will try to run, it will skip first row and try to get next 30 values means [1:31] , 
# here we will start getting error as in the last row for the open 
# column we have “nan”, so we have replace the “nan” with the prediction each time .
 
 
 # inverse transform on our prediction
 
new_array=np.array(all_data)

new_array=new_array.reshape(-1,1)

prediction_copies_array = np.repeat(new_array,3, axis=-1)

y_pred_future_30_days = scaler.inverse_transform(np.reshape(prediction_copies_array,(len(new_array),3)))[:,0]
    
    print(y_pred_future_30_days)
    
  from tensorflow.keras.models import Model

  from tensorflow.keras.models import load_model


my_model.save('C:/ML_workshop/Example/Model_future_value.h5')

print('Model Saved!')

scaler

import pickle

scalerfile = 'scaler_model_future_value.pkl'

pickle.dump(scaler, open(scalerfile, 'wb'))
