import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Fetching and Processing Data
# Test and Train Files seperate
# 1258 rows 7 columns (for each variable: open, close)
prices_dataset_train = pd.read_csv('SP500_train.csv')
prices_dataset_test = pd.read_csv('SP500_test.csv')  # 20 rows 7 columns

# selected the adjusted closing price column (index 5) and volume (index 6).
trainingset = prices_dataset_train.iloc[:, 5:6].values  # 1258 rows 1 column
testset = prices_dataset_test.iloc[:, 5:6].values  # 20 rows 1 column


# we use min-max normalization to normalize the dataset
# ML algorithms will give higher accuracy with normalized data!
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
# instead of actual stock proces, transformed to be between 0 and 1258
scaled_trainingset = min_max_scaler.fit_transform(trainingset)

# create training dataset; features are the previous values
# we have n previous values: and we predict the next value in the time series
X_train = []
y_train = []

days = 60

# creating 2D arrays for feature and targets
for i in range(days, 1258):
    # 0 is the column index (last value) because we have a single column
    # using  previous days prices to forecast the next one, subtract days because we have days features
    X_train.append(scaled_trainingset[i-days:i, 0])  # training features
    # indexes start with 0 so this is the target (the price tomorrow)
    y_train.append(scaled_trainingset[i, 0])  # training target variable

# we want to handle numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# input shape for LSTM architecture, reshape the dataset (numOfSamples,numOfFeatures,1)
# we have 1 at the end because we want to predict the price tomorrow (so 1 value)
# numOfFeatures: the past prices we use as features (days)
# X_train.shape[0] = total number of samples (rows), X_train.shape[1] = days features
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# building the LSTM architecture
# return sequence true because we have another LSTM after this one
# keras and tensorflow is backend of implementing LSTM architecture
model = Sequential()
# first layer, return sequence is true because we are using another LSTM right after it, first layer need to define shape: # features , # target variables
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.5))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.3))  # changing Dropout rates
model.add(LSTM(units=50))
model.add(Dropout(0.3))
model.add(Dense(units=1))  # in the end we have a densely connected neural network with a single unit
# Note this will be a regression problem, not classes whether price goes up or down, but getting a price.
# can get different accuracy with more layers t

# using ADAM optimizer, and since regression problem using MSE
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=32)

# testing the algorithm
# training set plus testset
# if we want to make predictions in the test dataset, need to use previous days samples which is in the training set, so need to concat them
# vertical axis=0 horizontal axis=1
dataset_total = pd.concat(
    (prices_dataset_train['adj_close'], prices_dataset_test['adj_close']), axis=0)
# all inputs for test set
inputs = dataset_total[len(dataset_total)-len(prices_dataset_test)-days:].values
inputs = inputs.reshape(-1, 1)

# neural net trained on the scaled values, have to min-max normalize the inputs
# it is already fitted so we can use transform directly
inputs = min_max_scaler.transform(inputs)

X_test = []  # 1D array for test set

for i in range(days, len(prices_dataset_test)+days):
    X_test.append(inputs[i-days:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# predictions from testing
predictions = model.predict(X_test)

# inverse the predicitons because we applied normalization but we want to compare with the original prices
predictions = min_max_scaler.inverse_transform(predictions)

# plotting the results
plt.plot(testset, color='blue', label='Actual S&P500 Prices')
plt.plot(predictions, color='green', label='LSTM Predictions')
plt.title('S&P500 Predictions with Reccurent Neural Network')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# not able to grasp daily changes, but can understand the trends
# notice how the prediction price goes up as the actual goes up, and down when it goes down
# dataset was from 2012-2016, and there was almost constant volatility
