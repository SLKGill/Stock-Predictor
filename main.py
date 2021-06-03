import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Load Data
company = 'FB'

start = dt.datetime(2012, 1, 1)  # time point we want to start the data from
end = dt.datetime(2020, 1, 1)

data = web.DataReader(company, 'yahoo', start, end)  # getting data from yahoo finance API


# Prepare Data for NN
# scale down all the values to be between 0 and 1, feature range from sklrean preprocessing module
scaler = MinMaxScaler(feature_range=(0, 1))
# predicitng price after markets closed
# transforming data frame of closing frame
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

prediction_days = 60  # how many days we want to base the prediction on (looking in the past)

# Training Data, starting with 2 empty lists
x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):  # 60th index to length of scaled data, filling up the list
    # add values to x_train each iteration, first 60 values will be known, and starting to train it so it will know the next value
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

# Convert to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# print(x_train)

# Build Model

model = Sequential()  # sequential allows you to make models layer by layer in a step-by-step fashion
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))  # dropout layer is to prevent overfitting, 20% of the layers will be dropped
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # prediction of the next closing value

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

# figure out how well the model would perform on past data, accuracy
# this data has to be data we have not seen before
# Test the Model Accuracy on Existing Data

# Load Test Data
test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

test_data = web.DataReader(company, 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)
