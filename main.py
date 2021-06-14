import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Load Data
company = 'IBM'

start = dt.datetime(2010, 1, 1)  # time point we want to start the data from
end = dt.datetime(2014, 12, 31)

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
test_start = dt.datetime(2015, 1, 1)  # data model has not seen before
test_end = dt.datetime(2016, 6, 1)

test_data = web.DataReader(company, 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values  # sqaure brackets!

total_dataset = pd.concat((data['Close'], test_data['Close']),
                          axis=0)  # combines training data and test data

# what the model will see as input so it can predict the next price
model_inputs = total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)  # scale it down


# Make Predictions on Test DataReader
x_test = []
for i in range(prediction_days, model_inputs.shape[0]):
    x_test.append(model_inputs[i - prediction_days:i])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)  # note that now the prices will be scaled
predicted_prices = scaler.inverse_transform(
    predicted_prices.reshape(predicted_prices[:, :, 0].shape))

# Plot the Test Predictions
plt.plot(actual_prices, color="black", label=f"Actual {company} Price")
plt.plot(predicted_prices[:, -1], color="red", label=f"Predicted {company} Price")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f"{company} Share Price")
plt.legend()
plt.show()

# Predicting Next Day

real_data = [model_inputs[len(model_inputs)-prediction_days:len(model_inputs+1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1]))

# print(scaler.inverse_transform(real_data[-1]))

# real data is the input and will predict next day we dont know
prediction = model.predict(real_data.reshape(real_data.shape[0], 60, 1))
prediction = scaler.inverse_transform(prediction.reshape(prediction.shape[:2]))
print(f"Prediction: {prediction}")

#why are the lines so close
#generate to try and predict next 4 days
#pick 5 different stocks 
