import requests
import numpy as np
import pandas as pd
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
import json
import pickle

# The JSON send to the api
return_json = {}

# URLs
url = "https://n6hfls3gbf.execute-api.ap-south-1.amazonaws.com/dev/updateprices"
delhi_price_csv_path = "DelhiPrice.csv"
model_json_path = "model.json"
model_h5_path = "model.h5"

# The dataset
df = pd.read_csv(delhi_price_csv_path)

sc = MinMaxScaler()

# Gets today's petrol price from the file and add it to json


def get_todays_price():
    # First element of csv file is the latest
    return_json['PriceToday'] = str(df['Weighted_Price'][0])
    print(return_json)

# Get the last 7 days' petrol price and add it to json


def get_last_seven_days_price():
    price_of_days = []
    for i in range(7):
        price_of_days.append(str(df['Weighted_Price'][i]))
    price_of_days.reverse()  # Prices in the order - Day 7 to Day 1(Today)
    return_json['PriceLastSevenDays'] = price_of_days
    print(return_json)

# Get the predicted price for tomorrow


def get_tomorrows_price():
    return_json['PriceTomorrow'] = str(predict_petrol_price())
    print(return_json)

# Get the next seven days predicted price


def get_next_seven_days_price():
    predicted_price_week = predict_petrol_price_week()
    return_json['PriceNextSevenDays'] = predicted_price_week
    print(return_json)

# Function to predict tomorrow's petrol price


def predict_point_by_point(number):
    # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    # load json and create model
    json_file = open(model_json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(model_h5_path)
    print("Loaded model from disk")
    sc = pickle.load(open("MinMaxScaler.dat", "rb"))
    value = sc.transform([[number]])
    value = np.reshape(value, (len(value), 1, 1))
    predicted = model.predict(value)
    print("Predicted", predicted)
    predicted = sc.inverse_transform(predicted)
    print("transformed", predicted)
    return predicted


def predict_petrol_price():
    df = pd.read_csv('DelhiPrice.csv')
    df['date'] = pd.to_datetime(df['Timestamp'], unit='s').dt.date
    group = df.groupby('date')
    Real_Price = group['Weighted_Price'].mean()
    value = Real_Price[-1]
    predicted_Petrol_price = predict_point_by_point(value)
    print(predicted_Petrol_price[-1][0])
    return predicted_Petrol_price[-1][0]


def predict_petrol_price_week():
    week = []
    df = pd.read_csv('DelhiPrice.csv')
    df['date'] = pd.to_datetime(df['Timestamp'], unit='s').dt.date
    group = df.groupby('date')
    Real_Price = group['Weighted_Price'].mean()
    value = Real_Price[-1]
    for _ in range(6):
        predicted_Petrol_price = predict_point_by_point(value)
        predicted_Petrol_price = predicted_Petrol_price[-1][0]
        print("printing prediction value of day",
              _, ":", predicted_Petrol_price)
        week.append(str("{0:.2f}".format(predicted_Petrol_price)))
        value = predicted_Petrol_price
    return week


get_todays_price()
get_last_seven_days_price()
get_tomorrows_price()
get_next_seven_days_price()

return_json = json.dumps(return_json)
r = requests.post(url, json=return_json)
print(r)
