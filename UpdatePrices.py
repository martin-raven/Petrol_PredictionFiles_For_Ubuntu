import requests
import numpy as np
import pandas as pd
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
import json

# The JSON send to the api
return_json = {}

# URLs
url = "https://n6hfls3gbf.execute-api.ap-south-1.amazonaws.com/dev/updateprices" 
delhi_price_csv_path = "DelhiPrice.csv"
model_json_path = "model.json"
model_h5_path = "model.h5"

# The dataset
df = pd.read_csv(delhi_price_csv_path)

sc=MinMaxScaler()

# Gets today's petrol price from the file and add it to json
def get_todays_price():	
	return_json['PriceToday'] = str(df['Weighted_Price'][0]) # First element of csv file is the latest
	print(return_json)

# Get the last 7 days' petrol price and add it to json
def get_last_seven_days_price():
	price_of_days = []
	for i in range(7):
		price_of_days.append(str(df['Weighted_Price'][i]))
	price_of_days.reverse() # Prices in the order - Day 7 to Day 1(Today)
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
def predict_petrol_price():
	df=pd.read_csv(delhi_price_csv_path)
	df['date']=pd.to_datetime(df['Timestamp'], unit='s').dt.date
	group=df.groupby('date')
	Real_Price=group['Weighted_Price'].mean()
	prediction_days=30
	df_train=Real_Price[:len(Real_Price) - prediction_days]
	df_test=Real_Price[len(Real_Price) - prediction_days:]
	training_set=df_train.values
	training_set=np.reshape(training_set, (len(training_set), 1))
	training_set=sc.fit_transform(training_set)
	# load json and create model
	json_file=open(model_json_path, 'r')
	loaded_model_json=json_file.read()
	json_file.close()
	loaded_model=model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights(model_h5_path)
	print("Loaded model from disk")
	# Fitting the RNN to the Training set
	regressor=loaded_model
	# Making the predictions
	test_set=df_test.values
	print("test Values", test_set)
	inputs=np.reshape(test_set, (len(test_set), 1))
	inputs=sc.transform(inputs)
	print(sc.inverse_transform(inputs), type(inputs))
	inputs=np.reshape(inputs, (len(inputs), 1, 1))
	predicted_Petrol_price=regressor.predict(inputs)
	predicted_Petrol_price=sc.inverse_transform(predicted_Petrol_price)
	print("Predicted Price\n", predicted_Petrol_price)
	predicted_Petrol_price=predicted_Petrol_price.tolist()
	print(predicted_Petrol_price[-1][0])
	return predicted_Petrol_price[-1][0]

def predict_petrol_price_week():
	week=[]
	df=pd.read_csv(delhi_price_csv_path)
	df['date']=pd.to_datetime(df['Timestamp'], unit='s').dt.date
	group=df.groupby('date')
	Real_Price=group['Weighted_Price'].mean()
	prediction_days=30
	df_train=Real_Price[:len(Real_Price) - prediction_days]
	df_test=Real_Price[len(Real_Price) - prediction_days:]
	training_set=df_train.values
	training_set=np.reshape(training_set, (len(training_set), 1))
	training_set=sc.fit_transform(training_set)
	# load json and create model
	json_file=open(model_json_path, 'r')
	loaded_model_json=json_file.read()
	json_file.close()
	loaded_model=model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights(model_h5_path)
	print("Loaded model from disk")
	# Fitting the RNN to the Training set
	regressor=loaded_model
	# Making the predictions
	test_set=df_test.values
	print("test Values", test_set)
	inputs=np.reshape(test_set, (len(test_set), 1))
	inputs=sc.transform(inputs)
	print(sc.inverse_transform(inputs), type(inputs))
	inputs=np.reshape(inputs, (len(inputs), 1, 1))
	predicted_Petrol_price=regressor.predict(inputs)
	predicted_Petrol_price=sc.inverse_transform(predicted_Petrol_price)
	print("Predicted Price\n", predicted_Petrol_price)
	price=predicted_Petrol_price[prediction_days-1]
	# print("DEBUG:\n",price)
	for i in range(7):
		# print("DEBUG:",price[0],price,"\n")
		# print("DEBUG:\n",price)
		try:
			week.append(str(price[0][0]))
			price=predict_point_by_point(regressor,price)
		except:
			week.append(str(price[0]))
			price=predict_point_by_point(regressor,price)
	return week

def predict_point_by_point(model, data):
	# Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
	inputs = np.reshape(data, (len(data), 1))
	inputs = sc.transform(inputs)
	inputs = np.reshape(inputs, (len(inputs), 1, 1))
	predicted = model.predict(inputs)
	predicted = sc.inverse_transform(predicted)
	print("After Prediction: ",predicted)
	return predicted



get_todays_price()
get_last_seven_days_price()
get_tomorrows_price()
get_next_seven_days_price()

return_json = json.dumps(return_json)
r = requests.post(url, json=return_json)
print(r)

