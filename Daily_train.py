import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
# Importing the Keras libraries and packages
from keras.models import Sequential,model_from_json

df = pd.read_csv('DelhiPrice.csv')
df['date'] = pd.to_datetime(df['Timestamp'], unit='s').dt.date
group = df.groupby('date')
Real_Price = group['Weighted_Price'].mean()
df_train = Real_Price[len(Real_Price) - 2:]
training_set = df_train.values
training_set = np.reshape(training_set, (len(training_set), 1))

print("Training data",training_set)

sc = pickle.load( open( "MinMaxScaler.dat", "rb" ) )
# print("Training set",training_set)
# sc.fit(training_set)
training_set = sc.transform(training_set)
print(training_set)

X_train = training_set[0]
# print("Train data X",X_train)
y_train = training_set[1]
# print("Train data y",y_train)
X_train = np.reshape(X_train, (len(X_train), 1, 1))
print(X_train,y_train)

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
# Fitting the RNN to the Training set
regressor = loaded_model
# # Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs=12000)
model_json = regressor.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
regressor.save_weights("model.h5")
print("Saved model to disk")
