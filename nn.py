import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt

#Read data
X = pd.read_csv('vowelX.txt',sep=",",header=None)
y = pd.read_csv('vowelY.txt',sep=",",header=None)

#predict_data = X.iloc[[0,25,50,75,100,125,140],:].values
#predict_labels = y.iloc[[0,25,50,75,100,125,140],:]
#X = X.drop([0,25,50,75,100,125,140])
#y = y.drop([0,25,50,75,100,125,140])

#Remove missing values
X = X.dropna()
y = y.dropna()

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
y = np_utils.to_categorical(encoded_y)

#Make np.array out of pd.df
X = X.values
y = y.values

#Normalize input values
X = preprocessing.normalize(X)

#Train and validation split
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.2)

#Create model
model = Sequential()
model.add(Dense(512, input_dim=10, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(11, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(X_train,
                   y_train,
                   epochs=200,
                   batch_size=12,
                   validation_data=(X_val, y_val))

#Plot accuracy accross time for both training and test data
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#Predict new data

#data_to_predict = model.predict(predict_data)
#classes = np.argmax(data_to_predict, axis = 1)



