import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Read data
X = pd.read_csv('irisX.txt',sep=",",header=None)
y = pd.read_csv('irisY.txt',sep=",",header=None)

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

#Train and validation split
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.3)

#Create model
model = Sequential()
model.add(Dense(64, input_dim=4, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(X_train,
                   y_train,
                   epochs=100,
                   batch_size=5,
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



