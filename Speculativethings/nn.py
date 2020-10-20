import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt

#Read data
df = pd.read_csv('TrainOnMe.csv',sep=",",header=0)
eval_df = pd.read_csv('EvaluateOnMe.csv',sep=",",header=0)
eval_df = eval_df.iloc[:,1:11]
df = df.drop(columns='id',axis=1)

#Remove missing values
df = df.replace('?', np.nan)
df = df.dropna()
df.shape
#Map false/true string to 0/1 and gradevariabel 0-5
df['x5'] = df['x5'].astype('category')
df['x5'] = df['x5'].cat.codes

df['x6'] = df['x6'].astype('category')
df['x6'] = df['x6'].cat.codes

#Create in and output dfs
X = df.iloc[:,1:11]
y = df.iloc[:,0]

#Realized x1 and x2 were strings
X['x1'] = X['x1'].astype(float)
X['x2'] = X['x2'].astype(float)


#Normalize continous variables
X = preprocessing.normalize(X)

# encode class values as integers 
#Atsuto = 0, Bob = 1, Jörg = 2
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
y = np_utils.to_categorical(encoded_y)

#Train and validation split
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.3)

#Create model
model = Sequential()
model.add(Dense(16, input_dim=10, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(X_train,
                   y_train,
                   epochs=100,
                   batch_size=5,
                   validation_data=(X_val, y_val))

results = model.evaluate(X_val,y_val)
results

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



