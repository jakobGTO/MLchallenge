import numpy as np
import pandas as pd
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
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
df = df.drop(index=453)
df = df.drop(index=539)


for i in range(df.shape[0]):
    if df.iloc[i,6] == 'F':
        df.iloc[i,6] = 0
    if df.iloc[i,6] == 'Fx':
        df.iloc[i,6] = 1
    if df.iloc[i,6] == 'E':
        df.iloc[i,6] = 2
    if df.iloc[i,6] == 'D':
        df.iloc[i,6] = 3
    if df.iloc[i,6] == 'C':
        df.iloc[i,6] = 4
    if df.iloc[i,6] == 'B':
        df.iloc[i,6] = 5
    if df.iloc[i,6] == 'A':
        df.iloc[i,6] = 6


#Map false/true string to 0/1 and gradevariabel 0-6

df['x5'] = df['x5'].astype('category')
df['x5'] = df['x5'].cat.codes

eval_df['x5'] = eval_df['x5'].astype('category')
eval_df['x5'] = eval_df['x5'].cat.codes


#Create in and output dfs
X = df.iloc[:,1:11]
y = df.iloc[:,0]

#Realized x1 and x2 were strings
X['x1'] = X['x1'].astype(float)
X['x2'] = X['x2'].astype(float)

eval_df['x1'] = eval_df['x1'].astype(float)
eval_df['x2'] = eval_df['x2'].astype(float)

#X_discrete = X.loc[:,['x5','x6']]

#X_cont = X.loc[:,['x1','x2','x3','x4','x7','x8','x9','x10']]

#Normalize continous variables

X = preprocessing.normalize(X)
X_eval = preprocessing.normalize(X_eval)

#X = np.concatenate((X_cont_norm,X_discrete.values),axis=1)

#X = X.astype(float)

# encode class values as integers 
#Atsuto = 0, Bob = 1, Jörg = 2

encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
y = np_utils.to_categorical(encoded_y)
y[0:10]
#Train and validation split
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.3)

#Create model
es = EarlyStopping(monitor='val_loss', verbose=1, patience=75)
opt = keras.optimizers.Adam(lr=0.001)

model = Sequential()
model.add(Dense(10, input_dim=10, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

history = model.fit(X_train,
                   y_train,
                   epochs=500,
                   batch_size=24,
                   validation_data=(X_val, y_val),
                   callbacks=es
                   )

#Plot accuracy accross time for both training and test data
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#Predict new data

data_to_predict = model.predict(X_eval)
classes = np.argmax(data_to_predict, axis = 1)

output_array = []
for i in classes:
    if i == 0:
        output_array.append('Atsuto')
    if i == 1:
        output_array.append('Bob')
    if i == 2:
        output_array.append('Jörg')

with open('output_array_nn.txt', 'w') as f:
    for item in output_array:
        f.write("%s\n" % item)



