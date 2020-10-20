import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier

df = pd.read_csv('TrainOnMe.csv',sep=",",header=0)
eval_df = pd.read_csv('EvaluateOnMe.csv',sep=",",header=0)
eval_df = eval_df.iloc[:,1:11]
df = df.drop(columns='id',axis=1)

#Remove missing values
df = df.replace('?', np.nan)
df = df.dropna()

df['y'] = df['y'].astype('category')
df['y'] = df['y'].cat.codes

'''
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
'''

def calc_smooth_mean(df1, df2, cat_name, target, weight):
    # Compute the global mean
    mean = df[target].mean()

    # Compute the number of values and the mean of each group
    agg = df.groupby(cat_name)[target].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # Compute the "smoothed" means
    smooth = (counts * means + weight * mean) / (counts + weight)

    # Replace each value by the according smoothed mean
    if df2 is None:
        return df1[cat_name].map(smooth)
    else:
        return df1[cat_name].map(smooth),df2[cat_name].map(smooth.to_dict())

WEIGHT = 5
df['x6'] = calc_smooth_mean(df1=df,df2=None,cat_name='x6',target='y',weight=WEIGHT)
df['x5'] = calc_smooth_mean(df1=df,df2=None,cat_name='x5',target='y',weight=WEIGHT)

#Map false/true string to 0/1 and gradevariabel 0-6

df['x5'] = df['x5'].astype('category')
df['x5'] = df['x5'].cat.codes

#df['x6'] = df['x6'].astype('category')
#df['x6'] = df['x6'].cat.codes

#Create in and output dfs
X = df.iloc[:,1:11]
y = df.iloc[:,0]

#Realized x1 and x2 were strings
X['x1'] = X['x1'].astype(float)
X['x2'] = X['x2'].astype(float)

#X_discrete = X.loc[:,['x5','x6']]

#X_cont = X.loc[:,['x1','x2','x3','x4','x7','x8','x9','x10']]

#Normalize continous variables

#X_cont_norm = preprocessing.normalize(X_cont)

#dummies = pd.get_dummies(X_discrete['x6'],prefix='x6')
#X_discrete = pd.concat([X_discrete['x5'],dummies],axis=1)

#X = np.concatenate((X_cont_norm,X_discrete.values),axis=1)

X = preprocessing.normalize(X)
#X=X.values
#X = X.astype(float)
# encode class values as integers 
#Atsuto = 0, Bob = 1, Jörg = 2

#Train and validation split
#X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.3)

clf = AdaBoostClassifier(n_estimators=1050)
scores = cross_val_score(clf,X,y,cv=10)
scores.mean()