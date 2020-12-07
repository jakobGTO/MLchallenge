import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

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

df = df.drop(['x4', 'x5'])

#Map false/true string to 0/1 and gradevariabel 0-6
df['x5'] = df['x5'].astype('category')
df['x5'] = df['x5'].cat.codes

df['x6'] = df['x6'].astype('category')
df['x6'] = df['x6'].cat.codes

eval_df['x5'] = eval_df['x5'].astype('category')
eval_df['x5'] = eval_df['x5'].cat.codes

eval_df['x6'] = eval_df['x6'].astype('category')
eval_df['x6'] = eval_df['x6'].cat.codes

#Create in and output dfs
X = df.iloc[:,1:11]
y = df.iloc[:,0]

#Realized x1 and x2 were strings
X['x1'] = X['x1'].astype(float)
X['x2'] = X['x2'].astype(float)

eval_df['x1'] = eval_df['x1'].astype(float)
eval_df['x2'] = eval_df['x2'].astype(float)

# Instantiate model
model = GradientBoostingClassifier(learning_rate=0.1,max_depth=3,n_estimators=100,subsample=1)
model.fit(X,y)

# Train the model on training data
score = cross_val_score(model,X,y,cv=5)
np.average(score)

# Predict new data
output_array = model.predict(eval_df)

# Write to txt file
output_list = []
for i in output_array:
    output_list.append(i)

with open('101331.txt', 'w') as f:
    for item in output_list:
        f.write("%s\n" % item)


