from sklearn import tree
from matplotlib import pyplot as plt

X = [[0,0,0,0],[0,0,1,1],[0,1,0,1],[0,1,1,0],[1,0,0,0],[1,0,1,1],[1,1,1,0],[1,1,0,1]]
y = [0,1,1,0,1,1,0,0]
clf = tree.DecisionTreeClassifier(max_depth = 2)
clf = clf.fit(X,y)

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf,
                   filled=True,class_names=['0','1'])
fig.savefig('heyo.png')