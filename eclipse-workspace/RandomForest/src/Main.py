'''
Created on Nov 29, 2018

@author: Parsa
'''

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.DecisionTree import DecisionTree
from src.RandomForest import RandomForest
from src.CreateTrainTest import CreateTrainTest

data = pd.read_excel("Churn.xlsx")

data = pd.DataFrame(data)
data = data.loc[:,"Account Length":"Intl Charge"]
X = data.drop("Churn", axis=1)
X = np.mat(X)
y = data["Churn"]
y = np.mat(y)
y = y.T

data = np.concatenate((X,y),axis=1)
# data = data[0:100,:]
print(data)
print(data.ndim)

traintest = CreateTrainTest(X,0.8)
trainset, testset = traintest.train_test()
trainset = np.asarray(trainset)
trainset = pd.DataFrame(trainset)
trainset = np.mat(trainset)

testset = np.asarray(testset)
testset = pd.DataFrame(testset)
testset = np.mat(testset)
print("train set: \n",trainset)
print("test set: \n",testset)



forestsk = RandomForestClassifier(n_estimators=1)
forestsk.fit(X,y)
acc = 0
tot = 0
print("\nnumber of trees: 1\n")
print("Random Forest with sklearn:")
for i in testset:
    x = (1 if forestsk.predict(i)>=0.5 else 0)
    if i.item(-1) == x:
        acc+=1
    tot += 1
acc /= tot
print("accuracy is: ",acc)


print("\nMy random forest:")
tree = DecisionTree()
tree.fit(data)
acc = 0
tot = 0
for i in data[2:20,:]:
    x = (1 if tree.predict(i)>=0.5 else 0)
    if i.item(-1) == x:
        acc+=1
    tot += 1
acc /= tot
print("accuracy is: ",acc)


forestsk = RandomForestClassifier(n_estimators=2)
forestsk.fit(X,y)
# print(forest.predict(data[10,:0-1]))
acc = 0
tot = 0
print("\nnumber of trees: 2\n")
print("Random Forest with sklearn:")
for i in testset:
    x = (1 if forestsk.predict(i)>=0.5 else 0)
    if i.item(-1) == x:
        acc+=1
    tot += 1
acc /= tot
print("accuracy is: ",acc)


print("\nMy random forest:")
forest = RandomForest(2)
forest.fit(trainset)
acc = 0
tot = 0
for i in testset:
    x = (1 if forest.predict(i)>=0.5 else 0)
    if i.item(-1) == x:
        acc+=1
    tot += 1
acc /= tot
print("accuracy is: ",acc)


forestsk = RandomForestClassifier(n_estimators=5)
forestsk.fit(X,y)
# print(forest.predict(data[10,:0-1]))
acc = 0
tot = 0
print("\nnumber of trees: 5\n")
print("Random Forest with sklearn:")
for i in testset:
    x = (1 if forestsk.predict(i)>=0.5 else 0)
    if i.item(-1) == x:
        acc+=1
    tot += 1
acc /= tot
print("accuracy is: ",acc)


print("\nMy random forest:")
forest = RandomForest(5)
forest.fit(trainset)
acc = 0
tot = 0
for i in testset:
    x = (1 if forest.predict(i)>=0.5 else 0)
    if i.item(-1) == x:
        acc+=1
    tot += 1
acc /= tot
print("accuracy is: ",acc)



forestsk = RandomForestClassifier(n_estimators=10)
forestsk.fit(X,y)
# print(forest.predict(data[10,:0-1]))
acc = 0
tot = 0
print("\nnumber of trees: 10\n")
print("Random Forest with sklearn:")
for i in testset:
    x = (1 if forestsk.predict(i)>=0.5 else 0)
    if i.item(-1) == x:
        acc+=1
    tot += 1
acc /= tot
print("accuracy is: ",acc)


print("\nMy random forest:")
forest = RandomForest(10)
forest.fit(trainset)
acc = 0
tot = 0
for i in testset:
    x = (1 if forest.predict(i)>=0.5 else 0)
    if i.item(-1) == x:
        acc+=1
    tot += 1
acc /= tot
print("accuracy is: ",acc)




