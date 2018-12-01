'''
Created on Nov 30, 2018

@author: Parsa
'''
import random

from numpy import sort

import numpy as np
from src.DecisionTree import DecisionTree
from concurrent.futures.thread import ThreadPoolExecutor


class RandomForest:
    def __init__(self,numOfTrees):
        self.numOfTrees = numOfTrees
    
    def fit(self,data):
        self.permutation(data)
        self.forest = []
        for i in range(self.numOfTrees):
            self.forest.append(DecisionTree())
        with ThreadPoolExecutor(max_workers=4) as executor:
            for i in range(self.numOfTrees):
                executor.submit(self.forest[i].fit(self.forestData[i]))
        
    def predict(self,row):
        sum = 0
        tot = 0
        with ThreadPoolExecutor(max_workers=4) as executor:
            for i in self.forest:
                sum += i.predict(row)
                tot +=1
        return sum/tot
        
    def permutation(self,data):
        self.forestData = []
        r = random.sample(range(1, data.shape[1]-3), self.numOfTrees-1)
        self.breakingPoints = [0]
        for i in r:
            self.breakingPoints.append(i)
        if data.shape[1]-1 not in self.breakingPoints:
            self.breakingPoints.append(data.shape[1]-2)
        self.breakingPoints = sort(self.breakingPoints)
        self.ranges = []
        
        for i in range(len(self.breakingPoints)-1):
            self.ranges.append([self.breakingPoints[i], self.breakingPoints[i+1]])
        
        for i in self.ranges:
            tree = data[:,i[0]:i[1]]
            tree = np.concatenate((tree, data[:,-1]),1)
            self.forestData.append(tree)
        
            
            
        