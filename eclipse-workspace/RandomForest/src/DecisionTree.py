'''
Created on Nov 29, 2018

@author: Parsa
'''
import numpy as np
from math import log2

class DecisionTree:
    
    class Node:
        def __init__(self,result=None, column=None, val=None, leftTree=None, rightTree=None):
            self.result = result
            self.col = column
            self.val = val
            self.leftTree = leftTree
            self.rightTree = rightTree
            self.tags = [0,1]
    
    def predict(self,row):
        row = np.matrix(row)
        self.traverse(self.root, row)
        return self.result
         
    def traverse(self,node,row):
        if node.result == None:
            column = node.col
            value = node.val
            if row.item(column) >= value:
                self.traverse(node.leftTree,row)
            elif row.item(column) < value:
                self.traverse(node.rightTree,row)
        else:
            self.result = node.result
    def fit(self, data):
        data = np.matrix(data)
        self.root = self.createTree(data,1000)
        
    def createTree(self, data, maxDepth):
        bestGain = 0
        splitSets = None
        column = None
        for col in range(data.shape[1]-1):
            for row in range(data.shape[0]):
                s1, s2 = self.split(data, col, data[row,col])
                if len(s1)==0 or len(s2)==0:
                    continue;
                info = self.info(data, col)
                gain = self.gain(data, s1, s2, col, info)
                if gain > bestGain:
                    bestGain = gain
                    value = data[row,col]
                    splitSets = (s1, s2)
                    column = col
        if bestGain>0 or maxDepth<=0:
            maxDepth -= 1
            right = self.createTree(splitSets[1], maxDepth)
            left = self.createTree(splitSets[0], maxDepth)
            return self.Node(column=col, val=value, leftTree=left, rightTree=right)
        else:
            return self.Node(result=self.result(data))
    
    def split(self, data, col, value):
        s1 = []; s2 = []
        for i in data:
            if i.item(col)>= value:
                s1.append(i)
            else:
                s2.append(i)
        s1 = np.asarray(s1).reshape(len(s1),data.shape[1])
        s2 = np.asarray(s2).reshape(len(s2),data.shape[1])
        return s1,s2
    
    def info(self, data, col):
        if len(data[:,0])==0:
            return 0
        p = len(np.argwhere(data[:,-1]==1))
        p = float(float(p)/ len(data[:,0]))
        return self.entropy(p)
            
    def entropy(self, p):
        if p==0:
            return 0
        return float(-1 * float(p) * log2(float(p)))
        
    def gain(self,data, s1, s2, col, info):
        info_s1 = self.info(s1, col)
        info_s2 = self.info(s2, col)
        D = data.shape[0]
        D_s1 = s1.shape[0]
        D_s2 = s2.shape[0]
        return info - ((D_s1/D) * info_s1 + (D_s2/D) * info_s2)
    
    def result(self,data):
        win = len(np.argwhere(data[:,-1]==1))/len(data[:,0])
        return win
    