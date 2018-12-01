'''
Created on Nov 30, 2018

@author: Parsa
'''
import numpy as np

class CreateTrainTest:
    def __init__(self, data, splitPercentage):
        self.data = data
        self.data = np.array(self.data)
        self.s = splitPercentage
    
    def train_test(self):
        ones = np.argwhere(self.data[:,-1]==1)
        self.trainSet = []
        self.testSet = []
        for i in ones[:,0]:
            r = np.random.rand()
            if r <= 0.8:
                self.trainSet.append(self.data[i,:])
            else:
                self.testSet.append(self.data[i,:])
        zeros = np.argwhere(self.data[:,-1]==0)
        for i in zeros[:,0]:
            r = np.random.rand()
            if r <= 0.8:
                self.trainSet.append(self.data[i,:])
            else:
                self.testSet.append(self.data[i,:])
        return self.trainSet, self.testSet