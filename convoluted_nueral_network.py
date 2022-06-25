# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 23:04:07 2016

@author: wolf
"""
import numpy as np
import matplotlib.image as mpimg
from scipy import misc
import matplotlib.pyplot as plt
import time

def conv(x, kernel, padding = 0):
    X = np.zeros((x.shape[0]+2*padding, x.shape[1]+2*padding))
    X[padding:x.shape[0]+padding, padding:x.shape[1]+padding] = x

    #todo matrix
    m = np.zeros((X.shape[0]-kernel.shape[0]+1, X.shape[1]-kernel.shape[1]+1))
    
    for i in range(X.shape[0]-kernel.shape[0]+1):
        for j in range(X.shape[1]-kernel.shape[1]+1):
            sub = X[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            m[i, j] = np.sum(sub * kernel)
    
    return m
    
#    y = kernel.copy()
#    y.shape=(kernel.size, 1)
#    m = []
#    for i in range(X.shape[0]-kernel.shape[0]+1):
#        for j in range(X.shape[1]-kernel.shape[1]+1):
#            sub = X[i:i+kernel.shape[0], j:j+kernel.shape[1]].copy()
#            sub.shape = (1, kernel.size)
#            m.append(sub.tolist())
#
#    m = np.mat(np.array(m))
#    y = np.mat(y)
#    ans = (m * y).getA()
#    ans.shape = (int(np.sqrt(ans.shape[0])), int(np.sqrt(ans.shape[0])))
#    return ans

class fc_layer: #todo dynamic
    def __init__(self, n, m, dropout=0):
        self.Weights = np.random.random((n,m))*0.01
        self.Income = np.zeros(n) 
        self.Delta = np.zeros(m)
        self.Bias = np.random.random((m))*0.01
        self.dropout = np.ones(m)
        self.dropout_p = dropout
    
    def forward(self, income):
        self.Income = income
        outcome = np.mat(self.Income) * np.mat(self.Weights)
        
        for i in range(len(self.dropout)):
            self.dropout[i] = (np.random.random()>=self.dropout_p)
        
        return (outcome.getA1() + self.Bias) * self.dropout
    
    def backward(self, loss, lr=0.3):
        lossm = np.mat(loss)
        
        weights = np.mat(self.Weights / np.sqrt(np.sum(self.Weights**2, axis = 0)))
        
        self.Delta = (lossm * weights.T).getA1()
        
        deltaW = lr * np.mat(self.Income).T * lossm
        self.Bias += lr * loss
        self.Weights += deltaW.getA()
        return self.Delta

class cv_layer: #todo accelerate
    def __init__(self, core_num=2, core_size=3, padding = 1):
        self.Income = np.array([])
        self.Cores = np.array(np.random.random((core_size, core_size, core_num))*0.01)
        self.Delta = np.array([])
        self.core_size = core_size
        self.core_num = core_num
        self.padding = padding
        self.Bias = np.random.random((core_num))*0.01
    
    def forward(self, income):
        self.Income = income

        shape = income.shape
        outcome = np.zeros((shape[0],shape[1], self.core_num))
        
        for i in range(self.core_num):
            outcome[:,:,i] += self.Bias[i]
            for j in range(shape[2]):
                outcome[:,:,i] += conv(income[:,:,j], self.Cores[:,:,i], padding = self.padding)
        return outcome
    
    def backward(self, loss, lr = 0.3):
        shape = self.Income.shape

        deltaSum = np.array(np.zeros(shape))
        deltaW = np.zeros((self.core_size, self.core_size, self.core_num))
        
        for i in range(self.core_num):
            deltaSum[:,:,0] += conv(loss[:,:,i], np.rot90(np.rot90(self.Cores[:,:,i])), padding=self.core_size-self.padding-1)
            self.Bias[i] += np.sum(lr*loss[:,:,i])
            for j in range(shape[2]):
                deltaW[:,:,i] += conv(self.Income[:,:,j], loss[:,:,i], padding = self.padding)
        
        for j in range(shape[2]):
            deltaSum[:,:,j] = deltaSum[:,:,0]
                
        self.Delta = deltaSum
        self.Cores += deltaW
        return self.Delta

class ReLU_layer: 
    def __init__(self):
        self.Income = np.array([])
    
    def forward(self, income):
        self.Income = income
        outcome = income.copy()
        outcome[outcome<0] = 0
        return outcome
    
    def backward(self, loss, lr):
        return (self.Income>0) * loss
 
class softmax_layer:
    def forward(self, income):
        exp = np.exp(income - np.max(income))
        sume = np.sum(exp)
        return exp/sume
    
    def backward(self, loss, lr):
        return loss
           
class maxPooling_layer:
    def __init__(self, size = 2):
        self.Income = np.array([])
        self.poolingSize = size
    
    def forward(self, income):
        self.Income = income
        shape = income.shape
        outcome = np.zeros((int(shape[0]/self.poolingSize), int(shape[1]/self.poolingSize), shape[2]))
        for i in range(shape[2]):
            x=0
            while (x<shape[0]):
                y=0
                while(y<shape[1]):
                    sub = income[x:x+self.poolingSize, y:y+self.poolingSize, i]
                    outcome[int(x/self.poolingSize), int(y/self.poolingSize), i] = np.max(sub)
                    x+=self.poolingSize
                    y+=self.poolingSize
        
        return outcome
    
    def backward(self, loss, lr):
        income = self.Income
        shape = income.shape
        outcome = np.zeros(shape)
        for i in range(shape[2]):
            x=0
            while (x<shape[0]):
                y=0
                while(y<shape[1]):
                    tx = 0
                    ty = 0
                    for ix in range(self.poolingSize):
                        for iy in range(self.poolingSize):
                            if income[x+ix,x+iy, i]>income[x+tx, x+ty, i]:
                                tx = ix
                                ty = iy
                    outcome[x+tx, x+ty, i] = loss[int(x/self.poolingSize), int(y/self.poolingSize), i]
                    x+=self.poolingSize
                    y+=self.poolingSize
        
        return outcome
                    
class cv_to_fc_layer: 
    def __init__(self):
        self.shape = (0,0,0)
    
    def forward(self, income):
        self.shape = income.shape
        outcome = income.copy()
        outcome.shape = (1, income.size)
        return outcome
    
    def backward(self, loss, lr):
        delta = loss.copy()
        delta.shape = self.shape
        return delta

                               
class network:
    def __init__(self, layers):
        self.layers = layers
        return
    
    def forward(self, income):
        for i in range(len(self.layers)):
            income = self.layers[i].forward(income)
        return income
        
    def backward(self, outcome, target, lr):
        loss = target - outcome
        for i in range(len(self.layers))[::-1]:
            loss = self.layers[i].backward(loss, lr=lr)
        return outcome - target
    
    def train(self, data, label, lr=0.001):
        return self.backward(self.forward(data), label, lr=lr)

layers = [cv_layer(core_num=5, core_size=5, padding=2), ReLU_layer(), cv_to_fc_layer(), fc_layer(8*8*5, 50), ReLU_layer(), fc_layer(50, 2), softmax_layer()]
#layers = [cv_to_fc_layer(), fc_layer(8*8*1, 50),  ReLU_layer(),  fc_layer(50, 2), softmax_layer()]
n = network(layers)

for i in range(1000):
    data = np.zeros((8,8,1))
    data[np.random.randint(8),:,0] = 1
    data2 = np.zeros((8,8,1))
    data2[:,np.random.randint(8),0] = 1
    x = n.train(data, np.array([1, 0]))
    y = n.train(data2, np.array([0,1]))
    #print(x, y)

print("=====")
data = np.zeros((8,8,1))
data[np.random.randint(8),:,0] = 1
data2 = np.zeros((8,8,1))
data2[:,np.random.randint(8),0] = 1  
print(n.forward(data))
print(n.forward(data2))

#size = 8
#layers = [cv_layer(core_num=8, core_size=5, padding=2), ReLU_layer(), cv_to_fc_layer(), fc_layer(size*size*8, 8), softmax_layer()]


#plt.imshow(p)
#plt.axis('off')
#plt.show()

#t = time.time()
#size=32
#cn = 6
#layers = [cv_layer(core_num=cn, core_size=5, padding=2), ReLU_layer(), maxPooling_layer(size=2), cv_layer(core_num=cn, core_size=5, padding=2), ReLU_layer(), maxPooling_layer(size=2), cv_to_fc_layer(), fc_layer(size/4*size/4*cn, 100), ReLU_layer(), fc_layer(100, 2), softmax_layer()]
#n = network(layers)
#x = 0
#y=0
#for i in range(10):
#    print(i)
#    data1=misc.imresize(mpimg.imread("cnndata1/"+str(i+1)+".JPG"), (size, size)) #time comsuming
#    data2=misc.imresize(mpimg.imread("cnndata2/"+str(i+1)+".png"), (size, size))
#    
#    
#    data1 = np.dot(data1[..., :3], [0.299, 0.587, 0.114])    
#    data2 = np.dot(data2[..., :3], [0.299, 0.587, 0.114]) 
#
#    data1.shape=(data1.shape[0], data1.shape[1], 1)
#    data2.shape=(data2.shape[0], data2.shape[1], 1)
#    
#    loss1 = n.train(data1, np.array([1, 0]), lr=0.3)
#    loss2 = n.train(data2, np.array([0, 1]), lr=0.3)
#    print(loss1, loss2)
#    
#    x = data1
#    y = data2
#
#    print(n.forward(x)-n.forward(y))
#
#print(n.forward(x))
#print(n.forward(y))
#
#print(time.time()-t)
#
#x.shape = (x.shape[0],x.shape[1])
#y.shape = (y.shape[0], y.shape[1])
#plt.imshow(x, cmap='Greys_r')
#plt.axis('off')
#plt.show()
#
#plt.imshow(y, cmap='Greys_r')
#plt.axis('off')
#plt.show()

    