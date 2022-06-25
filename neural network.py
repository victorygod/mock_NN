# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 12:15:25 2016

@author: god
"""

import numpy as np

##===========================Policies====================
#class ReLU:
#    def func(self, x):
#        return x*np.diag((x>=0).getA1())
#    def derivative(self, x):
#        return np.sign(x)
#
#class softmax:
#    def func(self, o):
#        expo = np.exp(o-np.max(o))
#        sume = np.sum(expo)
#        return np.mat(np.round(expo/sume , 3))
#    def derivative(self, o, t):
#        delta = o-t
#        expo = np.exp(delta-np.max(delta))
#        sume = np.sum(expo)
#        return expo/sume
#
#
#class sigmoid:
#    def func(self, x):
#        return np.mat([1/(1+np.exp(-i)) for i in x.getA1()])
#    def derivative(self, x):
#        return np.mat([i*(1-i) for i in x.getA1()])
#
#class mse:
#    def func(self, x):
#        return x
#    def derivative(self, o, t):
#        return o-t
#
##=====================================================================
#
#class layer:
#    def __init__(self,n,m):
#        self.Weights = np.mat(np.random.random((n,m))*2-1)
#        self.Income = np.mat(np.zeros(n))
#        self.Delta = np.mat(np.zeros(m))
#
#class network:
#    #this initialization function will generate a network with designate size, activate function , classifier and learningRate
#    #the activate function must have two interfaces: 
#    #  1.func() represents the activation function itself,which will be used to calculate the outcome
#    #  2.derivative() represents the derivative of the activation function
#    #the classifier must have on interface:
#    #  derivate() represents the derivate of the classifier, for the classifier itself will not be used in the loss calculation
#    def __init__(self, size, activateFunc=ReLU(), classifier=softmax()):
#        self.activateFunc = activateFunc
#        self.classifier = classifier
#        self.net = []
#        for i in range(len(size)):
#            if i==len(size)-1:
#                self.net.append(layer(size[i], size[i]))
#            else:
#                self.net.append(layer(size[i],size[i+1]))
#    
#    #this is the forward function.
#    #you can use this function directly to get an answer
#    def forward(self, data):
#        self.net[0].Income = data;
#        ln = len(self.net)-1        
#        
#        for i in range(ln-1): 
#            self.net[i+1].Income = self.activateFunc.func(self.net[i].Income * self.net[i].Weights)
#        self.net[ln].Income = self.net[ln-1].Income * self.net[ln-1].Weights
#        
#        return self.classifier.func(self.net[ln].Income)
#        
#    #this is the backpropagation function.
#    #here the targets means labels
#    def backPropagation(self, outcomes, targets, lr=0.3):
#        ln=len(self.net)-1
#        
#        #lossD = self.classifier.derivative(outcomes, targets)
#        lossD = targets - outcomes
#        afD = self.activateFunc.derivative(outcomes)
#        self.net[ln].Delta = lossD*np.diag(afD.getA1()) #[-1*lossD[i]*afD[i] for i in range(len(outcomes))]
#
#        for i in range(ln)[::-1]:
#            afD = self.activateFunc.derivative(self.net[i].Income)
#            
#            deltaSum = self.net[i+1].Delta *self.net[i].Weights.T
#            
#            self.net[i].Delta = deltaSum*np.diag(afD.getA1()) #[deltaSum[j] * afD[j] for j in range(len(afD))]
#            
#            deltaW = lr * self.net[i].Income.T * self.net[i+1].Delta
#            self.net[i].Weights += deltaW
#
#    #this train function will train a single trial of data, which means a forward action and a following backpropagation action
#    #label is a zero-one array
#    #make sure data and label have the same size
#    def train(self, data, label):
#        self.backPropagation(self.forward(data), label)
#
##initialize the networks size, now its size is 8*3*8 which means there are 3 nodes in the hidden layer and each 8 nodes in the input layer and output layer
#n = network([8,6,8], classifier=sigmoid()) 


class ReLU_layer: 
    def __init__(self):
        self.Income = np.array([])
    
    def forward(self, income):
        self.Income = income
        outcome = income.copy()
        outcome[outcome<0] = 0
        return outcome
    
    def backward(self, loss, lr):
        return loss*(self.Income>0)
 
class softmax_layer:
    def forward(self, income):
        exp = np.exp(income - np.max(income))
        sume = np.sum(exp)
        return np.round(exp/sume, 3)
    
    def backward(self, loss, lr):
        return loss

class sigmoid_layer:
    def forward(self, income):
        return np.round(1/(1+np.exp(-income)), 3)
    
    def backward(self, loss, lr):
        return loss
           
class fc_layer: #todo dynamic
    def __init__(self, n, m):
        self.Weights = np.mat(np.random.random((n,m))*0.02-0.01)
        self.Income = np.zeros(n)
        self.Delta = np.zeros(m)
        self.Bias = np.random.random((m))*0.02-0.01
    
    def forward(self, income):
        self.Income = income 
        outcome = np.mat(self.Income) * self.Weights
        return outcome.getA1()+self.Bias
    
    def backward(self, loss, lr):
        lossm = np.mat(loss)
        self.Delta = (lossm * self.Weights.T).getA1()
        deltaW = lr * np.mat(self.Income).T * lossm
        self.Bias += lr * loss
        self.Weights += deltaW
        return self.Delta
        
class network:
    def __init__(self, layers):
        #self.layers = [fc_layer(8, 4), ReLU_layer(), fc_layer(4, 8), sigmoid_layer()]
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
        loss = target - outcome
        return loss
    
    def train(self, data, label, lr=0.00003):
        return self.backward(self.forward(data), label, lr=lr)

n = network([fc_layer(1, 2), ReLU_layer(),fc_layer(2, 1)])

for k in range(5000):
    x = np.random.random(1)[0]
    data = np.array([x])
    
    n.train(data, data*1000)
    
print(n.forward(np.array([0.4])))
print(n.forward(np.array([0.1])))
print(n.forward(np.array([0.8])))
#
#for k in range(3000):
#    data = np.zeros(8)
#    data2 = np.zeros(8)
#    v = np.random.randint(8)
#    data[v] = 1
#    #data[(v+1)%8] = 1
#    
#    data2[v] = 1
#    #data2[(v+1)%8] = 1
#
#    n.train(data, data2)
#
##data = np.mat([0,1,0,0,0,0,0,0])
##c = n.forward(data) 
##print (c)
##n.train(data, data)
#
#r = 0
#num = 8
#ar = 0
#for k in range(8):
#    data = np.zeros(8)
#    data[k] = 1
#    ans = n.forward(data)
#    ans = ans
#    print(ans)
