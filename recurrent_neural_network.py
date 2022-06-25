# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 22:03:04 2017

@author: god
"""

import numpy as np

def ReLu(income):
    outcome = income.copy()
    outcome[outcome<0] = 0
    return outcome
    
def ReLu_derivative(income, loss):
    return (income>0) * loss

def sigmoid(income):
    return 1/(1+np.exp(-income))

class recurrent_layer:
    def __init__(self, n, m):
        self.Weights_in = np.random.random((n,m))*0.2 - 0.1
        self.Weights_recurrent = np.random.random((m, m))*0.2 - 0.1
        self.Bias_in = np.random.random((m))*0.01
        self.Bias_recurrent = np.random.random((m))*0.01
        self.Income_stack = []
        self.Outcome_stack = []
    
    def forward(self, income):
        self.Income_stack = income
        self.Outcome_stack = []
        timestep = 0
        for i in range(len(income) + 1):
            outcome = np.mat(np.zeros(self.Weights_recurrent.shape[0]))
            if i<len(income):
                outcome += np.mat(income[i]) * np.mat(self.Weights_in) + np.mat(self.Bias_in)
            if timestep > 0:
                outcome += np.mat(self.Outcome_stack[timestep - 1]) * np.mat(self.Weights_recurrent) + np.mat(self.Bias_recurrent)
            outcome = ReLu(outcome.getA1())
            self.Outcome_stack.append(outcome)
            timestep += 1

        return self.Outcome_stack
    
    def backward(self, loss, lr):
        delta_recurrent = np.zeros(loss[0].shape)
        delta_in = []
        deltaW_recurrent = np.zeros(self.Weights_recurrent.shape)
        deltaW_in = np.zeros(self.Weights_in.shape)
        for i in range(len(loss))[::-1]:
            lossm = np.mat(ReLu_derivative(self.Outcome_stack[i], loss[i] + delta_recurrent))

            if i>0:
                delta_recurrent = (lossm * np.mat(self.Weights_recurrent).T).getA1()
                deltaW_recurrent += (lr * np.mat(self.Outcome_stack[i-1]).T * lossm).getA()
                self.Bias_recurrent += lr * lossm.getA1()
            if i<len(self.Income_stack):
                delta_in.append((lossm * np.mat(self.Weights_in).T).getA1())
                deltaW_in += (lr * np.mat(self.Income_stack[i]).T * lossm).getA()
                self.Bias_in += lr * lossm.getA1()
            
        self.Weights_in += deltaW_in
        self.Weights_recurrent += deltaW_recurrent
        
        delta_in.reverse()
        return delta_in

class fc_layer:
    def __init__(self, n, m, dropout=0):
        self.Weights = np.random.random((n,m))*0.2 - 0.1
        self.Income_stack = []
        self.Outcome_stack = []
        self.Delta = []
        self.Bias = np.random.random((m))*0.01
    
    def forward(self, income):
        self.Income_stack = income
        self.Outcome_stack = [(np.mat(income[i]) * np.mat(self.Weights)).getA1()+self.Bias for i in range(len(income))]
        return self.Outcome_stack
    
    def backward(self, loss, lr):
        self.Delta = []
        deltaW = np.zeros(self.Weights.shape)
        
        for i in range(len(loss)):
            lossm = np.mat(loss[i])

            self.Delta.append((lossm * np.mat(self.Weights).T).getA1())
            deltaW += (lr * np.mat(self.Income_stack[i]).T * lossm).getA()
            self.Bias += lr * loss[i]
            
        self.Weights += deltaW
        return self.Delta
        
class network:
    def __init__(self):
        self.layers = [recurrent_layer(2, 8), fc_layer(8, 1)]
    
    def forward(self, income):
        for i in range(len(self.layers)):
            income = self.layers[i].forward(income)
        return [sigmoid(i) for i in income]
    
    def backward(self, outcome, target, lr):
        loss = [outcome[i]*(1-outcome[i])*(target[i] - outcome[i]) for i in range(len(outcome))]
        for i in range(len(self.layers))[::-1]:
            loss = self.layers[i].backward(loss, lr=lr)
            
        return [target[i] - outcome[i] for i in range(len(outcome))]
        
    def train(self, data, label, lr=0.1):
        return self.backward(self.forward(data), label, lr=lr)
        
        
def getList(numArray, maxlen):
    a = []
    for i in range(maxlen):
        a.append(numArray & 1)
        numArray >>= 1
    return a
        
net = network()
for i in range(1000):
    a = np.random.randint(1024)
    b = np.random.randint(1024)
    c = a + b
    data = getList(np.array([a, b]), 10)
    label = getList(np.array([c]), 11)
    loss = net.train(data, label, 0.1)
#    if i<50 or i>950:
#        sum = 0
#        for j in loss:
#            sum += j * j
#        print(sum)
        

a = np.random.randint(1024)
b = np.random.randint(1024)
c = a + b
data = getList(np.array([a, b]), 10)
label = getList(np.array([c]), 11)
outcome = net.forward(data)
print([i[0] for i in outcome])
print([i[0] for i in label])
print("==========")
#net.backward(outcome, label, 0.1)
sum = 0
for i in range(1000):
    a = np.random.randint(64)
    b = np.random.randint(64)
    c = a + b
    data = getList(np.array([a, b]), 6)
    label = getList(np.array([c]), 7)
    outcome = net.forward(data)
    sum += [int(i[0]>0.5) for i in outcome] == [i[0] for i in label]

print(sum/1000)