# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 12:15:25 2016

@author: god
"""

import numpy as np

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
        return loss * (1-loss)
           
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
        return loss*loss/2
    
    def train(self, data, label, lr=0.0003):
        return self.backward(self.forward(data), label, lr=lr)

#n = network()
#
#for k in range(30000):
#    x = np.random.randint(20) - 10 
#    data = np.array([x, x*x])
#    label = np.array([2*x*x + 3*x + 1])
#    print(n.train(data, label))
#print("===========================")
#print(n.forward(np.array([-4, 16])))
#print(2*16 - 3*4 +1)

map_size = (5, 10)


def not_out_range(pos):
    return (pos[0]<map_size[0] and pos[0]>=0 and pos[1]<map_size[1] and pos[1]>=0)

def possible_next_move(pos):
    direction = np.array([[0,0], [-1,0], [1,0],[0,-1],[0,1]])
    ans = []
    for i in range(5):
        new_pos = (pos[0]+direction[i][0], pos[1]+direction[i][1])
        if not_out_range(new_pos):
            ans.append(new_pos)
    return ans

def get_next_move_according_to_policy(food_moves, agent_moves, Q):
    Qnext = []
    temp = 0
    for ap in range(len(agent_moves)):
        Qnext.append(0)
        for fp in food_moves:
            Qnext[ap] += Q[agent_moves[ap][0], agent_moves[ap][1], fp[0], fp[1]]
                #Qnext[ap] += Q.forward(np.array([agent_moves[ap][0], agent_moves[ap][1], fp[0], fp[1]]))[0]
        Qnext[ap] /= len(food_moves)
        if Qnext[ap] > temp:
            temp = Qnext[ap]
    epsilon = 0.2
    maxQ = max(Qnext)
    
    choice = []
    for i in range(len(Qnext)):
        if Qnext[i] == maxQ:
            choice.append(i)

    p = np.random.random(1)[0]
    if p>epsilon:
        agent_pos = agent_moves[choice[np.random.randint(len(choice))]]
    else:
        agent_pos = agent_moves[np.random.randint(len(agent_moves))]
    
    food_pos = food_moves[np.random.randint(len(food_moves))]
    return agent_pos, food_pos

alpha = 0.9
lr = 0.3       
gama = 0.9
#Q = network([fc_layer(4, 50), ReLU_layer(), fc_layer(50, 50), ReLU_layer(), fc_layer(50, 1)])
Q = np.zeros(map_size+map_size)

for episode in range(3000):
    food_pos = (np.random.randint(map_size[0]), np.random.randint(map_size[1]))
    agent_pos = (np.random.randint(map_size[0]), np.random.randint(map_size[1]))
    while (food_pos==agent_pos):
        agent_pos = (np.random.randint(map_size[0]), np.random.randint(map_size[1]))
    dist = np.abs(agent_pos[0]-food_pos[0]) + np.abs(agent_pos[1]-food_pos[1])
    #move|sample
    step = 0
    #trajectory = []
    while (food_pos!=agent_pos):
        #train
        food_moves = possible_next_move(food_pos)
        agent_moves = possible_next_move(agent_pos)
        #print(loss)
        #trajectory.append([agent_pos[0], agent_pos[1], food_pos[0], food_pos[1], temp])
        #policy
        next_agent_pos, next_food_pos = get_next_move_according_to_policy(food_moves, agent_moves, Q)
        step += 1
        
        Qnext = Q[next_agent_pos + next_food_pos]  
        
        if gama * Qnext > Q[agent_pos + food_pos]:
            Q[agent_pos + food_pos] = alpha * Q[agent_pos + food_pos] + (1-alpha) * gama * Qnext       
        
        agent_pos = next_agent_pos
        food_pos = next_food_pos
        
     
    Q[food_pos + agent_pos] = 1
#    loss = 0
#    for i in range(step)[::-1]:
#        #Q[trajectory[i][0], trajectory[i][1], trajectory[i][2], trajectory[i][3]] = gama * trajectory[i][4]
#        data = np.array([trajectory[i][0], trajectory[i][1], trajectory[i][2], trajectory[i][3]])
#        label = np.array([gama * trajectory[i][4]])
#        loss = Q.train(data, label, lr = lr)
#    data = np.array([agent_pos[0], agent_pos[1], food_pos[0], food_pos[1]])
#    label = np.array([1])
#    print(loss)
#    loss = Q.train(data, label, lr = lr)
#    print(loss)
    #Q[agent_pos[0], agent_pos[1], food_pos[0], food_pos[1]] = 1
    
    print(step/dist)
    print(episode)
    print("==================")