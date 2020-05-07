from random import randrange
import sys
import torch
import random
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
# import matplotlib.pyplot as plt
from itertools import count
class Env:
    def __init__(self,n,m):
        self.row_size = n
        self.col_size = m
        self.action = -1
        self.action_space = [[-1,0],[0,-1],[1,0],[1,0]]

        self.tiles = np.array((5,1,5,1,3,
                                2,-1,-2,-5,1,
                                1,-3,-5,4,1,
                                2,4,-1,0,4,
                                3,2,-4,2,-5)).reshape(5,5)
        self.grid = np.zeros((n, m))

        self.sx = 0
        self.sy = 0
    def reset(self):
        self.done = False
        self.grid = np.zeros((self.row_size, self.col_size))
        t = random.randint(0, self.row_size * self.col_size - 1)
        self.sx = t // self.col_size
        self.sy = t % self.col_size

        self.grid[self.sx][self.sy] = 1
        return self.sx * self.row_size + self.sy
    def render(self):
        print("<=========================================>")
        print('Applied action: {:d}\n'.format(self.action))
        print("===================Grid===================")
        for i in range(0, self.row_size):
            for j in range(0, self.col_size):
                print('{:2d} '.format(int(self.grid[i][j])), end = '')
            print('');
        
        print("===================tiles===================")
        for i in range(0, self.row_size):
            for j in range(0, self.col_size):
                print('{:2d} '.format(int(self.tiles[i][j])), end = '')
            print('');
        print("<=========================================>")
        input()
    def step(self,action):
        self.action = action
        x, y = self.sx, self.sy
        x += self.action_space[action][0]
        y += self.action_space[action][1]

        if x < 0 or y < 0 or x >= self.row_size or y >= self.col_size:
            reward = -5
            return self.sx * self.row_size + self.sy, reward
        reward = self.tiles[x,y]
        self.tiles[self.sx][self.sy] = -1
        self.grid[self.sx][self.sy] = 0
        # TODO: daraa ni ywsan zam deeree dahij ywbal reward baihgvi
        # uuruur helbel tiles[self.sx][self.sy]-g 0 bolgono

        self.grid[x][y] = 1

        self.sx, self.sy = x, y

        return self.sx * self.row_size + self.sy, reward
        
def training():
    env = Env(5,5)
    # table that will say which action is great in which state
    q_table = np.zeros((env.row_size*env.col_size,len(env.action_space)))
    # row_n = n,x col_n = m,y
    learning_rate = 0.07
    discount = 0.9
    max_steps = 25
    times_won = 0
    epsilon = 1
    max_epsilon = 1
    min_epsilon = 0.01
    epsilon_decay = 0.001

    for episode in count(1):
        state = env.reset()
        step = 0
        episode_reward = 0
        for step in range(max_steps):
            exp_tradeoff = random.uniform(0,1)

            if(exp_tradeoff > epsilon):
                action = np.argmax(q_table[state,:])
            else:
                action = randrange(len(env.action_space))#0-3 hvrtel random utga awna
            
            new_state,reward = env.step(action)

            episode_reward += reward

            q_table[state,action] = q_table[state,action] + learning_rate * (reward + discount * np.max(q_table[new_state,:]) - q_table[state,action])

            state = new_state

            if episode_reward > 30:
                times_won += 1
                break

            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay*episode) 
        if(times_won == 30):
            break
        print(episode)
    np.savetxt("q_table_temp3.txt",q_table)
def testing():
    env = Env(5,5)
    q_table = np.loadtxt('q_table_temp3.txt', dtype = float)

    total_test_episodes = 100
    max_steps = 25

    rewards = []

    for episode in range(total_test_episodes):
        state = env.reset()
        episode_reward = 0
    
        while True:
            action = np.argmax(q_table[state,:])
            new_state, reward = env.step(action)
            episode_reward += reward
            env.render()
            print("reward:",reward)
            if episode_reward > 30:
                rewards.append(episode_reward)
                print("this episode is done with episode reward: ",episode_reward)
                break
            state = new_state
training()