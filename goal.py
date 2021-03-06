#!/usr/bin/python3
import sys
import torch
import random
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import matplotlib.pyplot as plt
from itertools import count

# Constants
GAMMA = 0.9
eps = np.finfo(np.float32).eps.item()
DEPTH = 15
alpha = 0.05

class Env:

    def __init__(self,n, m, threshold, bounty = 10):
        random.seed()
        self.done = False
        self.bounty = bounty
        self.threshold = threshold
        self.action = -1
        self.row_size = n
        self.col_size = m

        self.move = [[0, 1], [0, -1], [1, 0], [-1, 0]]

        self.grid = np.zeros((n, m))
            
        self.sx = 0
        self.sy = 0
        self.dx = 0
        self.dy = 0
        
    def reset(self):
        self.done = False
        for i in range(self.row_size):
            for j in range(self.col_size):
                self.grid[i][j] = 0
                
        t = random.randint(0, self.row_size * self.col_size - 1)
        self.sx = t // self.col_size;
        self.sy = t % self.col_size;
        t = random.randint(0, self.row_size * self.col_size - 1)
        self.dx = t // self.col_size;
        self.dy = t % self.col_size;

        if self.sx == self.dx and self.sy == self.dy:
            self.reset()

        self.grid[self.sx][self.sy] = 1
        self.grid[self.dx][self.dy] = 2
        return self.grid

    def render(self):
        print("===========================================")
        print('Applied action: {:d}\n'.format(self.action))
        for i in range(0, self.row_size):
            for j in range(0, self.col_size):
                print('{:2d} '.format(int(self.grid[i][j])), end = '')
            print('');
        print("===========================================")
        input()

    def step(self,n):
        self.action = n
        x, y = self.sx, self.sy
        x += self.move[n][0]
        y += self.move[n][1]

        reward = -1

        if x < 0 or y < 0 or x >= self.row_size or y >= self.col_size:
            reward = -2
            return self.grid, reward, False
        
        self.grid[self.sx][self.sy] = 0

        self.grid[x][y] = 1
        
        if x == self.dx and y == self.dy:
            self.done = True

        self.sx, self.sy = x, y
            
        if self.done == True:
            return self.grid, self.bounty, True

        return self.grid, reward, False


class PolicyNetwork(nn.Module):
    def __init__(self, in_features, num_actions, hidden_size, learning_rate=1e-2):
        super(PolicyNetwork, self).__init__()

        self.num_actions = num_actions
        self.linear1 = nn.Linear(in_features, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        # self.softmax = nn.LogSoftmax(dim=1)
        self.rewards = []
        self.log_probs = []
   
    def forward(self, state):
        x = F.relu(self.linear1(state.view(-1)))
        x = self.linear2(x)
        x = F.softmax(x,dim=0)
        return x 


policy = PolicyNetwork(5*5*5*5, 4, 128)
env = Env(5, 5, 10)
optimizer = optim.SGD(policy.parameters(), lr=1e-2)

def select_action(state):
    state = torch.from_numpy(state).float()
    probs = policy(state)
    action_n = np.random.choice(4, p=np.squeeze(probs.detach().numpy()))
    policy.log_probs.append(torch.log(probs.squeeze()[action_n]))
    return action_n

def finish_episode(policy):
    R = 0
    policy_loss = []
    Gt = []
    for r in policy.rewards[::-1]:
        R = r + GAMMA * R
        Gt.insert(0, R)
    Gt = torch.tensor(Gt)
    Gt = (Gt - Gt.mean()) / (Gt.std() + eps)

    for log_prob, R in zip(policy.log_probs, Gt):
        policy_loss.append(-log_prob * R)

    optimizer.zero_grad()
    policy_loss = torch.stack(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    
    del policy.rewards[:]
    del policy.log_probs[:]

    

def main():
                      
    # running_reward = 10
    for i_episode in count(1):
        state, ep_reward = env.reset(), 0
        for t in range(1, DEPTH): 
            action = select_action(state)
            state, reward, done = env.step(action)

            if i_episode >= 100000:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break
        if (i_episode % 100 == 0):
            print('run: {:d} / 100000 ({:d}%)\r'.format(i_episode, int(i_episode / 100000 * 100)), end = '')
            # print('run: ', i_episode)

        # running_reward = alpha * ep_reward + (1 - alpha) * running_reward
        finish_episode(policy)
        # if running_reward > env.threshold:
        #     print("Solved! Running reward is now {} and "
        #           "the last episode runs to {} time steps!".format(running_reward, t))
        #     break

if __name__ == '__main__':
    main()

