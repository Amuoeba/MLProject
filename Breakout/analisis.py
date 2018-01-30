# -*- coding: utf-8 -*-
from envs import create_atari_env
from model import ActorCritic
from torch.autograd import Variable
import time
from collections import deque
import pickle
import matplotlib.pyplot as plt


with open("./DUMPS/model_DUMP1.txt","rb") as f1:
    model = pickle.load(f1)
    
with open("./DUMPS/reward_DUMP.txt","rb") as f2:
    rewards = pickle.load(f2)

with open("./DUMPS/reward_DUMP1.txt","rb") as f2:
    rewards1 = pickle.load(f2)
    
with open("./DUMPS/reward_DUMP2.txt","rb") as f2:
    rewards2 = pickle.load(f2)

rewards = rewards + rewards1
    
def MovingAverage(data,average):
    x_axis = list(range(len(data)))
    y_axis_rewards = [i["ep-len"] for i in data]    
    y_averaged = []
    for i in enumerate(y_axis_rewards[:-average]):
        avg_sum = 0
        for j in range(average):
            avg_sum += y_axis_rewards[j+i[0]]        
        avg = avg_sum/average
        y_averaged.append(avg)   
    plt.plot(x_axis[average:],y_averaged)
    return None

MovingAverage(rewards,200)
MovingAverage(rewards2,1)




