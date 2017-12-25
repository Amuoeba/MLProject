# -*- coding: utf-8 -*-
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T






use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor






env = gym.make("Breakout-v0")





resize = T.Compose([T.ToPILImage(),
                    T.Resize(100,Image.BICUBIC),
                    T.ToTensor()])


def get_screen():
    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))  # transpose into torch order (CHW)    
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).type(Tensor)

env.reset()
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
plt.title('Example extracted screen')
plt.show()











observation = env.reset()
r = 0

for i in range(500):
      env.render()
      action = env.action_space.sample()
      observation, reward, done, info = env.step(action)
      print("Observation: ",len(observation)," ",len(observation[0])," ",len(observation[0][0]), "\n","Reward: ",reward,"\n","Done: ",done,"\n","Info: ",info,"\n")
      
      
      plt.figure()
      plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
      plt.title('Example extracted screen')
      plt.show()
      r += reward
      print(r)
      if done:
            env.close()
      
      
      
env.close()