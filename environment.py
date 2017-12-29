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

import cv2
from gym.spaces.box import Box
from gym import wrappers
from meshPlot import MeshPlot







use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor








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

def Plot_Screen(env):
      env.reset()
      plt.figure()
#      print(get_screen())
#      print(get_screen().cpu().squeeze(0))
#      print(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy())
      plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
                 interpolation='none')
      plt.title('Example extracted screen')
      plt.show()      

def Plot_Frame(frame):
      plt.figure()
      plt.imshow(frame)
      plt.title('Example processed frame')
      plt.show() 


# Improvement of the Gym environment with universe




# Taken from https://github.com/openai/universe-starter-agent


def create_atari_env(env_id, video=False):
    env = gym.make(env_id)
    if video:
        env = wrappers.Monitor(env, 'test', force=True)
    env = MyAtariRescale42x42(env)
    env = MyNormalizedEnv(env)
    return env


def _process_frame42(frame):
    print(type(frame))
    #Plot_Frame(frame)
    frame = frame[20:34 + 180, :160]
    #Plot_Frame(frame)
#    print(len(frame),len(frame[0]))
#    print(frame)
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (100, 100))
    #Plot_Frame(frame)
    frame = cv2.resize(frame, (42, 42))
    #Plot_Frame(frame)
    frame = frame.mean(2)
    #Plot_Frame(frame)
    frame = frame.astype(np.float32)
    #Plot_Frame(frame)
    frame *= (1.0 / 255.0)
    #Plot_Frame(frame)
    #frame = np.reshape(frame, [1, 42, 42])
    return frame


class MyAtariRescale42x42(gym.ObservationWrapper):

    def __init__(self, env=None):
        super(MyAtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 42, 42])

    def _observation(self, observation):
    	return _process_frame42(observation)


class MyNormalizedEnv(gym.ObservationWrapper):

    def __init__(self, env=None):
        super(MyNormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def _observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        ret = (observation - unbiased_mean) / (unbiased_std + 1e-8)
        return np.expand_dims(ret, axis=0)
  
      
      
#env.reset()
#plt.figure()
#plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
#           interpolation='none')
#plt.title('Example extracted screen')
#plt.show()
#
#
#observation = env.reset()
#r = 0
#
#for i in range(500):
#      env.render()
#      action = env.action_space.sample()
#      observation, reward, done, info = env.step(action)
#      print("Observation: ",len(observation)," ",len(observation[0])," ",len(observation[0][0]), "\n","Reward: ",reward,"\n","Done: ",done,"\n","Info: ",info,"\n")
#      
#      
#      plt.figure()
#      plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
#           interpolation='none')
#      plt.title('Example extracted screen')
#      plt.show()
#      r += reward
#      print(r)
#      if done:
#            env.close()     
#      
#env.close()
        
Env_name = "Breakout-v0"
env = gym.make(Env_name)

env1 = create_atari_env(Env_name)
Plot_Screen(env1)

for i in range(500):
      action = env1.action_space.sample()
      observation = env1.step(action)[0]
      print("Observation ",type(observation))
      MeshPlot(observation)