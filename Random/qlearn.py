# -*- coding: utf-8 -*-
import gym
import numpy as np
import matplotlib.pyplot as plt
env = gym.make("FrozenLake-v0")

Q=np.zeros([env.observation_space.n,env.action_space.n])

learningRate = 0.8
y = 0.95

numEpisodes = 2000

stepList = []
revardList = []

for i in range(numEpisodes):
    s = env.reset()
    rAll = 0
    d = False
    j= 0
    while j < 99:
        j += 1
        action = np.argmax(Q[s,] + np.random.randn(1,env.action_space.n)*(1.0/(i+1)))
        
        s1,r,d,_ = env.step(action)
        Q[s,action] = Q[s,action] + learningRate*(r + y*np.max(Q[s1,:]) - Q[s,action])
        
        rAll += r
        s = s1
        
       #env.render()
        
        if d == True:
            break
    
    stepList.append(j)
    revardList.append(rAll)
    


print ("Score over time: " +  str(sum(revardList)/numEpisodes))
print ("Final Q-Table Values")
print (Q)

plt.plot(stepList)
plt.plot(revardList)