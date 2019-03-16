# used to reconstruct figure 5.4
# demonstrates the infinite variance of ordinary importance sampling

import numpy as np
import matplotlib.pyplot as plt

np_random = np.random.RandomState()

# samples from simple environment
# returns (reward, done)
def step(action):
    if action == 'right':
        return 0, True
    else: # if action == 'left'
        if np_random.uniform() < 0.1: # with probability 0.1
            return 1, True
        else: # with probability 0.9
            return 0, False

episodes = 1000000
runs = 10

fig = plt.figure()
ax = fig.add_subplot(111)

value_hist_is = np.zeros((runs,episodes)) # history of values of one particular state

for run in range(runs):
    value_is = 0 # current estimate of value

    for episode in range(episodes):
        if episode%100000==0:
            print(run, episode)
        done = False
        rho = 1

        while not done:
            action = np_random.choice(['left','right']) # choose action randomly
            if action == 'left':
                rho *= 1/(0.5)
            else: # if action == right
                rho *= 0/(0.5)
            reward, done = step(action) # take step 

        value_is += (rho*reward - value_is)/(episode+1) 
        value_hist_is[run,episode] = value_is

    ax.plot(range(episodes), value_hist_is[run,:])

plt.xscale('log')
plt.xlabel('Episodes (log scale)')
plt.show()
