# to reproduce Fig. 5.1
# plays blackjack with infinite deck
# evaluates the policy of sticking on 20 or 21
# uses every visit monte carlo evaluation

import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

env = gym.make('Blackjack-v0')
episodes = 500000

# initialize [avg, count] = [0,0] for every possible state
# each state depends on the sum of your current hand (12-21), the dealer's showing card (ace-10), and if you have a usable ace (0,1)
values = [[[[0.,0.] for k in range(2)] for j in range(10)] for i in range(10)]

for episode in range(episodes):
    print(episode)

    # initialize new episode
    observation = env.reset() 
    done = False
    visited_states = []

    # roll out new episode
    while not done:
        # get current state
        my_sum, showing_card, usable_ace = observation
        if my_sum >= 12:
            state = (my_sum-12, showing_card-1, usable_ace)
            visited_states.append(state)

        # choose action
        if my_sum == 20 or my_sum == 21:
            action = 0 # stick
        else:
            action = 1 # hit
        
        # get output
        observation, reward, done, info = env.step(action)
    
    # update values
    for state in visited_states:
        i,j,k = state
        values[i][j][k][1] += 1
        values[i][j][k][0] += (reward - values[i][j][k][0])/values[i][j][k][1]

# plot values
X = Y = range(10)
X, Y = np.meshgrid(X, Y)
Z1 = np.array(values)[:,:,0,0] # only nonusable ace states
Z2 = np.array(values)[:,:,1,0] # only usable ace states

fig = plt.figure()
ax = fig.add_subplot(211, projection='3d')
ax.plot_wireframe(X,Y,Z1)
ax.set_title("No usable ace")

ax2 = fig.add_subplot(212, projection='3d')
ax2.plot_wireframe(X,Y,Z2)
ax2.set_title("Usable ace")

plt.show()

