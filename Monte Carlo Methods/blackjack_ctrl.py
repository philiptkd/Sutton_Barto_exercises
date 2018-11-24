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
num_actions = env.action_space.n

def obs_to_state(observation):
    x,y,z = observation
    i,j,k = x-12, y-1, int(z)
    return i,j,k


# initialize [avg, count] = [0,0] for every possible state
# each state depends on the sum of your current hand (12-21), the dealer's showing card (ace-10), and if you have a usable ace (0,1)
values = np.zeros((10, 10, 2, num_actions, 2))
policy = np.random.choice([0, 1], (10,10,2))

for episode in range(episodes):
    if episode%10000==0:
        print(episode)

    # initialize new episode
    observation = env.reset() 
    done = False
    visited_states = []
    action = np.random.choice([0,1]) # random first action for exploring start

    # roll out new episode
    while not done:
        # record visit of state/action pair
        if observation[0] >= 12:
            i,j,k = obs_to_state(observation)
            state = (i, j, k, action)
            visited_states.append(state)

        # get output
        observation, reward, done, info = env.step(action)
    
        # get new action
        if observation[0] >= 12 and not done:
            i,j,k = obs_to_state(observation)
            action = policy[i,j,k]
        else:
            action = 1 # hit 

    # update values
    for state in visited_states:
        i,j,k,a = state
        values[i,j,k,a,1] += 1
        values[i,j,k,a,0] += (reward - values[i,j,k,a,0])/values[i,j,k,a,1]

    # update policy
    policy = np.argmax(values[:,:,:,:,0], axis=-1)

# plot values
X = Y = range(10)
X, Y = np.meshgrid(X, Y)
values = np.max(values[:,:,:,:,0], axis=-1)
Z1 = values[:,:,0]
Z2 = values[:,:,1]

fig = plt.figure()
ax = fig.add_subplot(211, projection='3d')
ax.plot_wireframe(X,Y,Z1)
ax.set_title("No usable ace")

ax2 = fig.add_subplot(212, projection='3d')
ax2.plot_wireframe(X,Y,Z2)
ax2.set_title("Usable ace")

plt.show()

