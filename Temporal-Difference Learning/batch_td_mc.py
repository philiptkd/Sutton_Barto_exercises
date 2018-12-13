# for recreating Fig. 6.2

from random_walk import get_trajectory, plot_error
import numpy as np
import matplotlib.pyplot as plt

episodes = 100
gamma = 1
runs = 100

# every visit monte-carlo prediction
# after every episode, treats all episodes seen so far as a batch
# alpha needs to be much smaller
def batch_gen(V, env, alpha, update_fn):
    batches = []
    for episode in range(episodes):
        trajectory = get_trajectory(env)
        batches.append(trajectory)
        increments = np.zeros_like(V)
        
        for traj in batches:
            update_fn(traj, increments, V, alpha/len(batches))

        V += increments
        yield V[1:-1]

# given a trajectory, calculates increment to value function
def mc_update(traj, increments, V, alpha):
    G = 0
    for i in range(len(traj)-1, -1, -1): # for each step in trajectory, from last to first
        state, reward = traj[i]
        G = reward + gamma*G
        increments[state] += alpha*(G - V[state])

# given a trajectory, calculates increment to value function
def td_update(traj, increments, V, alpha):
    for i in range(len(traj)): # can go first to last now
        state, reward = traj[i]
        if i==len(traj)-1:
            increments[state] += alpha*(reward + 0 - V[state])
        else:
            next_state = traj[i+1][0]
            increments[state] += alpha*(reward + gamma*V[next_state] - V[state])

alpha = .1
plot_error(alpha, batch_gen, "td ", td_update)
plot_error(alpha, batch_gen, "mc ", mc_update)
    

plt.legend()
plt.show()
