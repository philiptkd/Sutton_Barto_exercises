# used to recreate Fig. 9.2

from random_walk_env import RandomWalkEnv
import numpy as np
import matplotlib.pyplot as plt
import pickle

episodes = 100000
alphas = np.linspace(.02,.00002,episodes)  # decaying alpha
gamma = 1
w = np.zeros(10)
env = RandomWalkEnv()

# returns the group of a state for aggregation
def get_group(s):
    return (s-1)//100  # 1<=s<=1000, but group indices start at 0


# semi-gradient TD(0) as on page 203
for episode in range(episodes):
    if episode%1000==0:
        print(episode)

    alpha = alphas[episode]
    env.reset() # redundancy
    state = env.state
    done = False
    while not done:
        next_state, reward, done = env.step()
        group_s = get_group(state)
        group_ns = get_group(next_state)
        target = reward
        if not done:
            target += gamma*w[group_ns] # v(s,w) = w[group]
        w[group_s] += alpha*(target - w[group_s])
        state = next_state

# plot approximation
x = range(1,1001)
y = [w[(s-1)//100] for s in x]
plt.plot(x, y, label="aggregation")

# load and plot "true" values
with open("true_values.npy", "rb") as f:
    y2 = pickle.load(f)
plt.plot(x, y2, label="true values")

plt.ylim([-1,1])
plt.show()
