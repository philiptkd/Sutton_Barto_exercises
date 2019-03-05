# used to recreate Fig. 9.1

from random_walk_env import RandomWalkEnv
import numpy as np
import matplotlib.pyplot as plt
import pickle

episodes = 100000
alpha = 0.00002
gamma = 1
w = np.zeros(10)
env = RandomWalkEnv()

# returns a list of tuples (S_t, R_t)
#   the final tuple's state will be env.start and should not be used
def get_trajectory():
    trajectory = []
    env.reset() # redundancy
    trajectory.append((env.state, 0)) # R_0 = 0
    done = False
    while not done:
        next_state, reward, done = env.step()
        trajectory.append((next_state, reward))
    return trajectory

# gradient Monte Carlo as on page 202
for episode in range(episodes):
    if episode%1000==0:
        print(episode)

    trajectory = get_trajectory()
    G = 0
    for t in range(len(trajectory)-2, -1, -1): # t = T-1,...,1,0
        r = trajectory[t+1][1]
        s = trajectory[t][0]
        group = (s-1)//100  # 1<=s<=1000, but group indices start at 0
        G = gamma*G + r
        w[group] += alpha*(G - w[group])    # v(s,w) = w[group]

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
