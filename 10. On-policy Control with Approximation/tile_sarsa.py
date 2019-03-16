# to recreate Figure 10.2

import numpy as np
np.set_printoptions(linewidth=120)
import matplotlib.pyplot as plt
import pickle
from mountain_car_env import MountainCarEnv


runs = 1
episodes = 500
env = MountainCarEnv()
tilings = 8
offsets = [1,3] # offsets for each dimension.
gamma = 0.99
eps = 0.1

# scales both position and velocity to [0,8) for easy use with tiling
def rescale(s):
    x = tilings*(s[0]-env.position_bounds[0])/(env.position_bounds[1]-env.position_bounds[0])
    y = tilings*(s[1]-env.velocity_bounds[0])/(env.velocity_bounds[1]-env.velocity_bounds[0])
    return [x,y]

# gets the indices of the tile for a given state and tiling (and any action)
# integer division gives rounding behavior I want for negative numbers
def which_tile(s,t):
    scaled_s = rescale(s)
    width = 1   # tile width in both dimensions. same due to rescaling
    x_idx = int((scaled_s[0]-t*offsets[0])//width)%tilings
    y_idx = int((scaled_s[1]-t*offsets[1])//width)%tilings
    return [x_idx, y_idx]

#w has dims (num_actions, tiling, num_tiles_x, num_tiles_y)
# https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays
def get_q(w, s, a):
    # get features as indices into w
    idxs = np.array([which_tile(s,t) for t in range(tilings)])

    # q = sum of w at feature indices
    q = np.sum(w[a, range(tilings), idxs[:,0], idxs[:,1]])
    return q, idxs


def get_eps_action(s, w):
    # take random action with probability eps
    if env.np_random.random_sample() < eps:
        return env.np_random.choice(env.actions)

    # else take best action
    action_values = np.array([get_q(w, s, a)[0] for a in env.actions])
    max_actions = np.where(np.ravel(action_values) == action_values.max())[0]
    action_idx = env.np_random.choice(max_actions) # to select argmax randomly
    return env.actions[action_idx]


def tile_sarsa(alpha):
    # (action, tiling, x_idx, y_idx)
    w = np.zeros((len(env.actions), tilings, tilings, tilings))
    
    for episode in range(episodes):
        # print episode number
        if episode%1==0:
            print("episode: "+str(episode)+"/"+str(episodes))

        # initialize episode
        done = False
        state = env.state
        action = get_eps_action(state, w)
        steps = 0
      
        # semi-gradient sarsa with asymmetric tile coding function approximation
        while not done:
            steps += 1
            next_state, reward, done = env.step(action)
            target = reward
            if not done:
                next_action = get_eps_action(next_state, w)
                target += gamma*get_q(w, next_state, next_action)[0]
            q, idxs = get_q(w, state, action)
            w[action, range(tilings), idxs[:,0], idxs[:,1]] += alpha*(target - q)
            if not done:
                state = next_state
                action = next_action
        yield w, steps

def plot_performance(alpha):
    avg_steps_per_ep = np.zeros(episodes)  # avg steps per episode
    for run in range(runs):
        w_gen = tile_sarsa(alpha)
        steps_per_ep = np.zeros(episodes) # steps per episode
        for episode in range(episodes):
            w, steps = next(w_gen)  # perform one episode of training
            steps_per_ep[episode] = steps   # record num steps taken
        avg_steps_per_ep += (steps_per_ep - avg_steps_per_ep)/(run+1)
    
    # save and plot
    with open("mcar"+str(alpha)+".npy","wb") as f:
        pickle.dump(avg_steps_per_ep, f)
    label = "alpha="+str(alpha)
    plt.plot(avg_steps_per_ep, label=label)


plot_performance(0.1/tilings)
#plot_performance(0.2/tilings)
#plot_performance(0.5/tilings)
plt.legend()
plt.show()
