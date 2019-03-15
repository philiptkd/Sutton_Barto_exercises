# to recreate Figure 9.10
# I made a mistake in that I didn't allow partial fields to exist at the boundaries. This makes the boundaries less accurate. I haven't fixed it yet.

import numpy as np
import matplotlib.pyplot as plt
import pickle
from random_walk_env import RandomWalkEnv
from mc_state_aggregation import grad_mc
from coarse_coding import approx_v, get_all_bounds
from basis_fns import get_ve

runs = 30
episodes = 5000
alpha = 0.0001
env = RandomWalkEnv()
width = 200
offset = 200
n = (1000-width)//offset+1  # number of features

def plot_performance():
    bounds = get_all_bounds(width, offset, range(env.num_states))
    avg_ve = np.zeros(episodes) # avg performance over all runs
    for run in range(runs):
        w_gen = grad_mc(episodes, alpha, env, width, offset=offset)
        ve = np.zeros(episodes) # performance in this run
        for episode in range(episodes):
            w = next(w_gen)
            v = [approx_v(w,i,width,offset,bounds)[2] 
                    for i in range(1000)]
            ve[episode] = get_ve(v) 
        avg_ve += (ve - avg_ve)/(run+1) # online averaging
    
    # save and plot
    y = np.sqrt(avg_ve)
    with open("one_tiling.npy","wb") as f:
        pickle.dump(y, f)
    
    label = "one tiling" if offset==width \
            else str(width//offset)+" tilings"
    plt.plot(y, label=label)
    plt.legend()
    plt.show()

plot_performance()
