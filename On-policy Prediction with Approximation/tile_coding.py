# to recreate Figure 9.10
# this is only even for one tiling, but it doesn't behave well for small values of offset. I must have made a mistake somewhere, but I don't want to spend time on this anymore.

import numpy as np
import matplotlib.pyplot as plt
import pickle
from random_walk_env import RandomWalkEnv
from mc_state_aggregation import grad_mc
from coarse_coding import approx_v, get_all_bounds
from basis_fns import get_ve

runs = 3
episodes = 5000
alpha = 0.0002
env = RandomWalkEnv()
width = 200
offset = 70
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
    plt.plot(y, label="order "+str(n))
    plt.show()

plot_performance()
