# to recreate Figure 9.10

import numpy as np
import matplotlib.pyplot as plt
import pickle
from random_walk_env import RandomWalkEnv
from mc_state_aggregation import grad_mc
from coarse_coding import get_features, get_receptive_fields
from basis_fns import get_ve

runs = 3
episodes = 5000
alpha = 0.0001
env = RandomWalkEnv()
width = 200
offset = 4
n = (1000-width)//offset+1  # number of features
receptive_fields = get_receptive_fields(width, n, offset)

def plot_performance():
    avg_ve = np.zeros(episodes) # avg performance over all runs
    for run in range(runs):
        w_gen = grad_mc(episodes, alpha, env, width, offset=offset)
        ve = np.zeros(episodes) # performance in this run
        for episode in range(episodes):
            w = next(w_gen)
            v = [np.dot(w, get_features(i, receptive_fields)) 
                    for i in range(1000)]
            ve[episode] = get_ve(v) 
        avg_ve += (ve - avg_ve)/(run+1) # online averaging
    
    # save and plot
    y = np.sqrt(avg_ve)
    plt.plot(y, label="order "+str(n))
    plt.show()

plot_performance()
