# to recreate Figure 9.10
# uses fewer runs to save time

import numpy as np
import matplotlib.pyplot as plt
import pickle
from random_walk_env import RandomWalkEnv
from mc_state_aggregation import grad_mc, get_v
from basis_fns import get_ve

runs = 3
episodes = 5000
env = RandomWalkEnv()
width = 200

def plot_performance(tilings):
    alpha = 0.0001/tilings
    offset = width//tilings
    avg_ve = np.zeros(episodes) # avg performance over all runs
    for run in range(runs):
        w_gen = grad_mc(episodes, alpha, env, width, offset=offset, tilings=tilings)
        ve = np.zeros(episodes) # performance in this run
        for episode in range(episodes):
            w = next(w_gen)
            v = [get_v(w,s,width,offset,tilings)[0] 
                    for s in range(1,1001)]
            ve[episode] = get_ve(v) 
        avg_ve += (ve - avg_ve)/(run+1) # online averaging
    
    # save and plot
    y = np.sqrt(avg_ve)
    
    label = "one_tiling" if offset==width \
            else str(width//offset)+"_tilings"

    with open(label+".npy","wb") as f:
        pickle.dump(y, f)
    
    plt.plot(y, label=label)
    plt.legend()
    plt.show()

plot_performance(1)
plot_performance(50)
