from random_walk_env import RandomWalkEnv
import numpy as np
import matplotlib.pyplot as plt

runs = 100
episodes = 10

# output value function over all 19 states for each episode
def n_td(env, n, alpha, trajectories):
    pass

# calculate RMS error between 19 predicted and true values
# then average over number of episodes
def rms(env, values):
    pass

# get runs*episodes trajectories to use for all parameter settings
def get_trajectories(env):
    pass

def plot_rms():
    env = RandomWalkEnv()
    trajectories = get_trajectories(env)
    ns = [2**i for i in range(10)]
    for n in ns:
        plot_n_curve(env,n,trajectories)
    plt.legend()
    plt.xlabel("alpha")
    plt.ylabel("RMS error")
    plt.show()

def plot_n_curve(env,n,trajectories):
    alphas = [0.1*i for i in range(11)]
    for i,alpha in enumerate(alphas):
        avg_rms = np.zeros(len(alphas))
        for run in range(runs):
            values = n_td(env, n, alpha, trajectories[run])
            avg_rms[i] += (rms(env,values) - avg_rms[i])/(run+1)
    plt.plot(alphas, avg_rms, label="n={0}".format(n)

