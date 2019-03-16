# to reproduce Fig. 9.2
# runs here is lower than in text for time

from random_walk_env import RandomWalkEnv
import numpy as np
import matplotlib.pyplot as plt

runs = 100
episodes = 10
gamma = 1
num_groups = 20
true_values = np.array([x/500-1 for x in range(1,1001)]) # approximate
env = RandomWalkEnv()

# returns the group of a state for aggregation  
# 1<=s<=1000, but group indices start at 0
def get_group(s):
    return (s-1)//(env.num_states//num_groups)

# semi-gradient method with state aggregation
#   where 20 groups of 50 states each are used
def n_td(n, alpha, trajectories):
    w = np.zeros(num_groups)
    for episode in range(episodes):
        steps = trajectories[episode]
        T = len(steps)
        for tau in range(0, T):
            G = np.sum([(gamma**i)*r for (i,(s,r)) in enumerate(steps[tau+1:min(tau+n+1,T+1)])])
            if tau+n < T:
                next_state = steps[tau+n][0]
                group_ns = get_group(next_state)
                G += (gamma**n)*w[group_ns]
            state = steps[tau][0]
            group_s = get_group(state)
            w[group_s] += alpha*(G - w[group_s])
        yield w

# get runs*episodes trajectories to use for all parameter settings
# returns list of lists of lists of tuples (state, reward)
def get_trajectories():
    trajectories = []
    for run in range(runs):
        trajectories.append([])
        for episode in range(episodes):
            trajectories[run].append([])
            env.reset()
            # rewards start at R_1
            trajectories[run][episode].append((env.state, 0))
            done = False
            while not done:
                next_state, reward, done = env.step()
                trajectories[run][episode].append((next_state, reward))
    return trajectories

# calculate RMS error between predicted and true values
# average over number of episodes
def rms(n, alpha, trajectories):
    w_generator = n_td(n, alpha, trajectories)
    avg_rms = 0
    for episode in range(episodes):
        w = next(w_generator)
        values = np.array([w[get_group(s)] for s in range(1,1001)])
        rms = np.sqrt(np.average((values - true_values)**2)) # scalar
        avg_rms += (rms - avg_rms)/(episode+1)  # online averaging
    return avg_rms

def plot_n_curve(n, trajectories):
    alphas = [0.02*i for i in range(51)]    # alpha=0,.02,...,.98,1
    avg_rms = np.zeros(len(alphas))
    for run in range(runs):
        print("n:"+str(n)+", run:"+str(run))
        run_rms = np.zeros(len(alphas))
        for i,alpha in enumerate(alphas):
            run_rms[i] = rms(n, alpha, trajectories[run])
        avg_rms += (run_rms - avg_rms)/(run+1)
    plt.plot(alphas, avg_rms, label="n={0}".format(n))

def plot_rms():
    trajectories = get_trajectories()
    ns = [2**i for i in range(10)]
    for n in ns:
        plot_n_curve(n, trajectories)
    plt.legend()
    plt.xlabel("alpha")
    plt.ylabel("RMS error")
    plt.ylim([0.25,0.6])
    plt.show()

plot_rms()
