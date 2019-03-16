# to reproduce Fig. 7.2
# alphas here is lower resolution than in text for time

from random_walk_env import RandomWalkEnv
import numpy as np
import matplotlib.pyplot as plt

runs = 100
episodes = 10
gamma = 1

# output value function over all 19 states for each episode
def n_td(env, n, alpha, trajectories):
    values = np.zeros(env.num_states)
    for episode in range(episodes):
        steps = trajectories[episode]
        T = len(steps)
        for tau in range(0, T):
            G = np.sum([(gamma**i)*r for (i,(s,r)) in enumerate(steps[tau+1:min(tau+n+1,T+1)])])
            if tau+n < T:
                next_state = steps[tau+n][0]
                G += (gamma**n)*values[next_state]
            state = steps[tau][0]
            values[state] += alpha*(G - values[state])
    return values

# get runs*episodes trajectories to use for all parameter settings
# returns list of lists of lists of tuples (state, reward)
def get_trajectories(env):
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
                reward, done = env.step()
                next_state = env.state
                trajectories[run][episode].append((next_state, reward))
    return trajectories

# calculate RMS error between 19 predicted and true values
# then average over number of episodes
def rms(env, values):
   rms = np.sqrt(np.average((values - env.true_values)**2))
   return rms

def plot_n_curve(env,n,trajectories):
    alphas = [0.1*i for i in range(11)]
    avg_rms = np.zeros(len(alphas))
    for run in range(runs):
        run_rms = np.zeros(len(alphas))
        for i,alpha in enumerate(alphas):
            values = n_td(env, n, alpha, trajectories[run])
            run_rms[i] = rms(env,values)
        avg_rms += (run_rms - avg_rms)/(run+1)
    plt.plot(alphas, avg_rms, label="n={0}".format(n))

def plot_rms():
    env = RandomWalkEnv()
    trajectories = get_trajectories(env)
    ns = [2**i for i in range(10)]
    for n in ns:
        print(n)
        plot_n_curve(env,n,trajectories)
    plt.legend()
    plt.xlabel("alpha")
    plt.ylabel("RMS error")
    #plt.ylim(0.25,0.55)
    plt.show()

plot_rms()
