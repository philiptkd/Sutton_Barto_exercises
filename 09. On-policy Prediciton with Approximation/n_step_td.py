# used to get the "true" values for Fig. 9.1 and Fig. 9.2

from random_walk_env import RandomWalkEnv
import numpy as np
import matplotlib.pyplot as plt
import pickle

runs = 100
episodes = 2000
gamma = 1

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
                next_state, reward, done = env.step()
                next_state -= 1 # RandomWalkEnv indexing starts at 1
                trajectories[run][episode].append((next_state, reward))
    return trajectories

def plot_value_fn(n, alpha):
    env = RandomWalkEnv()
    trajectories = get_trajectories(env)
    avg_values = np.zeros(env.num_states)
    for run in range(runs):
        print(run)
        values =  n_td(env, n, alpha, trajectories[run])
        avg_values += (values - avg_values)/(run+1)
    
    # save and plot
    with open("true_values.npy","wb") as f:
        pickle.dump(avg_values, f)
    plt.plot(range(1,env.num_states+1), avg_values, label="true values")
    plt.show()

plot_value_fn(4,.5)
