# used to recreate Fig. 9.5
# the fourier curves are really noisy. Maybe I picked slightly different basis functions than they did?
#   Or maybe they didn't use all (n+1) basis functions? they didn't give details.
#   Fourier features outperformed polynomial ones, so I'm satisfied

from random_walk_env import RandomWalkEnv
import numpy as np
import matplotlib.pyplot as plt
import pickle

episodes = 5000
runs = 30
alpha = 0.00005
gamma = 1
ns = [5, 10, 20]
env = RandomWalkEnv()
true_values = np.array([x/500-1 for x in range(1,1001)]) # approximate
POLYNOMIAL = 1
FOURIER = 2
basis = FOURIER


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


# gets vector of alphas, one element per feature
#   simplified because the state space is one-dimensional
def get_alpha_vec(n):
    alphas = [alpha/c for c in range(1,n+1)]
    alphas = [alpha] + alphas   # prepend base alpha for c=0
    return np.array(alphas)


def get_features(s, n):
    s /= 1000   # normalize
    if basis==POLYNOMIAL:
        x = [s**c for c in range(n+1)]  # polynomial features up to order n
    else:   # if basis==FOURIER
    # shifts and scales cos to have range [-1,1]
        x = [2*np.cos(np.pi*s*c)-1 for c in range(n+1)]
    return np.array(x)


# gradient Monte Carlo as on page 202
def train(n):
    w = np.zeros(n+1)
    alphas = get_alpha_vec(n)   # get vector of alphas for Fourier basis fns
    for episode in range(episodes):
        trajectory = get_trajectory()
        G = 0
        for t in range(len(trajectory)-2, -1, -1): # t = T-1,...,1,0
            r = trajectory[t+1][1]
            s = trajectory[t][0]
            G = gamma*G + r
            x = get_features(s, n) # (n+1)-dimensional vector of features
            v = np.dot(w,x) # estimate of state value
            if basis==POLYNOMIAL:
                w += alpha*(G-v)*x    # gradient descent
            else:   # if basis==FOURIER
                w += alphas*(G-v)*x     # element-wise multiply alphas to give each feature a different learning rate
        yield w


# returns the mean squared value error for the given value estimates
def get_ve(estimate):
    with open("state_distribution.npy", "rb") as f:
        dist = pickle.load(f)
    diff = true_values - estimate
    return np.dot(dist, diff**2)    # Eqn. (9.1)


# gets, saves, and plots the state distribution for the random walk environment
def get_state_distribution():
    counts = np.zeros(1000) # times each state is visited
    num_steps = 0
    for episode in range(100000):
        if episode%1000==0:
            print(episode)
        done = False
        env.reset() # redundancy
        while not done:
            num_steps += 1
            counts[env.state-1] += 1
            _, _, done = env.step()
    counts /= num_steps

    with open("state_distribution.npy", "wb") as f:
        pickle.dump(counts, f)
    with open("state_distribution.npy", "rb") as f:
        dist = pickle.load(f)
        plt.plot(dist)
        plt.show()


# uses gradient MC with linear function approximation to estimate values of states
# plots sqrt of VE, the mean squared value error, for different sets of features
def plot_performance():
    outputs = np.zeros((len(ns), episodes))
    for i,n in enumerate(ns):
        avg_ve = np.zeros(episodes) # avg performance over all runs
        for run in range(runs):
            w_generator = train(n)  # yields weights w at the end of each episode 
            ve = np.zeros(episodes) # performance in this run
            for episode in range(episodes):
                if episode%1000==0:
                    print("n:"+str(n)+", run:"+str(run)+", episode:"+str(episode))
                w = next(w_generator)
                v = [np.dot(w, get_features(s, n)) for s in range(1,1001)] # current state value estimates
                ve[episode] = get_ve(v) 
            avg_ve += (ve - avg_ve)/(run+1) # online averaging
        
        # save and plot
        y = np.sqrt(avg_ve)
        outputs[i] = y
        plt.plot(y, label="order "+str(n))

    with open("fourier_performance.npy", "wb") as f:
        pickle.dump(outputs, f)
    
    plt.legend()
    plt.show()

plot_performance()
