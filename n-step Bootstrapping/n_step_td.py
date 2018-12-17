from random_walk_env import RandomWalkEnv
import numpy as np
import matplotlib.pyplot as plt

runs = 100
episodes = 10
gamma = 1

# output value function over all 19 states for each episode
def n_td(env, n, alpha):
    values = np.zeros(env.num_states)
    for episode in range(episodes):
        env.reset()
        T = np.inf
        t = 0
        steps = []
        steps.append((0, env.state)) # rewards start at R_1
        while True:
            # if we haven't hit a terminal state yet
            if t < T:   
                reward, done = env.step() # take random action
                if done:
                    T = t + 1
                next_state = env.state
                steps.append((reward, next_state))

            # update state we visited n-1 steps ago
            tau = t - (n - 1) 
            if tau >= 0:
                G = np.sum([(gamma**i)*r for (i,(r,s)) in enumerate(steps[tau+1:min(tau+n+1,T+1)])])
                if tau+n < T:
                    G += (gamma**n)*values[next_state]
                state = steps[tau][1]
                values[state] += alpha*(G - values[state])
            
            if tau == T-1:
                break
            t += 1
    return values

# calculate RMS error between 19 predicted and true values
# then average over number of episodes
def rms(env, values):
   rms = np.sqrt(np.average((values - env.true_values)**2))
   return rms

def plot_n_curve(env,n):
    alphas = [0.1*i for i in range(11)]
    avg_rms = np.zeros(len(alphas))
    for run in range(runs):
        print(run)
        run_rms = np.zeros(len(alphas))
        for i,alpha in enumerate(alphas):
            values = n_td(env, n, alpha)
            run_rms[i] = rms(env,values)
        avg_rms += (run_rms - avg_rms)/(run+1)
    plt.plot(alphas, avg_rms, label="n={0}".format(n))

def plot_rms():
    env = RandomWalkEnv()
    ns = [4]#[2**i for i in range(10)]
    for n in ns:
        plot_n_curve(env,n)
    plt.legend()
    plt.xlabel("alpha")
    plt.ylabel("RMS error")
    #plt.ylim(0.25,0.55)
    plt.show()

plot_rms()
