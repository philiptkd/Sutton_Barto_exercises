# for Example 6.2

import numpy as np
import matplotlib.pyplot as plt

runs = 100
episodes = 100
gamma = 1

# simple random walk environment
class WalkMRP():
    def __init__(self):
        self.states = ['Zero', 'A', 'B', 'C', 'D', 'E', 'One']
        self.state = 3 # 'C'
        self.np_random = np.random.RandomState()
        self.true_values = np.array([0, 1/6, 2/6, 3/6, 4/6, 5/6, 0])

    # returns (reward, done)
    def step(self):
        step = self.np_random.choice([-1, 1])
        self.state += step
        if self.states[self.state] == 'One':
            self.state = 3 # reset
            return 1, True
        if self.states[self.state] == 'Zero':
            self.state = 3 # reset
            return 0, True
        return 0, False

# plots value estimates for left figure
def td(alpha):
    env = WalkMRP()
    V = np.zeros(len(env.states))
    V[1:-1] = 0.5
    epi_gen = td_episodes(V, env, alpha)
    
    for episode,v in enumerate(epi_gen):
        if (episode+1) in [1,10,100]:
            plt.plot(range(5), v, label=str(episode+1))
    
    plt.legend()
    plt.show()

# generator that yields the value function at each step of the episode
# td(0) prediction
def td_episodes(V, env, alpha):
    for episode in range(episodes):
        done = False
        state = env.state
        while not done:
            reward, done = env.step()
            next_state = env.state
            if done:
                td_error = reward + 0 - V[state] # terminal states have value zero
            else:
                td_error = reward + gamma*V[next_state] - V[state]

            V[state] += alpha*td_error
            state = next_state
        yield V[1:-1]

# every visit monte-carlo prediction
def mc_episodes(V, env, alpha):
    for episode in range(episodes):
        trajectory = get_trajectory(env)
        G = 0
        for i in range(len(trajectory)-1, -1, -1): # for each step in trajectory, from last to first
            state, reward = trajectory[i]
            G = reward + gamma*G
            V[state] += alpha*(G - V[state])
        yield V[1:-1]

def get_trajectory(env):
    trajectory = []
    done = False
    while not done:
        state = env.state
        reward, done = env.step()
        trajectory.append((state,reward))
    return trajectory

# plots rms error of value function per episode, averaged over all runs
def plot_error(alpha, gen_fn, name, update_fn=None):
    rms_error = np.zeros((runs,episodes))
    for run in range(runs):
        env = WalkMRP()
        V = np.zeros(len(env.states))
        V[1:-1] = 0.5

        if update_fn is not None:
            epi_gen = gen_fn(V, env, alpha, update_fn)
        else:
            epi_gen = gen_fn(V, env, alpha)

        for episode,v in enumerate(epi_gen):
            rms_error[run, episode] = np.sqrt(np.average((v - env.true_values[1:-1])**2))
    plt.plot(range(episodes), np.average(rms_error, axis=0), label=name+str(alpha))

def go():
    for alpha in [.05, .1, .15]:
        plot_error(alpha, td_episodes, "td ")

    for alpha in [.01, .02, .03, .04]:
        plot_error(alpha, mc_episodes, "mc ")

    plt.legend()
    plt.show()
