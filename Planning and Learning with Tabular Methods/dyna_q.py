# for reproducing Fig. 8.2

import numpy as np
import matplotlib.pyplot as plt
from maze_env import MazeEnv

runs = 30
episodes = 50
gamma = 0.95
alpha = 0.1
eps = 0.1

def plot_fig():
    env = MazeEnv()
    for n in [0, 5, 50]:
        env.np_random = np.random.RandomState(1234) # set common seed
        plot_steps(n, env)
    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.show()

def plot_steps(n, env):
    steps = np.zeros(episodes) # steps per episode
    for run in range(runs):
        print(run)
        steps += (dyna_q(n,env) - steps)/(run+1)
    plt.plot(range(episodes), steps, label="n={0}".format(n))

def dyna_q(n, env):
    steps = np.zeros(episodes) # steps per episode
    Q = np.zeros((env.height*env.width, len(env.actions)))
    model = {} # record of past experience. assumes deterministic
    action_hist = {s:[] for s in range(env.width*env.height)}
    env.reset()
    episode = 0
    step = 0
    state_idx = get_state_idx(env.state, env)
    while episode < episodes:
        # step and record result
        action = get_eps_action(state_idx, Q, env)
        action_hist[state_idx].append(action)
        reward, done = env.step(action)
        next_state_idx = get_state_idx(env.state, env)
        step += 1
        if done:
            steps[episode] = step
            step = 0
            episode += 1

        # one-step q learning
        target = reward
        if not done:
            max_action = get_eps_action(next_state_idx, Q, env, 0)
            target += gamma*Q[next_state_idx, max_action]
        Q[state_idx, action] += alpha*(target - Q[state_idx, action])

        # save experience to model
        if not done:
            model[(state_idx, action)] = (reward, next_state_idx)
        else: # if done
            model[(state_idx, action)] = (reward, -1)

        # n iterations of one-step q planning
        states, state_probs = get_state_probs(action_hist)
        for i in range(n):
            s = env.np_random.choice(states, p=state_probs)
            a = env.np_random.choice(action_hist[s])
            r, sp = model[(s,a)]
            target = r
            if sp != -1:
                max_action = get_eps_action(sp, Q, env, 0)
                target += gamma*Q[sp, max_action]
            Q[s,a] += alpha*(target - Q[s,a])

        state_idx = next_state_idx
    return steps

# choose a state at random from history
def get_state_probs(action_hist):
    states = list(action_hist.keys())
    probs = np.array([len(action_hist[state]) for state in states])
    probs = probs/np.sum(probs)
    return states, probs

# convert state from (row, col) to int index
def get_state_idx(state, env):
    return state[0]*env.width + state[1]

# make eps-greedy action selection
def get_eps_action(state, Q, env, eps=eps):
    if env.np_random.uniform() < eps:
        action = env.np_random.randint(0, len(Q[state]))
    else:
        action = env.np_random.choice(np.flatnonzero(Q[state] == Q[state].max())) # to select argmax randomly
    return action

plot_fig()
