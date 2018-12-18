# for reproducing Fig. 8.4 and Fig. 8.5

import numpy as np
import matplotlib.pyplot as plt
from maze_env import MazeEnv
from dyna_q import dyna_q

runs = 1
gamma = 0.95
alpha = 0.1
eps = 0.1
kappa = 0.1
n = 5
steps = 3000
steps_to_switch = 1000

def plot_fig():
    for plus in [False]:#, True]:
        plt.plot(range(steps), plot_reward(plus), label="dyna-q+" if plus else "dyna-q")
    plt.xlabel("Time steps")
    plt.ylabel("Cumulative reward")
    plt.legend()
    plt.show()

def plot_reward(plus):
    reward_per_step = np.zeros(steps)
    env = MazeEnv(2)
    step_gen = dyna_q(n, env, plus)
    for step in range(steps):
        # switch to new environment
        if step == steps_to_switch:
            env = MazeEnv(3)
            step_gen = dyna_q(n, env, plus, Q, model, action_hist)
        
        _,_,reward,_,_,Q,model,action_hist = next(step_gen)
        reward_per_step[step] = reward
    return np.cumsum(reward_per_step)

plot_fig()
