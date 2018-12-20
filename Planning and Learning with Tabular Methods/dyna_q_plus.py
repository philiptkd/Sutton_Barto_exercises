# for reproducing Fig. 8.4 and Fig. 8.5

import numpy as np
import matplotlib.pyplot as plt
from maze_env import MazeEnv
from dyna_q import dyna_q

kappa = 0.001
n = 5
steps = 10000

def plot_fig():
    for plus in [False,True]:
        plt.plot(range(steps), plot_reward(plus), label="dyna-q+" if plus else "dyna-q")
    plt.xlabel("Time steps")
    plt.ylabel("Cumulative reward")
    plt.legend()
    plt.show()

def plot_reward(plus):
    reward_per_step = np.zeros(steps)
    env = MazeEnv(3)
    step_gen = dyna_q(n, env, plus, kappa=kappa, env_switch_to=(4,1000))
    for step in range(steps):
        _,_,reward,_,_ = next(step_gen)
        reward_per_step[step] = reward
    return np.cumsum(reward_per_step)

plot_fig()
