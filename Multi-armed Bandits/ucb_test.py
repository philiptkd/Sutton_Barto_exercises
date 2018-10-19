import numpy as np
import matplotlib.pyplot as plt
from bandit_env import BanditEnv
from ucb_actor import UcbActor
from eps_greedy_actor import EpsGreedyActor
import random


runs = 2000
steps = 1000
num_actions = 10
exploration_rate = 1.0
eps = 0.1
alpha = 0.15
rng = random.Random(1234)

reward_hist_ucb = np.zeros(steps)
reward_hist_eps = np.zeros(steps)

for run in range(runs):
    env = BanditEnv(rng, num_actions)
    ucb_actor = UcbActor(env, exploration_rate, alpha=alpha)
    rewards, correct_actions = ucb_actor.run(steps)
    reward_hist_ucb += (rewards - reward_hist_ucb)/(run+1)

    eps_actor = EpsGreedyActor(env, eps, alpha=alpha)
    rewards, correct_actions = eps_actor.run(steps)
    reward_hist_eps += (rewards - reward_hist_eps)/(run+1)

plt.plot(range(steps), reward_hist_ucb, label="UCB, c="+str(exploration_rate))
plt.plot(range(steps), reward_hist_eps, label="eps greedy, eps="+str(eps))
plt.ylim(0, plt.ylim()[1])
plt.ylabel("Average reward")
plt.xlabel("Steps")

plt.legend()
plt.show()
