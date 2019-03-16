import numpy as np
import random
import matplotlib.pyplot as plt
from bandit_env import BanditEnv
from actors import eps

runs = 2000
steps = 1000
num_actions = 10
fig = plt.figure()
rng = random.Random(1234)

for _eps in [0.1, 0.01, 0]:
    reward_hist = np.zeros(steps)
    percent_correct_action = np.zeros(steps)
    
    for run in range(runs):
        env = BanditEnv(rng, num_actions)
        actor = eps.EpsGreedyActor(env, _eps)
        rewards, correct_actions = actor.run(steps)
        reward_hist += (rewards - reward_hist)/(run+1)
        percent_correct_action += correct_actions

    ax1 = plt.subplot(2,1,1)
    ax1.plot(range(steps), reward_hist, label="eps="+str(_eps))
    #ax1.set_ylim([0, 2])
    ax1.set_ylabel("Average reward")
    #ax1.set_xlabel("Steps")

    ax2 = plt.subplot(2,1,2)
    ax2.plot(range(steps), percent_correct_action/runs*100, label="eps="+str(_eps))
    ax2.set_ylim([0, 100])
    ax2.set_ylabel("% Optimal action")
    ax2.set_xlabel("Steps")

plt.show()
