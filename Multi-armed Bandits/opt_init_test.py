import numpy as np
import matplotlib.pyplot as plt
from bandit_env import BanditEnv
from eps_greedy import EpsGreedyActor

runs = 2000
steps = 1000
k = 10
alpha = 0.1

env = BanditEnv(k)

fig = plt.figure()

for (eps, init) in [(0.,5.),(0.1,0.)]:
    #reward_hist = np.zeros(steps)
    percent_correct_action = np.zeros(steps)
    
    for run in range(runs):
        actor = EpsGreedyActor(env, eps, init, alpha)
        rewards, correct_actions = actor.run(steps)
        #reward_hist += (rewards - reward_hist)/(run+1)
        percent_correct_action += correct_actions

    plt.plot(range(steps), percent_correct_action/runs*100, label="eps="+str(eps)+" init="+str(init))
    plt.ylim([0, 100])
    plt.ylabel("% Optimal action")
    plt.xlabel("Steps")

plt.legend()
plt.show()
