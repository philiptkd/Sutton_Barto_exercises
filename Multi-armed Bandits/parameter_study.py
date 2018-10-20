import numpy as np
import random
import matplotlib.pyplot as plt
from bandit_env import BanditEnv
from actors import eps, ucb, grad
import pickle

runs = 200
steps = 1000
num_actions = 10
rng = random.Random(1234)

actor_list = [ eps.EpsGreedyActor,
               ucb.UcbActor,
               grad.GradientActor,
             ]

def getPlotPoint(actor_type, param_list):
    reward_hist = np.zeros(steps)
    for run in range(runs):
        env = BanditEnv(rng, num_actions, 0)
        actor = actor_list[actor_type](env, *param_list)
        rewards, _ = actor.run(steps)
        reward_hist += (rewards - reward_hist)/(run+1)
    return(np.mean(reward_hist))
    
ax1 = plt.subplot(1,1,1)
param_values = [2**i for i in range(-7,5)]
eps_curve, opt_curve, ucb_curve, grad_curve = [], [], [], []
for param in param_values:
    print(param)
    eps_curve.append(getPlotPoint(0, [param, 0., 0.1]))
    opt_curve.append(getPlotPoint(0, [1/16, param, 0.1]))
    ucb_curve.append(getPlotPoint(1, [param, 0., 0.15]))
    grad_curve.append(getPlotPoint(2, [param, True]))

#save just in case
f = open('param_curves', 'wb')
pickle.dump(eps_curve+opt_curve+ucb_curve+grad_curve, f)

ax1.plot(param_values, ucb_curve, label="UCB")
ax1.plot(param_values, opt_curve, label="eps greedy + opt. init.")
ax1.plot(param_values, grad_curve, label="gradient")
ax1.plot(param_values, eps_curve, label="eps greedy")

ax1.set_ylim([max(0,ax1.get_ylim()[0]), ax1.get_ylim()[1]])
ax1.set_ylabel("Average reward over first 1000 steps")
ax1.set_xlabel("Parameter value (eps, alpha, c, init)")
ax1.set_xscale('log')
ax1.set_xticks(param_values)
ax1.set_xticks([],True)
ax1.set_xticklabels(['1/128','1/64','1/32','1/16','1/8','1/4','1/2','1','2','4','8','16'])


plt.legend()
plt.show()
