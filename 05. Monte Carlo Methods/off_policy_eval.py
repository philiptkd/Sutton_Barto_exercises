# to reproduce Fig. 5.3
# plays blackjack with infinite deck
# evaluates the policy of sticking on 20 or 21
# generates data using random policy
# compares ordinary and weighted importance sampling

import gym
import numpy as np
import matplotlib.pyplot as plt
import gym.envs.toy_text.blackjack as bj

env = gym.make('Blackjack-v0')
episodes = 10000
runs = 100

value_hist_is = np.zeros((runs,episodes)) # history of values of one particular state
value_hist_wis = np.zeros((runs,episodes))
for run in range(runs):
    print(run)
    value_is = 0 # current estimate of value
    value_wis = 0
    wis_sum = 0 # sum of importance sampling ratios over all episodes/trajectories

    for episode in range(episodes):
        # initialize new episode to specifications of figure description
        env.player = [1, 2]
        env.dealer = [2, bj.draw_card(env.np_random)]
        observation = env._get_obs()
        done = False
        
        # initialize episode statistics
        rho = 1 # importance sampling ratio \Pi_T pi(a_t|s_t)/b(a_t|s_t)

        # roll out new episode
        while not done:
            # get current state
            my_sum, showing_card, usable_ace = observation

            # choose action randomly
            action = env.np_random.choice([0,1])

            # update episode statistics
            if action == 0: # stick
                if my_sum == 20 or my_sum == 21:
                    rho *= 1/0.5
                else:
                    rho *= 0/0.5
            else: # hit
                if my_sum == 20 or my_sum == 21:
                    rho *= 0/0.5
                else:
                    rho *= 1/0.5

            # get output
            observation, reward, done, info = env.step(action)
        
        # update values
        value_is += (rho*reward - value_is)/(episode+1) 
        value_hist_is[run,episode] = value_is

        wis_sum += rho
        if wis_sum != 0: # else, value_wis remains 0
            value_wis += (reward - value_wis)*(rho/wis_sum)
        value_hist_wis[run,episode] = value_wis

# plot values
true_value = -0.27726
plt.plot(range(episodes), np.average((value_hist_is-true_value)**2, axis=0), label="Ordinary Importance Sampling")
plt.plot(range(episodes), np.average((value_hist_wis-true_value)**2, axis=0), label="Weighted Importance Sampling")
plt.xscale('log')
plt.xlabel("Episodes (log scale)")
plt.ylabel("Mean square error (averaged over 100 runs)")
plt.ylim([0,5])
plt.legend()
plt.show()

