import numpy as np
import random
from bandit_env import BanditEnv

class Actor():
    def __init__(self, env, initial_estimate=0., alpha=None):
        self.env = env
        self.num_actions = env.num_actions
        self.alpha = alpha
        self.means = np.full(self.num_actions, float(initial_estimate))
        self.counts = np.zeros(self.num_actions)
        
    def run(self, steps=1000):
        optimal_action = np.argmax([mu for (mu,std) in self.env.r_dists])
        
        reward_hist = np.zeros(steps)
        took_correct_action = np.zeros(steps)   #binary array
        for step in range(steps):
            action = self.take_action()     #take action
            r = self.env.sample(action)      #sample reward
            self.update_stats(r, action)    #update class properties
            
            reward_hist[step] = r
            took_correct_action[step] = int(action==optimal_action)
        return reward_hist, took_correct_action

    def take_action(self):
        raise NotImplementedError()

    def update_stats(self, r, action):
        self.counts[action] += 1

        # use a constant step size if we have it. otherwise, do incremental
            # average calculation
        if self.alpha is None:
            self.means[action] += (r-self.means[action])/self.counts[action]
        else:
            self.means[action] += self.alpha*(r-self.means[action])
        
