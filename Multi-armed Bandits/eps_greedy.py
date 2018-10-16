import numpy as np
import random
from bandit_env import BanditEnv

class EpsGreedyActor():
    def __init__(self, env, eps, initial_estimate=0., alpha=None):
        self.env = env
        self.k = env.k
        self.eps = eps
        self.alpha = alpha
        self.means = np.full(self.k, float(initial_estimate))
        self.counts = np.zeros(self.k)
        
    def run(self, steps=1000):
        optimal_action = np.argmax([mu for (mu,std) in self.env.r_dists])
        
        reward_hist = np.zeros(steps)
        took_correct_action = np.zeros(steps)   #binary array
        for step in range(steps):
            if random.random() < self.eps:
                action = random.choice(range(self.k))   #choose random action
            else:
                action = np.argmax(self.means)   #choose the greedy action
            r = self.env.sample(action)      #sample reward
            
            reward_hist[step] = r
            self.counts[action] += 1

            # use a constant step size if we have it. otherwise, do incremental
                # average calculation
            if self.alpha is None:
                self.means[action] += (r-self.means[action])/self.counts[action]
            else:
                self.means[action] += self.alpha*(r-self.means[action])
                
            took_correct_action[step] = int(action==optimal_action)
        return reward_hist, took_correct_action
