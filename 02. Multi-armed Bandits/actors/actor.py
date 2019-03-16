import numpy as np
from bandit_env import BanditEnv

class Actor():
    def __init__(self, env):
        self.env = env
        self.num_actions = env.num_actions
        self.rng = env.rng

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
        raise NotImplementedError()
