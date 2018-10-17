import numpy as np
import random
from bandit_env import BanditEnv
from actor import Actor

class EpsGreedyActor(Actor):
    def __init__(self, env, eps, initial_estimate=0., alpha=None):
        super().__init__(env, initial_estimate=0., alpha=None)
        self.eps = eps

    def take_action(self):
        if random.random() < self.eps:      #sample uniform random variable in [0,1)
            action = random.choice(range(self.num_actions))   #choose random action
        else:
            action = np.argmax(self.means)   #choose the greedy action
        return action
