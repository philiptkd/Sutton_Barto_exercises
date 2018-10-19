import numpy as np
from bandit_env import BanditEnv
from value_estimate_actor import ValueEstimateActor

class EpsGreedyActor(ValueEstimateActor):
    def __init__(self, env, eps, rng, initial_estimate=0., alpha=None):
        super().__init__(env, initial_estimate=0., alpha=None)
        self.eps = eps
        self.rng = rng

    def take_action(self):
        if self.rng.random() < self.eps:      #sample uniform random variable in [0,1)
            action = self.rng.choice(range(self.num_actions))   #choose random action
        else:
            action = np.argmax(self.means)   #choose the greedy action
        return action
