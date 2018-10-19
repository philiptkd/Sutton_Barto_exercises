import numpy as np
from bandit_env import BanditEnv
from value_estimate_actor import ValueEstimateActor

class EpsGreedyActor(ValueEstimateActor):
    def __init__(self, env, eps, initial_estimate=0., alpha=None):
        super().__init__(env, initial_estimate=initial_estimate, alpha=alpha)
        self.eps = eps

    def take_action(self):
        if self.rng.random() < self.eps:      #sample uniform random variable in [0,1)
            action = self.rng.choice(range(self.num_actions))   #choose random action
        else:
            max_actions = np.argwhere(self.means == np.max(self.means)).flatten()
            action = self.rng.choice(max_actions)   #choose the greedy action
        return action
