import numpy as np
from bandit_env import BanditEnv
from actor import Actor

class ValueEstimateActor(Actor):
    def __init__(self, env, initial_estimate=0., alpha=None):
        super().__init__(env)
        self.alpha = alpha
        self.means = np.full(self.num_actions, float(initial_estimate))
        self.counts = np.zeros(self.num_actions)

    def update_stats(self, r, action):
        super().update_stats(r,action)

        # use a constant step size if we have it. otherwise, do incremental
            # average calculation
        if self.alpha is None:
            self.means[action] += (r-self.means[action])/self.counts[action]
        else:
            self.means[action] += self.alpha*(r-self.means[action])
        
