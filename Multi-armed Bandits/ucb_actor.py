import numpy as np
from bandit_env import BanditEnv
from actor import Actor

# uses an Upper Confidence Bound for the estimate of each action's value
# U(a) = c*sqrt[log(t) / counts(a)]
    # where t is the number of actions taken so far
    # and counts(a) is the number of times action a has been taken

class UcbActor(Actor):
    def __init__(self, env, c, initial_estimate=0., alpha=None):
        super().__init__(env, initial_estimate=0., alpha=None)
        self.c = c
        self.U = np.full(self.num_actions, np.Inf)  #initially, Nt(a) = 0

    def take_action(self):
        return np.argmax(self.means + self.U)

    def update_stats(self, r, action):
        super().update_stats(r, action)
        total_count = np.sum(self.counts)
        self.U[action] = self.c*np.sqrt(np.log(total_count)/self.counts[action])
