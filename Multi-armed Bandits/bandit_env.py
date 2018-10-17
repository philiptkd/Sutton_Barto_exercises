import random

class BanditEnv():
    def __init__(self, num_actions=10):
        self.num_actions = num_actions
        random.seed()
        # create (mu,std) for k stationary reward distributions
        self.r_dists = [(random.gauss(0, 1), 1) for _ in range(num_actions)]

    # given an action i, samples the ith reward distribution
    def sample(self, i):
        return random.gauss(*self.r_dists[i])
