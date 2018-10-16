import random

class BanditEnv():
    def __init__(self, k=10):
        self.k = k
        random.seed()
        # create (mu,std) for k stationary reward distributions
        self.r_dists = [(random.gauss(0, 1), 1) for _ in range(k)]

    # given an action i, samples the ith reward distribution
    def sample(self, i):
        return random.gauss(*self.r_dists[i])
