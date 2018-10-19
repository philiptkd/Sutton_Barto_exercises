class BanditEnv():
    def __init__(self, rng, num_actions=10, mean_mean=0):
        self.num_actions = num_actions
        # create (mu,std) for k stationary reward distributions
        self.r_dists = [(rng.gauss(mean_mean, 1), 1) for _ in range(num_actions)]
        self.rng = rng

    # given an action i, samples the ith reward distribution
    def sample(self, i):
        return self.rng.gauss(*self.r_dists[i])
