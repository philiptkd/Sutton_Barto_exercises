class BanditEnv():
    def __init__(self, rng, num_actions=10, mean_mean=0, stationary=True):
        self.num_actions = num_actions
        # create (mu,std) for k stationary reward distributions
        self.r_dists = [[rng.gauss(mean_mean, 1), 1] for _ in range(num_actions)]
        self.rng = rng
        self.stationary = stationary

    # given an action i, samples the ith reward distribution
    def sample(self, i):
        if not self.stationary:
            self.r_dists[i][0] += self.rng.gauss(0, 0.01)    #change q* a little
        return self.rng.gauss(*self.r_dists[i])
