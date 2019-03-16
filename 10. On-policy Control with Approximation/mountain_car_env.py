# environment as described in section 10.1

import numpy as np

class MountainCarEnv:
    def __init__(self):
        self.position_bounds = [-1.2, 0.5]
        self.velocity_bounds = [-0.07, 0.07]
        self.start_bounds = [-0.6, -0.4]    # actually [-0.6, -0.4)
        self.actions = [-1, 0, 1]
        self.np_random = np.random.RandomState()
        self.reset()

    def reset(self):
        # randomly sampled from start_bounds interval
        self.position = self.np_random.random_sample() \
                * (self.start_bounds[1]-self.start_bounds[0]) \
                + self.start_bounds[0]
        self.velocity = 0
        self.state = [self.position, self.velocity]

    # returns next_state, reward, done
    def step(self, action):
        assert action in self.actions, "action was "+str(action)
        done = False
        reward = -1

        # dynamics
        self.velocity = self.velocity + 0.001*action \
                - 0.0025*np.cos(3*self.position)
        self.velocity = bound(self.velocity, self.velocity_bounds)
        self.position += self.velocity
        self.position = bound(self.position, self.position_bounds)

        # handle reaching edges
        if self.position == self.position_bounds[0]:
            self.velocity = 0
        elif self.position == self.position_bounds[1]:
            done = True
            self.reset()

        self.state = [self.position, self.velocity]
        return self.state, reward, done


def bound(x, bounds):
    clipped = min(x, bounds[1])
    clipped = max(x, bounds[0])
    return clipped

