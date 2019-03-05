# environment as described in Example 9.1

import numpy as np

# states are numbered 1 to 1000
class RandomWalkEnv():
    def __init__(self):
        self.start = 500
        self.num_states = 1000
        self.possible_steps = list(range(-100,101))
        del self.possible_steps[100]    # removes 0
        self.np_random = np.random.RandomState()
        self.reset()

    def reset(self):
        self.state = self.start

    # returns next_state, reward, done
    def step(self):
        step = self.np_random.choice(self.possible_steps)
        self.state += step
        if self.state <= 0:
            self.reset()
            return self.state, -1, True
        if self.state > self.num_states:
            self.reset()
            return self.state, 1, True
        return self.state, 0, False

