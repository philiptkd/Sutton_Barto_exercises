import numpy as np

class Policy():
    def __init__(self, num_states, num_actions, rng):
        self.rng = rng
        self.probs = np.zeros((num_states, num_actions))

    def get_prob(self, state, action):
        return self.probs[state][action]

    def sample_action(self, state):
        uniform_sample = self.rng.uniform()
        action = -1
        while uniform_sample > 0:
            action += 1
            uniform_sample -= self.probs[state][action]
        return action

class RandPolicy(Policy):
    def __init__(self, num_states, num_actions, rng):
        self.rng = rng
        self.probs = np.full((num_states, num_actions), 1/num_actions)

class GreedyPolicy(Policy):
    def __init__(self, V, num_actions, rng):
        