class RandPolicy():
    def __init__(self, num_actions, rng):
        self.num_actions = num_actions
        self.rng = rng
        self.probs = [1./num_actions]*num_actions
    
    def sample_action(self, state):
        return self.rng.choice(range(self.num_actions))

