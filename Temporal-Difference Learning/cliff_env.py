# for Example 6.6
# gives slightly different results for Q-learning and sarsa
# I attribute this to the environments being slightly different
    # and averaging over fewer runs
# the paths found are the same, though. and the qualitative result that sarsa is "safer" is also demonstrated

import numpy as np

class CliffEnv():
    def __init__(self):
        self.height = 4
        self.width = 12
        self.start = (3, 0)
        self.goal = (3, 11)
        self.cliff = tuple(zip((3,)*10, range(1,11))) # (3,1) to (3,10)
        self.actions = ("left", "right", "up", "down")
        self.np_random = np.random.RandomState()
        self.reset()
    
    # resets to start position
    def reset(self):
        self.state = list(self.start)

    # returns (reward, done)
    def step(self, action):
        row, col = self.state
        
        # transition to next state
        if action == "left":
            self.state = [row, max(0, col-1)]
        elif action == "right":
            self.state = [row, min(self.width-1, col+1)]
        elif action == "up":
            self.state = [max(0, row-1), col]
        else: # if action == down
            self.state = [min(row+1, self.height-1), col]

        # see if done or fell off cliff
        done = False
        reward = -1
        if tuple(self.state) == self.goal:
            done = True
            self.reset()
        elif tuple(self.state) in self.cliff:
            done = True
            reward = -100
            self.reset()

        return (reward, done)
