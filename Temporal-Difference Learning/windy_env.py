# for Example 6.5

import numpy as np

class WindyEnv():
    def __init__(self):
        self.height = 7
        self.width = 10
        self.start = (3, 0)
        self.goal = (3, 7)
        self.wind = (0,0,0,1,1,1,2,2,1,0)
        self.actions = ("left", "right", "up", "down")
        self.np_random = np.random.RandomState()
        self.reset()
    
    # resets to start position
    def reset(self):
        self.state = list(self.start)

    # returns (state, reward, done)
    def step(self, action):
        row, col = self.state
        wind = self.wind[col]
        
        # transition to next state
        if action == "left":
            self.state = [max(0, row-wind), max(0, col-1)]
        elif action == "right":
            self.state = [max(0, row-wind), min(self.width-1, col+1)]
        elif action == "up":
            self.state = [max(0, row-wind-1), col]
        else: # if action == down
            self.state = [min(max(0, row-wind+1), self.height-1), col]

        # save state
        new_state = self.state

        # see if done
        done = False
        if tuple(self.state) == self.goal:
            done = True
            self.reset()

        return (new_state, -1, done)
