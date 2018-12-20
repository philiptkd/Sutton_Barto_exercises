import numpy as np

class MazeEnv():
    def __init__(self, maze):
        self.width = 9
        self.height = 6
        self.actions = ["left", "right", "up", "down"]
        self.set_grid(maze)
        self.reset()
        self.np_random = np.random.RandomState()

    def set_grid(self, maze):
        self.grid = np.zeros((self.height, self.width))
        if maze == 1:   # for Fig. 8.2
            self.grid[1:4,2] = 1
            self.grid[0:3,7] = 1
            self.grid[4,5] = 1
            self.start = (2,0)
            self.goal = (0,8)
        else:
            if maze == 2: # for Fig. 8.4
                self.grid[3,:8] = 1
            elif maze == 3: # for Fig. 8.4 and Fig. 8.5
                self.grid[3,1:] = 1
            elif maze == 4: # for Fig. 8.5
                self.grid[3,1:8] = 1
            self.start = (5, 3)
            self.goal = (0, 8)

    def reset(self):
        self.state = list(self.start)

    # returns (reward, done)
    def step(self, action):
        if self.actions[action] == "left":
            next_state = [self.state[0], max(0, self.state[1]-1)]
        elif self.actions[action] == "right":
            next_state = [self.state[0], min(self.width-1, self.state[1]+1)]
        elif self.actions[action] == "up":
            next_state = [max(0, self.state[0]-1), self.state[1]]
        else: # if action is "down"
            next_state = [min(self.height-1, self.state[0]+1), self.state[1]]
        
        if self.grid[next_state[0], next_state[1]] != 1:
            self.state = next_state

        if tuple(self.state) == self.goal:
            self.reset()
            return (1, True)

        return (0, False)
