import numpy as np

np_random = np.random.RandomState()
class turn_env():
    def generate_turn(self, size):
        assert size > 5

        grid = np.zeros((size,size)) # turn is subset of grid

        # set start line as bottom row
        start_line_len = 4
        left = size//5
        right = left+start_line_len
        grid[size-1-0, left:right+1] = 1

        # fill in the remaining rows from bottom to top
        for row in range(size-2, -1, -1):
            delta_left = int(np_random.normal()) # num of std devs
            
            # delta_right depends on row number
            if row > 3*size/4:
                mean = 0
            elif row > size/2:
                mean = 1
            elif row > size/4:
                mean = 2
            else:
                mean = None

            delta_right = 0
            if mean is not None:
                delta_right = int(np_random.normal(mean))

            # new edges for track on this row
            left = min(max(0, left+delta_left), right-4)
            right = max(min(size-1, right+delta_right), left+4)

            grid[row, left:right+1] = 1

        return grid
    
    def __init__(self):
        self.size = 20
        self.grid = self.generate_turn(size)
        self.y = self.size-1
        self.x = np_random.choice(np.where(self.grid[self.y] == 1))
        self.vy = 0
        self.vx = 0

    def step(self, dvx, dvy):
        # change velocity, clipped to be in [0,4] in each dimension
        # ensure vx and vy will not both be zero
        self.vx = max(0, min(4, self.vx+dvx))
        miny = int(self.vx == 0) # is 1 if vx is 0
        self.vy = max(miny, min(4, self.vy+dvy))

        # calculate new position
        old_pos = (self.x, self.y)
        self.x += self.vx
        self.y += self.vy
        new_pos = (self.x, self.y)

        # if crosses track boundary. (not sure how to do this for general shapes. will check points along line for now)
            # if boundary is finish line, success
            # else, reset position and velocity as at start of episode
        # reward is -1 per step
        # return new state and reward
        
