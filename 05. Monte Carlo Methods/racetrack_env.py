import numpy as np

class TurnEnv():
    def generate_turn(self, size):
        assert size >= 5

        grid = np.zeros((size,size)) # turn is subset of grid

        # set start line as bottom row
        start_line_len = 4
        left = size//5
        right = left+start_line_len
        grid[size-1-0, left:right+1] = 2

        # fill in the remaining rows from bottom to top
        for row in range(size-2, -1, -1):
            delta_left = int(self.np_random.normal()) # num of std devs
            left = min(max(0, left+delta_left), right-4)
            
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
                delta_right = int(self.np_random.normal(mean))
                right = max(min(size-1, right+delta_right), left+4)
                grid[row, left:right+1] = 1
            else:
                right = max(min(size-1, right+delta_right), left+4)
                grid[row, left:right] = 1
                grid[row,right] = 3  # finish line
        return grid
    
    def __init__(self, size):
        self.np_random = np.random.RandomState()
        self.size = size
        self.num_speeds = len([0,1,2,3,4])
        self.valid_steps = [-1, 0, 1]
        self.grid = self.generate_turn(self.size)

    def get_state(self):
        return (self.x, self.y, self.vx, self.vy)

    def reset(self):
        self.y = self.size-1
        indices, = np.where(self.grid[self.y] == 2)
        self.x = self.np_random.choice(indices)
        self.vy = 0
        self.vx = 0

    # returns (state, reward, done)
    def step(self, dvx, dvy):

        # change velocity, clipped to be in [0,4] in each dimension
        # ensure vx and vy will not both be zero
        self.vx = max(0, min(4, self.vx+dvx))
        miny = int(self.vx == 0) # is 1 if vx is 0
        self.vy = max(miny, min(4, self.vy+dvy))

        # calculate new position
        old_pos = [self.x, self.y]
        self.x += self.vx
        self.y -= self.vy

        # if crosses track boundary. (not sure how to do this for general shapes. will check points along line for now)
        for i in range(1,11):
            y = old_pos[1] - i*self.vy/10
            x = old_pos[0] + i*self.vx/10

            if y < 0 or x > self.size-1 or \
                    self.grid[int(y), int(x)] == 0:  # hit a wall
                self.reset()
                return self.get_state(), -1, False
            elif self.grid[int(y), int(x)] == 3:  # crossed finishline
                self.reset()
                return self.get_state(), -1, True

        return self.get_state(), -1, False
