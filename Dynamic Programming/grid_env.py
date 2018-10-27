import numpy as np

# creates a gridworld environment of the given width and height
# terminal_states is a list of states that have 0 reward and deterministically transition the agent back to themselves
# the states are numbered left to right and then top to bottom, starting with 0
class GridEnv():
    def __init__(self, rng, width, height, terminal_states, gamma=1., initial_state=None, wind=None):
        # make sure terminal_states are valid states for this size gridworld
        for state in terminal_states:
            assert state < width*height

        self.rng = rng
        self.width = width
        self.height = height
        self.num_states = width*height
        self.gamma = gamma
        self.terminal_states = terminal_states
        self.directions = ['North', 'East', 'South', 'West']
        self.num_actions = len(self.directions)

        # choose initial state as any of the valid states, all with equal probability
        if initial_state==None:
            valid_start_states = [i for i in range(width*height) if i not in terminal_states]
            self.state = rng.choice(valid_start_states)
        else:
            self.state = initial_state

        # wind is the amount of randomness in each direction
        # it's the probability of being pushed in the east and north directions
        # a negative number indicates a positive probability of being pushed west or south
        if wind==None:
            self.wind = [0.,0.]
        else:
            assert isinstance(wind, list) and len(wind)==2
            self.wind = wind

        self.transition_probs = self.get_transition_probs()

    def get_transition_probs(self):
        tp = np.zeros((self.num_states, self.num_actions, self.num_states)) # P(s,a,s'). r is always -1 unless in terminal state
        for state in range(self.num_states):
            for action in range(self.num_actions):
                next_state = self.next_state(state, self.directions[action])
                tp[state, action, next_state] = 1.

                # account for wind
                # assumes east and north winds are independent
                # assumes order of winds is irrelevant
                post_east_wind_state = next_state
                if self.wind[0] > 0:    # pushed east
                    post_east_wind_state = self.next_state(next_state, 'East')
                elif self.wind[0] < 0:  # pushed west
                    post_east_wind_state = self.next_state(next_state, 'West')

                post_north_wind_state = post_east_wind_state
                post_both_winds_state = post_east_wind_state
                if self.wind[1] > 0:    # pushed north
                    post_north_wind_state = self.next_state(next_state, 'North')
                    post_both_winds_state = self.next_state(post_east_wind_state, 'North')
                elif self.wind[1] < 0:  # pushed south
                    post_north_wind_state = self.next_state(next_state, 'South')
                    post_both_winds_state = self.next_state(post_east_wind_state, 'South')

                probEast = abs(self.wind[0])    # prob of being pushed east
                probNorth = abs(self.wind[1])   # prob of being pushed north
                tp[state, action, next_state] -= probEast + probNorth - probEast*probNorth # prob of staying
                tp[state, action, post_east_wind_state] += probEast*(1 - probNorth) # only pushed east
                tp[state, action, post_north_wind_state] += probNorth*(1 - probEast) # only pushed north
                tp[state, action, post_both_winds_state] += probEast*probNorth # pushed east and north

        tp[self.terminal_states, :, :] = 0.
        tp[self.terminal_states, :, self.terminal_states] = 1.
        return tp

    # transitions deterministically according to action
    # we assume we're not in a terminal state
    def next_state(self, state, action):
        if action == 'North':
            if state - self.width < 0:  # if we're on the top edge
                next = state
            else:
                next = state - self.width
        
        elif action == 'East':
            if (state + 1) % self.width == 0:   # if we're on the right edge
                next = state
            else:
                next = state + 1
        
        elif action == 'South':
            if state + self.width >= self.height*self.width:    # if we're on the bottom edge
                next = state
            else:
                next = state + self.width

        elif action == 'West':
            if state % self.width == 0: # if we're on the left edge
                next = state
            else:
                next = state - 1

        return next

    # given an action a, samples the reward and next state
    # all rewards are -1 unless in a terminal state
    def sample(self, action):
        if self.state in self.terminal_states:
            return 0, self.state
        else:
            next_state_distribution = self.transition_probs[self.state, action]
            uniform_sample = self.rng.uniform()
            next_state = -1
            while uniform_sample > 0:
                next_state += 1
                uniform_sample -= next_state_distribution[next_state]
            return -1, next_state