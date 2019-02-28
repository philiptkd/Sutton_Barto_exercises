# for recreating Fig. 8.8

import numpy as np

class BranchEnv():
    def __init__(self, num_states, b):
        self.eps = 0.1
        self.num_states = num_states
        self.terminal = num_states  # index of terminal state
        self.np_random = np.random.RandomState()
        self.b = b  # branching factor
        self.setup_dynamics()
        self.start = 0  # arbitrary starting state
        self.reset()

    def reset(self):
        self.state = self.start

    # sets transition probabilities P(s'|s,a) and reward fn R(s,a,s')
    def setup_dynamics(self):
        self.transitions = {}# dict of equi-likely state transitions
        self.rewards = {}# dict of rewards for each (s,a,s') tuple
        for state in range(self.num_states):
            for action in [0,1]:    # two possible actions
                self.transitions[(state,action)] = \
                        self.np_random.choice(self.num_states, 
                                                size=self.b)

                # includes reward for transition to terminal state
                for next_state in range(self.num_states+1):
                    self.rewards[(state,action,next_state)] = \
                            self.np_random.randn()# standard normal

    # returns next_state, reward, done
    def step(self, action):
        # if we go to terminal state
        if self.np_random.random_sample() < self.eps:
           r = self.rewards[(self.state, action, self.terminal)]
           done = True
           self.reset()
           return self.state, r, done
        
        # otherwise, transition to next state
        next_state = self.np_random.choice(
                self.transitions[(self.state, action)])
        r = self.rewards[(self.state, action, next_state)]
        done = False
        self.state = next_state
        return next_state, r, done


