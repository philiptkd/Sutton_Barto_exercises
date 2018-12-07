# for Example 6.2

import numpy as np
import matplotlib.pyplot as plt

episodes = 100
gamma = 1

class WalkMRP():
    def __init__(self):
        self.states = ['Zero', 'A', 'B', 'C', 'D', 'E', 'One']
        self.state = 3 # 'C'
        self.np_random = np.random.RandomState()

    # returns reward, done
    def step(self):
        step = self.np_random.choice([-1, 1])
        self.state += step
        if self.states[self.state] == 'One':
            self.state = 2 # reset
            return 1, True
        if self.states[self.state] == 'Zero':
            self.state = 2 # reset
            return 0, True
        return 0, False

def td(alpha):
    env = WalkMRP()
    V = np.zeros(len(env.states))
    V[1:-1] = 0.5
    for episode in range(episodes):
        done = False
        state = env.state
        while not done:
            reward, done = env.step()
            next_state = env.state
            if done:
                td_error = reward + 0 - V[state] # terminal states have value zero
            else:
                td_error = reward + gamma*V[next_state] - V[state]

            V[state] += alpha*td_error
            state = next_state
    
        if (episode+1) in [1,10,100]:
            plt.plot(range(5), V[1:-1], label=str(episode+1))
    plt.legend()
    plt.show()

td(.1)
