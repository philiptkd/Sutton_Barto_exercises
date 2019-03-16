# to recreate Fig. 6.5

import numpy as np
import matplotlib.pyplot as plt

episodes = 300
runs = 10000
eps = 0.1
alpha = 0.1
gamma = 1
num_b_actions = 10

class SimpleEnv():
    def __init__(self):
        self.states = ["A","B"]
        self.a_actions = ["left","right"]
        self.np_random = np.random.RandomState()
        self.reset()

    def reset(self):
        self.state = self.states.index("A")

    # returns (reward, done) given int action
    def step(self, action):
        if self.state == self.states.index("A"):
            if self.a_actions[action] == "right":
                self.reset()
                return 0, True
            else: # if action is "left"
                self.state = self.states.index("B")
                return 0, False

        else: # if self.state == "B"
            self.reset()
            reward = self.np_random.normal(-.1,1)
            return reward, True


def double_q(single_q=False):
    env = SimpleEnv()
    Q1 = [np.zeros(2), np.zeros(num_b_actions)] 
    Q2 = [np.zeros(2), np.zeros(num_b_actions)] 
    left = np.zeros(episodes) # whether or not took "left" from A

    for episode in range(episodes):
        done = False

        while not done:
            state = env.state
            action = get_eps_action(sum_qs(Q1,Q2), env)
            reward, done = env.step(action)
            next_state = env.state
            step = (state,action,reward,next_state,done)

            # update return statistics
            if env.states[state] == "A":
                if env.a_actions[action] == "left":
                    left[episode] = 1

            if single_q:
                double_q_update(Q1,Q1,env,step) 
            elif env.np_random.uniform() < 0.5:
                double_q_update(Q1,Q2,env,step)
            else:
                double_q_update(Q2,Q1,env,step)
    return left

def sum_qs(Q1, Q2):
    return [Q1[0]+Q2[0], Q1[1]+Q2[1]]

# update Q1 towards Q2 estimate of value of Q1's max action
def double_q_update(Q1, Q2, env, step):
    state, action, reward, next_state, done = step
    target = reward
    if not done:
        max_a = get_eps_action(Q1, env, 0)
        target += gamma*Q2[next_state][max_a]
    Q1[state][action] += alpha*(target - Q1[state][action])

def get_eps_action(Q, env, eps=eps):
    if env.np_random.uniform() < eps:
        action = env.np_random.randint(0, len(Q[env.state]))
    else:
        action = env.np_random.choice(np.flatnonzero(Q[env.state] == Q[env.state].max())) # to select argmax randomly
    return action

single_lefts = np.zeros(episodes)
double_lefts = np.zeros(episodes)
for run in range(runs):
    if run%100 == 0:
        print(run)

    # q-learning
    single_lefts += (double_q(single_q=True) - single_lefts)/(run+1)

    # double q-learning
    double_lefts += (double_q() - double_lefts)/(run+1)

print(single_lefts[0])
plt.plot(range(episodes), single_lefts, label="q-learning")
plt.plot(range(episodes), double_lefts, label="double q-learning")
plt.legend()
plt.xlabel("Episodes")
plt.ylabel("% left")
plt.show()
