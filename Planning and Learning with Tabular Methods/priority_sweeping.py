# for reproducing figure in Example 8.4

import numpy as np
import matplotlib.pyplot as plt
from maze_env import MazeEnv
from p_queue import pQueue
from dyna_q import get_eps_action
from dyna_q import get_state_idx

runs = 10
gamma = 0.95
alpha = 0.5
eps = 0 # greedy
delta = 0.0001
n = 5

def get_steps():
    # initialization
    env = MazeEnv(1)
    Q = np.zeros((env.width*env.height,len(env.actions)))
    model = {}
    pq = pQueue()
    steps = 0
    updates = 0

    while True:
        steps += 1
        
        # step and record result
        state = get_state_idx(env.state, env)
        action = get_eps_action(state, Q, env, eps)
        reward, done = env.step(action)
        next_state = -1 if done else get_state_idx(env.state, env)
        model[(state,action)] = (reward, next_state)
        
        # check if maze is solved
        if done:
            print(updates)
            if steps == 14:     # optimum solution
                return updates
            steps = 0

        # calculate priority and put in queue
        target = reward
        if not done:
            max_action = get_eps_action(next_state, Q, env, 0)
            target += gamma*Q[next_state, max_action]
        priority = abs(target - Q[state, action])
        if priority > delta:
            pq.enqueue(priority, state, action)

        # sweep through update queue
        for i in range(n):
            if len(pq.queue) == 0: # stop if the queue is empty
                break

            # one step qlearn from most urget (s,a)
            _,s,a = pq.dequeue()
            r,sp = model[(s,a)]
            target = r
            if sp != -1:
                max_a = get_eps_action(sp, Q, env, 0)
                target += gamma*Q[sp, max_a]
            Q[s,a] += alpha*(target - Q[s,a])

            # increment count of updates
            updates += 1
        
            # add predecessors to queue
            add_pred(model, s, pq, Q, env)


def add_pred(model, sp, pq, Q, env):
    for (s,a) in model.keys():
        transition = model[(s,a)]
        if transition[1] == sp:
            r = transition[0]
            max_a = get_eps_action(sp, Q, env, 0)
            P = abs(r + gamma*Q[sp, max_a] - Q[s,a])
            if P > delta:
                pq.enqueue(P, s, a)

get_steps()
