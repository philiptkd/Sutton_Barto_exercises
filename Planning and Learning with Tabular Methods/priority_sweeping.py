# for reproducing figure in Example 8.4,
#   but I'm only testing with the smallest grid size, one data point
# Sometimes gets stuck in suboptimal path,
#   but I did see the cited 5-10x improvement in number of updates
#   over random dyna

import numpy as np
import matplotlib.pyplot as plt
from maze_env import MazeEnv
from p_queue import pQueue
from dyna_q import get_eps_action, get_state_idx, dyna_q

runs = 10
gamma = 0.95
alpha = 0.5
delta = 0.0001
n = 5
eps = 0.1

def check_if_solved(Q):
    env = MazeEnv(1)
    for i in range(14): # optimal solution
        state = get_state_idx(env.state, env)
        action = get_eps_action(state, Q, env, 0)
        reward, done = env.step(action)
    if done:
        return True
    return False

def get_updates():
    # initialization
    env = MazeEnv(1)
    Q = np.zeros((env.width*env.height,len(env.actions)))
    model = {}
    pq = pQueue()
    steps = 0
    updates = 0

    while True:
        if updates%100 == 0:
            if check_if_solved(Q):
                return updates

        steps += 1
        
        # step and record result
        state = get_state_idx(env.state, env)
        action = get_eps_action(state, Q, env, eps)
        reward, done = env.step(action)
        next_state = -1 if done else get_state_idx(env.state, env)
        model[(state,action)] = (reward, next_state)
        
        if done:
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

# function to add predecessors to state sp to the priority queue
def add_pred(model, sp, pq, Q, env):
    for (s,a) in model.keys():
        transition = model[(s,a)]
        if transition[1] == sp:
            r = transition[0]
            max_a = get_eps_action(sp, Q, env, 0)
            P = abs(r + gamma*Q[sp, max_a] - Q[s,a])
            if P > delta:
                pq.enqueue(P, s, a)

# regular (random) dyna-q
def get_dyna_updates():
    # initialization
    env = MazeEnv(1)
    Q = np.zeros((env.width*env.height,len(env.actions)))
    model = {}
    steps = 0
    updates = 0

    step_gen = dyna_q(n, env)
    while True:
        # step
        _,_,_,_,done = next(step_gen)
        steps += 1
        updates += 1
        
        # update counts
        if done:
            updates += 1
            if steps == 14:
                return updates
            steps = 0
        else:
            updates += n

def compare():
    sweep_updates = 0
    dyna_updates = 0
    for run in range(runs):
        sweep_updates += (get_updates() - sweep_updates)/(run+1)
        print("running avg of # of sweep updates: {0}".format(sweep_updates))
        dyna_updates += (get_dyna_updates() - dyna_updates)/(run+1)
        print("running avg of # of dyna updates: {0}".format(dyna_updates))

compare()
