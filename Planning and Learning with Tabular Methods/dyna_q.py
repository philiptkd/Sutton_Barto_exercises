# for reproducing Fig. 8.2

import numpy as np
import matplotlib.pyplot as plt
from maze_env import MazeEnv

runs = 30
episodes = 50
gamma = 0.95
alpha = 0.5
eps = 0.1

def plot_fig():
    env = MazeEnv(1)
    for n in [0, 5, 50]:
        env.np_random = np.random.RandomState(1234) # set common seed
        plot_steps(n, env)
    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.show()

def plot_steps(n, env):
    steps = np.zeros(episodes) # steps per episode
    for run in range(runs):
        print(run)
        steps += (get_steps(n, env) - steps)/(run+1)
    plt.plot(range(episodes), steps, label="n={0}".format(n))

# records number of steps per episode
def get_steps(n, env):
    steps = np.zeros(episodes) # steps per episode
    episode = 0
    step = 0
    step_gen = dyna_q(n, env, False)
    while episode < episodes:
        _,_,_,_,done = next(step_gen)
        step += 1
        if done:
            steps[episode] = step
            step = 0
            episode += 1
    return steps

# generator that yields (state, action, reward, next_state, done)
def dyna_q(n, env, plus=False, kappa=None, env_switch_to=None):
    Q = np.zeros((env.height*env.width, len(env.actions)))
    model = {} # record of past experience. assumes deterministic
    staleness = np.zeros_like(Q) 
    step_num = 0
    env.reset()

    if plus:
        model = {(s,a):(0,s) for s in range(env.width*env.height) for a in range(len(env.actions))}

    state_idx = get_state_idx(env.state, env)
    while True:
        # step and record result
        action, reward, next_state_idx, done = step(state_idx, env, Q)
        
        # switch env after a certain number of steps
        step_num += 1
        if env_switch_to is not None:
            if step_num == env_switch_to[1]:
                env.set_grid(env_switch_to[0])


        # one-step q learning
        q_learn(state_idx, action, reward, next_state_idx, done, Q, env)

        # add exploration bonus for dyna-q+
        plan_target = reward
        if plus:
            plan_target += kappa*np.sqrt(staleness[state_idx, action])
            staleness += 1
            staleness[state_idx, action] = 0 

        # save experience to model
        if not done:
            model[(state_idx, action)] = (plan_target, next_state_idx)
        else: # if done
            model[(state_idx, action)] = (plan_target, -1)

        # n iterations of one-step q planning
        q_plan(n, env, model, Q, plus)
        
        yield state_idx, action, reward, next_state_idx, done

        # setup for next step
        state_idx = next_state_idx


def q_plan(n, env, model, Q, plus):
    keys = list(model.keys())
    for i in range(n):
        key = env.np_random.choice(range(len(keys)))
        s, a = keys[key]
        r, sp = model[(s,a)]
        target = r
        if sp != -1:
            max_action = get_eps_action(sp, Q, env, 0)
            target += gamma*Q[sp, max_action]
        Q[s,a] += alpha*(target - Q[s,a])

def q_learn(state_idx, action, reward, next_state_idx, done, Q, env):
    target = reward
    if not done:
        max_action = get_eps_action(next_state_idx, Q, env, 0)
        target += gamma*Q[next_state_idx, max_action]
    Q[state_idx, action] += alpha*(target - Q[state_idx, action])

def step(state_idx, env, Q):
    action = get_eps_action(state_idx, Q, env)
    reward, done = env.step(action)
    next_state_idx = get_state_idx(env.state, env)
    return action, reward, next_state_idx, done

# convert state from (row, col) to int index
def get_state_idx(state, env):
    return state[0]*env.width + state[1]

# make eps-greedy action selection
def get_eps_action(state, Q, env, eps=eps):
    if env.np_random.uniform() < eps:
        action = env.np_random.choice(range(len(env.actions)))
    else:
        action = env.np_random.choice(np.flatnonzero(Q[state] == Q[state].max())) # to select argmax randomly
    return action

#plot_fig()
