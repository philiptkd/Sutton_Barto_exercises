# for Example 6.5

from windy_env import WindyEnv
import numpy as np
import matplotlib.pyplot as plt

# parameters
eps = 0.1
alpha = 0.5
gamma = 1
max_steps = 8000

def main(env):
    # initialization
    num_episodes = np.zeros(max_steps) # which episode each step was a part of
    episode_count = 0 # which episode we're on
    steps = step_taker(env)

    # go for max_steps steps
    for step in range(max_steps):
        # save count for plot
        num_episodes[step] = episode_count
        _, done, Q = next(steps)
        if done:    
            episode_count += 1

    #print_trajectory(Q, env)

    plt.plot(range(max_steps), num_episodes)
    plt.xlabel("Time steps")
    plt.ylabel("Episodes")
    plt.show()

# generator for steps in environment that update Q
def step_taker(env,alpha=alpha):
    Q = np.zeros((env.height*env.width, len(env.actions)))
    s_idx = get_state_index(env) # state index
    a_idx = choose_action(Q, env) # action index

    while True:
        # get R, S', and A'
        _, reward, done = env.step(env.actions[a_idx])
        ns_idx = get_state_index(env) # new state index
        na_idx = choose_action(Q, env) # new action index
        
        # update value function
        if done:
            Q[s_idx,a_idx] += alpha*(reward + 0 - Q[s_idx,a_idx])
        else:
            Q[s_idx,a_idx] += alpha*(reward + gamma*Q[ns_idx,na_idx] - Q[s_idx,a_idx])


        # S := S', A := A'
        s_idx = ns_idx
        a_idx = na_idx

        yield reward, done, Q

# prints sequence of states from start to goal
def print_trajectory(Q, env):
    env.reset()
    trajectory = []
    done = False

    action = np.argmax(Q[get_state_index(env)])
    _, reward, done = env.step(env.actions[action])
    while not done:
        trajectory.append((env.state, action, reward))
        action = np.argmax(Q[get_state_index(env)])
        _, reward, done = env.step(env.actions[action])
    trajectory.append(list(env.goal))

    print(trajectory)

# converts current state from (row,col) to single index
def get_state_index(env):
    row,col = env.state
    return row*env.width + col

# returns action_num, an index into env.actions
def choose_action(Q, env, eps=eps, greedy=False):
    if not greedy and env.np_random.uniform() < eps:
        return env.np_random.randint(len(env.actions))
    else:
        state_index = get_state_index(env)
        action_num = np.argmax(Q[state_index])
        return action_num

#main(WindyEnv())
