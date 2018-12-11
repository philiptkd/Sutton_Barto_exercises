# for Example 6.6

from cliff_env import CliffEnv
import numpy as np
import matplotlib.pyplot as plt

# parameters
eps = 0.1
alpha = 0.5
gamma = 1
episodes = 500
runs = 100

def control(method):
    # initialization
    env = CliffEnv()
    Q = np.zeros((env.height*env.width, len(env.actions)))
    reward_sums = np.zeros(episodes)
    s_idx = get_state_index(env) # state index
    a_idx = choose_action(Q, env) # action index. eps-greedy

    # go for max_steps steps
    for episode in range(episodes):
        done = False
        while not done:
            # get R, S', and A'
            reward, done = env.step(env.actions[a_idx])
            reward_sums[episode] += reward

            ns_idx = get_state_index(env) # new state index
            na_idx = choose_action(Q, env) # new action index. the action to take next. eps-greedy

            # if doing q-learning, need to update toward optimal value function q*
            if method == "q":
                eval_action_idx = choose_action(Q, env, True) # new action idx. greedy
            else:
                eval_action_idx = na_idx
            
            # update value function
            if done:
                Q[s_idx,a_idx] += alpha*(reward + 0 - Q[s_idx,a_idx])
            else:
                Q[s_idx,a_idx] += alpha*(reward + gamma*Q[ns_idx,eval_action_idx] - Q[s_idx,a_idx])

            # S := S', A := A'
            s_idx = ns_idx
            a_idx = na_idx
    return reward_sums, Q
        
# prints sequence of states from start to goal
def print_trajectory(Q, env):
    env.reset()
    trajectory = []
    done = False

    action = np.argmax(Q[get_state_index(env)])
    reward, done = env.step(env.actions[action])
    while not done:
        trajectory.append((env.state, action, reward))
        action = np.argmax(Q[get_state_index(env)])
        reward, done = env.step(env.actions[action])
    trajectory.append(list(env.goal))

    print(trajectory)

# converts current state from (row,col) to single index
def get_state_index(env):
    row,col = env.state
    return row*env.width + col

# returns action_num, an index into env.actions
def choose_action(Q, env, greedy=False):
    if not greedy and env.np_random.uniform() < eps:
        return env.np_random.randint(len(env.actions))
    else:
        state_index = get_state_index(env)
        action_num = np.argmax(Q[state_index])
        return action_num

# averages results over many runs
def avg_results():
    for method in ["sarsa", "q"]:
        reward_sum = np.zeros(episodes)
        Q = None
        for run in range(runs):
            print(run)
            r, q = control(method)
            reward_sum += (r - reward_sum)/(run+1)
            if Q is None:
                Q = q
            else:
                Q += (q - Q)/(run+1)

        plt.plot(range(episodes), reward_sum, label=method)
        print_trajectory(Q, CliffEnv())

    plt.ylabel("Reward sum")
    plt.xlabel("Episodes")
    plt.legend()
    plt.show()

avg_results()
