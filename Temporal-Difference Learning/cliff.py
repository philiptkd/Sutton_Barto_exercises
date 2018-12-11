# for Example 6.6

from cliff_env import CliffEnv
from windy_sarsa import step_taker, get_state_index, choose_action, \
        print_trajectory
import numpy as np
import matplotlib.pyplot as plt

# parameters
eps = 0.1
alpha = 0.5
gamma = 1
episodes = 500
runs = 100

def q_step_taker(env,alpha=alpha):
    Q = np.zeros((env.height*env.width, len(env.actions)))
    s_idx = get_state_index(env) # state index

    while True:
        # get R, S', and A'
        a_idx = choose_action(Q, env, eps=eps) # action index. eps-greedy
        _, reward, done = env.step(env.actions[a_idx])
        
        ns_idx = get_state_index(env) # new state index
        eval_action_idx = choose_action(Q, env, greedy=True) # new action idx. greedy
        
        # update value function
        if done:
            Q[s_idx,a_idx] += alpha*(reward + 0 - Q[s_idx,a_idx])
        else:
            Q[s_idx,a_idx] += alpha*(reward + gamma*Q[ns_idx,eval_action_idx] - Q[s_idx,a_idx])

        # S := S', A := A'
        s_idx = ns_idx

        yield reward, done, Q

def control(step_gen):
    env = CliffEnv()
    reward_sums = np.zeros(episodes)
    steps = step_gen(env) # takes sarsa action and update step
    for episode in range(episodes):
        done = False
        while not done:
            reward, done, Q = next(steps)
            reward_sums[episode] += reward
    return reward_sums, Q

# averages results over many runs
def avg_results():
    for method,step_gen in zip(["sarsa","qlearning"],[step_taker, q_step_taker]):
        reward_sum = np.zeros(episodes)
        Q = None
        for run in range(runs):
            print(run)
            r, q = control(step_gen)
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

#avg_results()
