# for Fig. 6.3
# uses many fewer iterations than the authors did. to save time

from cliff_env import CliffEnv
from windy_sarsa import step_taker, choose_action, get_state_index
from cliff import q_step_taker
import numpy as np
import matplotlib.pyplot as plt

# parameters
eps = 0.1
gamma = 1
alphas = np.linspace(.1,1,10)
interim_episodes = 100
asymptotic_episodes = 1000
interim_runs = 50
asymptotic_runs = 5

# does TD control with given update generator and for given number of episodes
def control(step_gen, episodes, runs):
    env = CliffEnv()
    avg_reward_sum = np.zeros(len(alphas))
    for i,alpha in enumerate(alphas):
        print("alpha="+str(alpha))
        for run in range(runs):
            steps = step_gen(env, alpha)
            for episode in range(episodes):
                reward_sum = 0
                done = False
                while not done:
                    reward, done, Q = next(steps)
                    reward_sum += reward
                avg_reward_sum[i] += (reward_sum - avg_reward_sum[i])/(run*episodes + episode+1)
        del steps
    return avg_reward_sum

# averages results over many runs
def plot_results():
    for method,step_gen in zip(["ex_sarsa","sarsa","qlearning"],[ex_sarsa_step_taker, step_taker, q_step_taker]):
        for stage,episodes,runs in zip(["interim ", "asymptotic "], [interim_episodes,asymptotic_episodes], [interim_runs,asymptotic_runs]):
            print(stage+method)
            avg_reward_sum = control(step_gen, episodes, runs)
            plt.plot(alphas, avg_reward_sum, label=stage+method)

    plt.ylabel("Avg reward sum per episode")
    plt.xlabel("Alpha")
    plt.legend()
    plt.show()

def ex_sarsa_step_taker(env,alpha):
    Q = np.zeros((env.height*env.width, len(env.actions)))
    s_idx = get_state_index(env) # state index

    while True:
        # get R, S', and A'
        a_idx = choose_action(Q, env, eps) # action index. eps-greedy
        _, reward, done = env.step(env.actions[a_idx])
        
        ns_idx = get_state_index(env) # new state index
        max_action_idx = choose_action(Q, env, greedy=True) # new action idx. greedy

        # update value function
        if done:
            Q[s_idx,a_idx] += alpha*(reward + 0 - Q[s_idx,a_idx])
        else:   # make expected update
            ex_next_value = 0
            for na_idx in range(len(env.actions)): # for each possible next action index
                ex_next_value += (eps/len(env.actions))*Q[ns_idx, na_idx]   # contribute to expected value of next state
            ex_next_value += (1-eps)*Q[ns_idx, max_action_idx]

            Q[s_idx,a_idx] += alpha*(reward + gamma*ex_next_value - Q[s_idx,a_idx])

        # S := S', A := A'
        s_idx = ns_idx

        yield reward, done, Q

plot_results()
