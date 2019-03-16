# for recreating Fig. 8.8
# (8.1): Q(s,a) = sum p(s'|s,a)[r(s,a,s')+gamma*max_{a'}Q(s',a')]
#   one-step expected tabular updates

# The resulting figure isn't exactly the same as the one in the book.
# I would guess that the environment isn't set up the same way.
# The same qualitative behavior is observed, though.

from branch_env import BranchEnv
from dyna_q import get_eps_action
import numpy as np
import matplotlib.pyplot as plt

gamma = 1
runs = 10

def plot():
    for b in [1,3,10]:
        plot_branch(b)

    plt.legend()
    plt.show()

# averages many runs of experiment and plots results
def plot_branch(b):
    num_updates = np.floor(np.linspace(1,20000,50))   # x values 
    avg_uniform = np.zeros(len(num_updates))
    avg_on_policy = np.zeros(len(num_updates))
    for run in range(runs):
        env = BranchEnv(1000, b)
        print(str(run)+"/"+str(runs))
        avg_uniform += (uniform(num_updates, env)-avg_uniform)/(run+1)
        avg_on_policy += (on_policy(num_updates, env)-avg_on_policy) \
                /(run+1)
    plt.plot(num_updates, avg_uniform, label="uniform"+str(b))
    plt.plot(num_updates, avg_on_policy, label="on-policy"+str(b))


# sweeps through state-action space to update Q table
def uniform(num_updates, env):
    update_count = 0    # number of expected updates so far
    Q = np.zeros((env.num_states, 2))   # value table
    start_values = []    # y values for plot

    while update_count < 20000:
        for state in range(env.num_states): # sweep
            for action in [0,1]:
                update(Q, state, action, env)
                update_count += 1

                # update plot
                if update_count in num_updates:
                    start_values.append(get_start_value(Q,env))

            # check if done
            if update_count >= 20000:
                break
    return np.array(start_values)


def on_policy(num_updates, env):
    update_count = 0    # number of expected updates so far
    Q = np.zeros((env.num_states, 2))   # value table
    start_values = []    # y values for plot

    while update_count < 20000:
        state = env.state
        action = get_eps_action(state, Q, env, 0) # argmax action
        update(Q, state, action, env)   # one-step expected update
        update_count += 1

        # update plot
        if update_count in num_updates:
            start_values.append(get_start_value(Q,env))

        # transition to next state according to current policy
        env.step(action)
    return np.array(start_values)


# one-step expected update to Q table
def update(Q, state, action, env, update=True):
    expectation = 0
    prob = 1/(env.b + 1) # uniform state transition probability

    # non-terminal next state
    for next_state in env.transitions[(state,action)]:
        r = env.rewards[(state,action,next_state)]
        expectation += prob*(r + gamma*np.max(Q[next_state]))

    # for terminal next state
    next_state = env.terminal
    r = env.rewards[(state,action,next_state)]
    expectation += prob*r

    # save result
    if update:
        Q[state,action] = expectation
    else:
        return expectation

# gets value of starting state given current greedy policy wrt Q
def get_start_value(Q, env):
    V = np.zeros(env.num_states)
    while True:
        delta = 0
        for state in range(env.num_states):
            v = V[state]
            action = np.argmax(Q[state]) # picks 0 on ties
            V[state] = update(Q, state, action, env, False)
            delta = max(delta, abs(v - V[state]))
        if delta < 0.05:
            break
    return V[env.start]
plot()
