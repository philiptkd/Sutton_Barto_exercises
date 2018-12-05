# for Exercise 5.12
from racetrack_env import TurnEnv
import numpy as np
import matplotlib.pyplot as plt

def main():
    # initialize
    env = TurnEnv(7)
    num_actions = len(env.valid_steps)**2
    Q = np.zeros((env.size, env.size, env.num_speeds, env.num_speeds, num_actions))
    C = np.zeros((env.size, env.size, env.num_speeds, env.num_speeds, num_actions))
    env.reset()

    # parameters
    episodes = 1000
    gamma = 1
    eps = 0.1

    # for each episode
    for episode in range(episodes):
        print(episode)
        trajectory = soft_episode(Q, env, eps)
        G = 0   # return
        W = 1   # importance sampling weight

        # for each step in episode, in reverse order
        for t in range(len(trajectory)-1, -1, -1):
            state, action, reward = trajectory[t]
            x,y,vx,vy = state
            
            # update statistics
            G = gamma*G + reward
            c = C[x,y,vx,vy,action]
            q = Q[x,y,vx,vy,action] 
            c += W
            Q[x,y,vx,vy,action] = q + (W/c)*(G-q)
            C[x,y,vx,vy,action] = c

            # no need to continue if W will always be 0 starting now
            greedy_action = np.argmax(Q[x,y,vx,vy])
            if action != greedy_action:
                break

            # update product of importance sampling ratios
            b = (1-eps) + (eps/num_actions)
            W = W*(1/b)

            


def soft_episode(Q, env, eps):
    num_actions = len(env.valid_steps)**2
    done = False

    # generate trajectory with soft policy
    x,y,vx,vy = env.get_state()
    trajectory = []
    while not done:
        # choose action
        if env.np_random.uniform() < eps:
            action = env.np_random.choice(range(num_actions))
        else:
            action = np.argmax(Q[x,y,vx,vy])

        # take action
        state, reward, done = env.step(*expand_action(action, env))

        # record
        old_state = x,y,vx,vy
        trajectory.append((old_state, action, reward))

        # prepare for next step
        x,y,vx,vy = state

    return trajectory

# converts action from range(9) to {-1,0,1} X {-1,0,1}
def expand_action(action, env):
    dvx = action//len(env.valid_steps) - 1
    dvy = action%len(env.valid_steps) - 1
    
    return dvx, dvy

# converts action from range(3)Xrange(3) to range(9)
def compress_action(dvx, dvy, env):
    return (dvx+1)*len(env.valid_steps) + (dvy+1)


main()
