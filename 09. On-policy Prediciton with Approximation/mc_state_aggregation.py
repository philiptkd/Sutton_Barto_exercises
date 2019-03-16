# used to recreate Fig. 9.1

from random_walk_env import RandomWalkEnv
import numpy as np
import matplotlib.pyplot as plt
import pickle


# returns a list of tuples (S_t, R_t)
#   the final tuple's state will be env.start and should not be used
def get_trajectory(env):
    trajectory = []
    env.reset() # redundancy
    trajectory.append((env.state, 0)) # R_0 = 0
    done = False
    while not done:
        next_state, reward, done = env.step()
        trajectory.append((next_state, reward))
    return trajectory


# gradient Monte Carlo as on page 202
# assumes 1000 states in env
def grad_mc(episodes, alpha, env, width, gamma=1, offset=None, tilings=1):
    if offset is None:
        assert tilings==1
        offset = width   # no overlap between groups
    assert tilings*offset==width
    num_features = int(np.ceil(env.num_states/width))
    w = np.zeros((tilings,num_features+1))
    for episode in range(episodes):
        if episode%1000==0:
            print("episode: "+str(episode)+"/"+str(episodes))

        trajectory = get_trajectory(env)
        G = 0
        for t in range(len(trajectory)-2, -1, -1): # t = T-1,...,1,0
            r = trajectory[t+1][1]
            s = trajectory[t][0]
            G = gamma*G + r

            if offset == width: # state aggregation. faster.
                group = (s-1)//width  # 1<=s<=1000, but group indices start at 0
                w[0,group] += alpha*(G - w[0,group])    # v(s,w) = w[group]
            else:   # more general version
                v, idxs = get_v(w,s,width,offset,tilings)
                w[range(tilings), idxs] += alpha*(G - v)
        yield w

def which_tile(s, width, offset, t):
    tile_idx = (s-1-t*offset)//width
    if t!=0:
        tile_idx += 1
    return tile_idx

#w has dims (tilings,num_features)
#1<=s<=env.num_states
def get_v(w,s,width,offset,tilings):
    # get features as indices into w
    idxs = [which_tile(s,width,offset,t) for t in range(tilings)]

    # v = sum of w at feature indices
    v = np.sum(w[range(tilings), idxs])
    return v, idxs

# to plot things for the figure
def plot_approximation():
    env = RandomWalkEnv()
    episodes = 10000
    width = 100
    tilings = 1
    offset = width//tilings
    alpha = 0.0002/tilings
    w_generator = grad_mc(episodes, alpha, env, width, 
            offset=offset, tilings=tilings)
    for episode in range(episodes): # we only need the last one
        w = next(w_generator)

    x = range(1,1001)
    y = [get_v(w,s,width,offset,tilings)[0] for s in x] 
    plt.plot(x, y, label="aggregation")

    # load and plot "true" values
    with open("data/true_values.npy", "rb") as f:
        y2 = pickle.load(f)
    plt.plot(x, y2, label="true values")

    plt.ylim([-1,1])
    plt.show()

#plot_approximation()
