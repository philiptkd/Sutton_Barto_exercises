from policies import RandPolicy
import numpy as np

class GridActor():
    def __init__(self, env):
        self.env = env
        self.num_actions = env.num_actions
        self.rng = env.rng
        self.policy = RandPolicy(self.num_actions, self.rng)
        self.V = self.rng.uniform(-25, -5, env.num_states)
        for state in env.terminal_states:
            self.V[state] = 0.
        
    # evaluate the current policy and return the state-value function
    # done in-place, asynchronously, and in an arbitrary order
    def evaluate(self, policy, eps):
        # iterate
        while(True):
            maxDiff = 0
            for state in range(len(self.V)):
                tmp = self.V[state]
                self.V[state] = getExpectation(V, state, policy)
                maxDiff = max(maxDiff, np.abs(tmp - V[state]))
            if maxDiff < eps:
                break

        return V.reshape((4,4))

    # improves the policy used to create V by finding the greedy policy
    def improve(V):
        dt = np.dtype("i4,f4,f4")
        policy = np.full((4, 4, 4), 0, dtype=dt)
        
        for h in range(policy.shape[0]):
            for w in range(policy.shape[1]):
                a = getMaxDir(V,h,w)
                ns = nextState(h,w,a)
                if get_state(h,w) == 0 or get_state(h,w) == 15:
                    policy[h,w,a] = (ns, 0, 1)
                else:
                    policy[h,w,a] = (ns, -1, 1)

        return policy

    V = evaluate(0.01, build_rand_policy())
    print(V); print(improve(V))
