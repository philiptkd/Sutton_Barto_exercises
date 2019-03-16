# value iteration. gambler's problem as in Example 4.3

import numpy as np
import matplotlib.pyplot as plt

# returns state-value function
def iterate(eps):
    V = np.zeros(101)
    Vnew = np.zeros(101)
    policy = np.zeros(101)

    while(True):
        maxDiff = 0
        for s in range(1,len(V)-1):
            tmp = V[s]
            # take max over all a
            maxExpect = 0
            for a in range(min(s, 100-s)+1):
                r = 1 if s+a == 100 else 0
                expect = (1-ph)*(0 + gamma*V[s-a])
                expect += ph*(r + gamma*V[s+a])
                if expect > maxExpect:
                    maxExpect = expect
                    policy[s] = a
            Vnew[s] = maxExpect
            maxDiff = max(maxDiff, np.abs(tmp - Vnew[s]))
        V = Vnew[:]
        if maxDiff < eps:
            break
        else:
            print(maxDiff)
    return V, policy

# constants
gamma = 1
ph = 0.4

# value iteration
V, policy = iterate(1e-4)

# show optimal value function and policy
plt.plot(V)
plt.show()
plt.scatter(range(101), policy)
plt.show()
