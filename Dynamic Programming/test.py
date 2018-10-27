from grid_env import GridEnv
import random

env = GridEnv(random.Random(), 2, 2, [0], wind=[.1, -.1])
print(env.transition_probs)