import numpy as np
import random
import matplotlib.pyplot as plt
from bandit_env import BanditEnv
from gradient_actor import GradientActor

runs = 2000
steps = 1000
num_actions = 10
rng = random.Random(1234)

for baseline in [False, True]:
    for alpha in [0.1, 0.4]:
        percent_correct_action = np.zeros(steps)

        for run in range(runs):
            env = BanditEnv(rng, num_actions, 4)
            actor = GradientActor(env, alpha, baseline)
            _, correct_actions = actor.run(steps)
            percent_correct_action += correct_actions

        plt.plot(range(steps), percent_correct_action/runs*100, label="baseline="+str(baseline)+", alpha="+str(alpha))
        plt.ylim([0, 100])
        plt.ylabel("% Optimal action")
        plt.xlabel("Steps")

plt.legend()
plt.show()
