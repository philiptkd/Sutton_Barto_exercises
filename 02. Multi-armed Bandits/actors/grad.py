from actor import Actor
import numpy as np

class GradientActor(Actor):
    def __init__(self, env, alpha, use_baseline=True):
        super().__init__(env)
        self.alpha = alpha
        self.preferences = np.zeros(self.num_actions)
        self.policy = np.full(self.num_actions, 1/self.num_actions) # equally likely
        self.avg_reward = 0
        self.count = 0
        self.use_baseline = use_baseline
        
    def take_action(self):
        softmax_sample = self.rng.random()
        action = -1
        while softmax_sample > 0:
            action += 1
            softmax_sample -= self.policy[action]

        return action

    def update_stats(self, r, action):
        self.count += 1

        if self.use_baseline:
            self.avg_reward += (r-self.avg_reward)/self.count   #incremental calculation

        #Eq. 2.12
        self.preferences -= self.alpha*(r - self.avg_reward)*self.policy
        self.preferences[action] += self.alpha*(r - self.avg_reward)

        exp_policy = np.exp(self.preferences)
        self.policy = exp_policy/np.sum(exp_policy) #softmax
        
