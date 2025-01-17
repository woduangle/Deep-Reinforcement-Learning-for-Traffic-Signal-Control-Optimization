import numpy as np


class OrnsteinUhlenbeckProcess:
    """docstring for OUNoise"""
    def __init__(self,action_dimension,mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self, state):
        i = 0
        # state = list(state[0].values())
        for lane in state:
            agent_state = lane
            dx = self.theta * (self.mu - agent_state) + self.sigma
            agent_state = agent_state + dx
            state[i] = agent_state
            i+=1
        return state

