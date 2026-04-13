import torch
import torch.nn as nn
from torch.distributions import Normal

class PilotNet(nn.Module):
    def __init__(self, input_dim=11):
        super(PilotNet, self).__init__()
        self.common = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(128, 2),
            nn.Tanh()
        )
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.common(x)
        return self.actor(x), self.critic(x)

    def act(self, x, noise_std):
        action_mean, value = self.forward(x)
        std = torch.full_like(action_mean, noise_std)
        dist = Normal(action_mean, std)
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        return action.detach(), action_logprob.detach(), value.detach()

    def evaluate(self, x, action, noise_std):
        action_mean, value = self.forward(x)
        std = torch.full_like(action_mean, noise_std)
        dist = Normal(action_mean, std)
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        return action_logprobs, value, dist_entropy
