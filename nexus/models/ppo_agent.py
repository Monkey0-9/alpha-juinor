import torch
import torch.nn as nn
import torch.nn.functional as F

class PPOActorCritic(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64):
        super(PPOActorCritic, self).__init__()
        
        # Shared feature extractor
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor head (outputs continuous action mean in [-1, 1])
        self.actor_mean = nn.Linear(hidden_dim, 1)
        # We can omit log_std for inference, or keep it as a trainable parameter
        self.actor_log_std = nn.Parameter(torch.zeros(1))
        
        # Critic head (outputs value of state)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # Expect x: (batch_size, input_dim)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        action_mean = torch.tanh(self.actor_mean(x))
        state_value = self.critic(x)
        
        # For ONNX export, we just need the deterministic action (the mean)
        return action_mean, state_value

    def get_action(self, x):
        action_mean, _ = self.forward(x)
        std = self.actor_log_std.exp()
        dist = torch.distributions.Normal(action_mean, std)
        action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        return action, dist.log_prob(action)
