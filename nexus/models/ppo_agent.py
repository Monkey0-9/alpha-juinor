import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class PPOActorCritic(nn.Module):
    """
    Institutional PPO Actor-Critic Reinforcement Learning Agent.
    Features:
      - Orthogonal initialization (gain=sqrt(2) for hidden, 0.01 for actor, 1.0 for critic)
      - Clipped Value Loss (L^CLIP+VF) & Target Network support
      - Generalized Advantage Estimation (GAE-lambda) calculation
      - Entropy Annealing for dynamic exploration control
    """
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, entropy_coef_start: float = 0.01):
        super().__init__()
        
        # Shared Feature Extractor
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Actor Head (continuous action mean in [-1, 1])
        self.actor_mean = nn.Linear(hidden_dim, 1)
        self.actor_log_std = nn.Parameter(torch.zeros(1))
        
        # Critic Head (state value V(s))
        self.critic = nn.Linear(hidden_dim, 1)

        # Entropy annealing state
        self.entropy_coef = entropy_coef_start
        
        # Apply Orthogonal Initialization
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization for deep RL stability."""
        for m in [self.fc1, self.fc2]:
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.0)
        
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)
        
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        h = F.gelu(self.norm1(self.fc1(x)))
        h = F.gelu(self.norm2(self.fc2(h)))
        
        action_mean = torch.tanh(self.actor_mean(h))
        state_value = self.critic(h)
        
        return action_mean, state_value

    def get_action(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean, value = self.forward(x)
        std = self.actor_log_std.exp()
        dist = torch.distributions.Normal(action_mean, std)
        action = dist.sample()
        action_clamped = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action_clamped)
        return action_clamped, log_prob, value

    def compute_gae(
        self, rewards: torch.Tensor, values: torch.Tensor, next_value: torch.Tensor,
        dones: torch.Tensor, gamma: float = 0.99, gae_lambda: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation (GAE-lambda)."""
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_val = values[t + 1]
                
            delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            
        returns = advantages + values
        return advantages, returns

    def compute_clipped_loss(
        self,
        old_log_probs: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
        clip_eps: float = 0.2,
        vf_clip_eps: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Clipped Policy surrogate loss + Clipped Value function loss."""
        action_mean, values = self.forward(states)
        std = self.actor_log_std.exp()
        dist = torch.distributions.Normal(action_mean, std)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # Policy Loss
        ratios = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value Function Loss with Clipping
        v_clipped = old_values + torch.clamp(values - old_values, -vf_clip_eps, vf_clip_eps)
        vf_loss1 = (values - returns).pow(2)
        vf_loss2 = (v_clipped - returns).pow(2)
        value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

        total_loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
        return total_loss, policy_loss, value_loss

    def anneal_entropy(self, decay_rate: float = 0.995, min_entropy: float = 0.001):
        """Anneal entropy coefficient per epoch for convergence."""
        self.entropy_coef = max(min_entropy, self.entropy_coef * decay_rate)
