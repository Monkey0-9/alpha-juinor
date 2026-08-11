import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def orthogonal_init(module, gain=math.sqrt(2)):
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class PPOActorCritic(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=256, num_layers=3):
        super().__init__()
        layers = []
        dims = [input_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.GELU(),
            ])
        self.feature_net = nn.Sequential(*layers)
        self.actor_mean = nn.Linear(hidden_dim, 1)
        self.actor_log_std = nn.Parameter(torch.zeros(1))
        self.critic = nn.Linear(hidden_dim, 1)
        self.apply(orthogonal_init)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        features = self.feature_net(x)
        action_mean = torch.tanh(self.actor_mean(features))
        state_value = self.critic(features)
        return action_mean, state_value

    def get_action(self, x, deterministic=False):
        action_mean, state_value = self.forward(x)
        std = self.actor_log_std.exp()
        dist = torch.distributions.Normal(action_mean, std)
        if deterministic:
            action = action_mean
        else:
            action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action)
        return action, log_prob, state_value

    def evaluate_actions(self, x, actions):
        action_mean, state_value = self.forward(x)
        std = self.actor_log_std.exp()
        dist = torch.distributions.Normal(action_mean, std)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        return log_probs, state_value, entropy

class PPOAgent:
    def __init__(self, input_dim=5, hidden_dim=256, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5, ppo_epochs=10, batch_size=64, target_kl=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PPOActorCritic(input_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.target_kl = target_kl

    def compute_gae(self, rewards, values, dones):
        advantages = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = 0
            else:
                next_val = values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def update(self, states, actions, rewards, dones):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        with torch.no_grad():
            _, values, _ = self.model.get_action(states)
        advantages, returns = self.compute_gae(rewards, values.squeeze(), dones.squeeze())
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        with torch.no_grad():
            old_log_probs, _, _ = self.model.evaluate_actions(states, actions)
        total_loss = 0
        for _ in range(self.ppo_epochs):
            indices = torch.randperm(len(states))
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                log_probs, values_pred, entropy = self.model.evaluate_actions(states[batch], actions[batch])
                ratio = torch.exp(log_probs - old_log_probs[batch].detach())
                surr1 = ratio * advantages[batch].unsqueeze(1)
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[batch].unsqueeze(1)
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values_pred.squeeze(), returns[batch])
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                total_loss += loss.item()
                with torch.no_grad():
                    kl = (old_log_probs[batch].detach() - log_probs).mean()
                    if kl > self.target_kl * 1.5:
                        break
        self.scheduler.step()
        return total_loss / max(self.ppo_epochs * (len(states) // self.batch_size + 1), 1)