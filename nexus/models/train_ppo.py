import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Training PPO on device: {device}")

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=20, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, 3)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        features = self.net(x)
        action_probs = torch.softmax(self.actor(features), dim=-1)
        state_value = self.critic(features)
        return action_probs, state_value

class TradingEnvironment:
    def __init__(self, seq_len=100):
        self.seq_len = seq_len

    def generate_state(self, prices, position=0, idx=0):
        if idx < 20 or idx >= len(prices):
            return np.zeros(20, dtype=np.float32)
        window = prices[idx - 20:idx]
        returns = np.diff(window) / (window[:-1] + 1e-8)
        state = np.concatenate([
            [position / 100.0],
            returns[-5:],
            [np.mean(returns), np.std(returns)],
            [window[-1] / window[0] - 1],
            np.zeros(11)
        ])[:20]
        return state.astype(np.float32)

    def step(self, action, price_now, price_next, position):
        if action == 0:
            pass
        elif action == 1 and position < 100:
            position += 10
        elif action == 2 and position > -100:
            position -= 10
        pnl = position * (price_next - price_now)
        reward = np.tanh(pnl * 10)
        return position, reward

def train_ppo():
    np.random.seed(42)
    torch.manual_seed(42)

    prices = 100 + np.cumsum(np.random.randn(5000) * 0.5)
    prices = np.maximum(prices, 10)

    env = TradingEnvironment()
    policy = PolicyNetwork(state_dim=20).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.0003)

    clip_epsilon = 0.2
    gamma = 0.99
    gae_lambda = 0.95
    epochs = 10
    batch_size = 64

    for episode in range(500):
        states, actions, rewards, values, dones, log_probs = [], [], [], [], [], []
        position = 0
        idx = 50
        episode_reward = 0

        while idx < len(prices) - 2:
            state = env.generate_state(prices, position, idx)
            state_t = torch.tensor(state, device=device).unsqueeze(0)

            action_probs, value = policy(state_t)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            position, reward = env.step(action.item(), prices[idx], prices[idx + 1], position)

            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            values.append(value.item())
            dones.append(0)
            log_probs.append(log_prob.item())
            episode_reward += reward
            idx += 1

        if not states:
            continue

        returns = np.zeros(len(rewards))
        advantages = np.zeros(len(rewards))
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = 0
            else:
                next_val = values[t + 1]
            delta = rewards[t] + gamma * next_val - values[t]
            gae = delta + gamma * gae_lambda * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        states_arr = np.array(states)
        actions_arr = np.array(actions)
        old_log_probs_arr = np.array(log_probs)
        advantages_arr = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns_arr = np.array(returns)

        for _ in range(epochs):
            indices = np.arange(len(states_arr))
            np.random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch = indices[start:start + batch_size]
                if len(batch) < 2:
                    continue

                s_batch = torch.tensor(states_arr[batch], device=device)
                a_batch = torch.tensor(actions_arr[batch], device=device)
                old_ll_batch = torch.tensor(old_log_probs_arr[batch], device=device)
                adv_batch = torch.tensor(advantages_arr[batch], device=device)
                ret_batch = torch.tensor(returns_arr[batch], device=device)

                action_probs, values_pred = policy(s_batch)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(a_batch)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_ll_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv_batch
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.functional.mse_loss(values_pred.squeeze(), ret_batch)
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

        if episode % 50 == 0:
            logger.info(f"Episode {episode}: reward={episode_reward:.2f}")

    torch.save(policy.state_dict(), "nexus/models/ppo_trade_executor.pt")
    logger.info("PPO model saved")

    dummy_input = torch.randn(1, 20).to(device)
    torch.onnx.export(policy, dummy_input, "nexus/models/ppo_trade_executor.onnx",
                     input_names=["state"], output_names=["action_probs", "state_value"],
                     dynamic_axes={"state": {0: "batch"}},
                     opset_version=17)
    logger.info("PPO exported to ONNX")

if __name__ == "__main__":
    train_ppo()