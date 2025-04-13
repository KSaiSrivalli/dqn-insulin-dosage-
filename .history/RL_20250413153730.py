import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
from matplotlib.animation import FuncAnimation
import imageio_ffmpeg

# Custom Medication Environment
class InsulinEnv(gym.Env):
    def __init__(self):
        super(InsulinEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=np.array([40, 0]), high=np.array([300, 3]), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)  # 0: none, 1: low, 2: medium, 3: high
        self.state = None
        self.step_count = 0

    def reset(self):
        self.state = np.array([random.uniform(150, 200), 0], dtype=np.float32)  # glucose level, last action
        self.step_count = 0
        return self.state

    def step(self, action):
        glucose, last_action = self.state

        # Simulate glucose response to insulin (simplified)
        insulin_effect = [0, 10, 20, 35][action]
        glucose += random.uniform(-5, 5)  # random body fluctuation
        glucose -= insulin_effect  # insulin effect lowers glucose

        # Clip glucose level to valid range
        glucose = max(40, min(glucose, 300))

        self.state = np.array([glucose, action], dtype=np.float32)
        self.step_count += 1

        # Reward logic
        if 80 <= glucose <= 120:
            reward = 10
        elif glucose < 70 or glucose > 180:
            reward = -20
        else:
            reward = -1

        done = self.step_count >= 50 or glucose < 50 or glucose > 250

        return self.state, reward, done, {}

    def render(self):
        pass

# DQN Network
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (torch.FloatTensor(states),
                torch.LongTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(next_states),
                torch.FloatTensor(dones))

    def __len__(self):
        return len(self.buffer)

# Training Setup
def train():
    env = InsulinEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    buffer = ReplayBuffer()

    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05
    update_target = 20

    rewards_history = []
    glucose_over_time = []

    for episode in range(200):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            glucose_over_time.append(state[0])

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.FloatTensor(state).unsqueeze(0))
                    action = q_values.argmax().item()

            next_state, reward, done, _ = env.step(action)
            buffer.push((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)

                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
                next_q_values = target_net(next_states).max(1)[0]
                targets = rewards + gamma * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, targets.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        rewards_history.append(total_reward)

        if episode % update_target == 0:
            target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        print(f"Episode {episode} — Reward: {total_reward:.2f} — Epsilon: {epsilon:.2f}")

    # Plotting
    plt.plot(rewards_history)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid()
    plt.savefig("reward_plot.png")
    plt.close()

    # Animation of glucose over time
    fig, ax = plt.subplots()
    ax.set_xlim(0, len(glucose_over_time))
    ax.set_ylim(40, 300)
    line, = ax.plot([], [], lw=2)

    def update(frame):
        line.set_data(range(frame), glucose_over_time[:frame])
        return line,

    ani = FuncAnimation(fig, update, frames=len(glucose_over_time), blit=True)
    ani.save("dqn_insulin_simulation.mp4", writer="ffmpeg")

if __name__ == "__main__":
    train()
