# Reinforcement Learning Fundamentals - Coding Guide

**Time:** 20-25 hours
**Difficulty:** Intermediate to Advanced
**Prerequisites:** Python, NumPy, basic neural networks, PyTorch

## What You'll Build

Implement core RL algorithms from scratch:
1. Tabular Q-Learning (FrozenLake environment)
2. Deep Q-Network (DQN) for CartPole
3. Policy Gradients (REINFORCE algorithm)
4. Actor-Critic methods (A2C)
5. **Final Project:** Train agent to play Atari Pong or solve LunarLander

---

## Project Setup

```bash
mkdir rl-from-scratch
cd rl-from-scratch

# Create files
touch q_learning.py
touch dqn.py
touch policy_gradient.py
touch actor_critic.py
touch requirements.txt
```

### requirements.txt
```
numpy>=1.24.0
torch>=2.0.0
gymnasium>=0.29.0  # OpenAI Gym successor
matplotlib>=3.7.0
tqdm>=4.65.0
```

Install:
```bash
pip install -r requirements.txt
```

---

## Part 1: Tabular Q-Learning

### Theory: Markov Decision Processes (MDPs)

**Components:**
- **States (S)**: Environment configurations
- **Actions (A)**: Choices available to agent
- **Rewards (R)**: Feedback from environment
- **Transitions (P)**: State change probabilities
- **Discount factor (γ)**: Future reward importance (0-1)

**Goal:** Learn optimal policy π*(s) that maximizes cumulative reward

**Bellman Equation:**
```
Q(s, a) = R(s, a) + γ * max_a' Q(s', a')
```

**Q-Learning Update Rule:**
```
Q(s, a) ← Q(s, a) + α * [R + γ * max_a' Q(s', a') - Q(s, a)]
```

Where:
- α = learning rate
- γ = discount factor
- R = immediate reward
- s' = next state

### Implementation: Q-Learning for FrozenLake

```python
# q_learning.py
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm

class QLearningAgent:
    """Tabular Q-Learning Agent"""

    def __init__(self, n_states, n_actions, learning_rate=0.1,
                 discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01):
        """
        Args:
            n_states: Number of states in environment
            n_actions: Number of possible actions
            learning_rate: Alpha in Q-learning update
            discount_factor: Gamma (future reward importance)
            epsilon: Exploration rate (start high)
            epsilon_decay: Multiply epsilon after each episode
            epsilon_min: Minimum exploration rate
        """
        # Initialize Q-table with zeros
        self.q_table = np.zeros((n_states, n_actions))

        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.n_actions = n_actions

    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: best known action
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        """Q-learning update"""
        # Current Q-value
        current_q = self.q_table[state, action]

        # Best next Q-value (0 if terminal state)
        if done:
            max_next_q = 0
        else:
            max_next_q = np.max(self.q_table[next_state])

        # TD target: R + γ * max Q(s', a')
        target = reward + self.gamma * max_next_q

        # TD error
        td_error = target - current_q

        # Update Q-value
        self.q_table[state, action] += self.lr * td_error

    def decay_epsilon(self):
        """Reduce exploration over time"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_q_learning(env_name='FrozenLake-v1', n_episodes=10000):
    """Train Q-learning agent"""

    # Create environment
    env = gym.make(env_name, is_slippery=True)

    # Create agent
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.9995
    )

    # Track metrics
    episode_rewards = []
    success_rate = []
    epsilons = []

    print(f"Training Q-Learning on {env_name}...")

    for episode in tqdm(range(n_episodes)):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Choose action
            action = agent.choose_action(state)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Learn from experience
            agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        # Decay exploration
        agent.decay_epsilon()

        # Track metrics
        episode_rewards.append(total_reward)
        epsilons.append(agent.epsilon)

        # Calculate success rate (last 100 episodes)
        if episode >= 100:
            success_rate.append(np.mean(episode_rewards[-100:]))

    env.close()

    # Plot results
    plot_training_results(episode_rewards, success_rate, epsilons)

    return agent


def plot_training_results(rewards, success_rate, epsilons):
    """Visualize training progress"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Episode rewards
    axes[0].plot(rewards, alpha=0.3, label='Episode Reward')
    if len(success_rate) > 0:
        axes[0].plot(range(100, len(rewards)), success_rate,
                     linewidth=2, label='Success Rate (100-ep avg)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Training Progress')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Epsilon decay
    axes[1].plot(epsilons, linewidth=2)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Epsilon')
    axes[1].set_title('Exploration Rate Decay')
    axes[1].grid(True, alpha=0.3)

    # Success rate zoomed
    if len(success_rate) > 0:
        axes[2].plot(range(100, len(rewards)), success_rate, linewidth=2)
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Success Rate')
        axes[2].set_title('Success Rate (Smoothed)')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('q_learning_results.png', dpi=150)
    plt.show()


def test_agent(agent, env_name='FrozenLake-v1', n_episodes=100):
    """Test trained agent"""
    env = gym.make(env_name, is_slippery=True, render_mode='human')

    wins = 0
    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            # Greedy action (no exploration)
            action = np.argmax(agent.q_table[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if reward > 0:
                wins += 1

    env.close()
    print(f"Test Win Rate: {wins}/{n_episodes} = {100*wins/n_episodes:.1f}%")


if __name__ == "__main__":
    # Train agent
    agent = train_q_learning(n_episodes=10000)

    # Test agent
    test_agent(agent, n_episodes=100)

    # Print learned Q-table
    print("\nLearned Q-Table (first 10 states):")
    print(agent.q_table[:10])
```

### Run and Experiment

```bash
python q_learning.py
```

**Experiments:**
1. Try `is_slippery=False` (deterministic) - should converge faster
2. Adjust learning rate (0.01, 0.1, 0.5)
3. Adjust discount factor (0.9, 0.95, 0.99)
4. Change epsilon decay rate

---

## Part 2: Deep Q-Network (DQN)

### Theory: From Tables to Neural Networks

**Problem with Q-Tables:**
- Doesn't scale to large state spaces (images, continuous states)
- Can't generalize to unseen states

**Solution: Function Approximation**
- Use neural network to approximate Q(s, a)
- Input: state features
- Output: Q-value for each action

**Key Innovations in DQN:**
1. **Experience Replay**: Store transitions, sample randomly
2. **Target Network**: Separate network for stability
3. **Epsilon-Greedy**: Balance exploration/exploitation

### Implementation: DQN for CartPole

```python
# dqn.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

class QNetwork(nn.Module):
    """Neural network for Q-function approximation"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        """Forward pass: state -> Q-values for all actions"""
        return self.network(state)


class ReplayBuffer:
    """Experience replay buffer"""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store transition"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample random batch"""
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network Agent"""

    def __init__(self, state_dim, action_dim, learning_rate=1e-3,
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01, buffer_size=10000):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-network and target network
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size)

    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """Add experience to replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self, batch_size=64):
        """Train on batch from replay buffer"""
        if len(self.replay_buffer) < batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(batch_size)

        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Target Q-values (use target network)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Loss: MSE between current and target Q-values
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Reduce exploration"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_dqn(env_name='CartPole-v1', n_episodes=500,
              batch_size=64, target_update_freq=10):
    """Train DQN agent"""

    env = gym.make(env_name)

    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )

    episode_rewards = []
    losses = []

    print(f"Training DQN on {env_name}...")

    for episode in tqdm(range(n_episodes)):
        state, _ = env.reset()
        total_reward = 0
        done = False
        episode_losses = []

        while not done:
            # Choose action
            action = agent.choose_action(state)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Train
            loss = agent.train_step(batch_size)
            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            total_reward += reward

        # Update target network
        if episode % target_update_freq == 0:
            agent.update_target_network()

        # Decay epsilon
        agent.decay_epsilon()

        # Track metrics
        episode_rewards.append(total_reward)
        if episode_losses:
            losses.append(np.mean(episode_losses))

    env.close()

    # Plot results
    plot_dqn_results(episode_rewards, losses)

    return agent


def plot_dqn_results(rewards, losses):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Rewards
    axes[0].plot(rewards, alpha=0.3, label='Episode Reward')
    if len(rewards) >= 100:
        smoothed = [np.mean(rewards[max(0, i-100):i+1])
                   for i in range(len(rewards))]
        axes[0].plot(smoothed, linewidth=2, label='100-ep Average')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('DQN Training Progress')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    if losses:
        axes[1].plot(losses, linewidth=1.5)
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Training Loss')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dqn_results.png', dpi=150)
    plt.show()


def test_dqn_agent(agent, env_name='CartPole-v1', n_episodes=10):
    """Test trained DQN agent with rendering"""
    env = gym.make(env_name, render_mode='human')

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = agent.q_network(state_tensor).argmax().item()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Episode {episode + 1}: Reward = {total_reward}")

    env.close()


if __name__ == "__main__":
    # Train
    agent = train_dqn(n_episodes=500)

    # Test
    test_dqn_agent(agent, n_episodes=5)
```

### Run and Experiment

```bash
python dqn.py
```

**Experiments:**
1. Increase network size (256, 512 hidden units)
2. Add more layers
3. Try different learning rates (1e-4, 1e-3, 1e-2)
4. Adjust replay buffer size
5. Change target network update frequency

---

## Part 3: Policy Gradients (REINFORCE)

### Theory: Direct Policy Learning

**Q-Learning learns:** Q(s, a) → derive policy
**Policy Gradients learn:** π(a|s) directly

**Key Idea:**
- Policy π_θ(a|s) = probability of action a in state s
- Optimize policy parameters θ to maximize expected return
- Use gradient ascent on objective J(θ)

**REINFORCE Algorithm:**
1. Collect full episode trajectory
2. Calculate returns G_t for each timestep
3. Update policy: ∇_θ J(θ) ∝ Σ_t ∇_θ log π_θ(a_t|s_t) * G_t

### Implementation

```python
# policy_gradient.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm

class PolicyNetwork(nn.Module):
    """Policy network (actor)"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)  # Output action probabilities
        )

    def forward(self, state):
        return self.network(state)


class REINFORCEAgent:
    """REINFORCE (Monte Carlo Policy Gradient) Agent"""

    def __init__(self, state_dim, action_dim, learning_rate=1e-3, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma

        # Storage for episode
        self.states = []
        self.actions = []
        self.rewards = []

    def choose_action(self, state):
        """Sample action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action_probs = self.policy(state_tensor)

        # Sample from distribution
        action = torch.multinomial(action_probs, 1).item()

        return action

    def store_transition(self, state, action, reward):
        """Store experience"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def calculate_returns(self):
        """Calculate discounted returns G_t"""
        returns = []
        G = 0

        # Work backwards from terminal state
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        return torch.FloatTensor(returns)

    def update_policy(self):
        """REINFORCE policy update"""
        # Calculate returns
        returns = self.calculate_returns()

        # Normalize returns (reduces variance)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Convert to tensors
        states = torch.FloatTensor(self.states)
        actions = torch.LongTensor(self.actions)

        # Calculate policy loss
        action_probs = self.policy(states)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())

        # Policy gradient: -Σ log π(a|s) * G
        loss = -(log_probs * returns).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear episode memory
        self.states = []
        self.actions = []
        self.rewards = []

        return loss.item()


def train_reinforce(env_name='CartPole-v1', n_episodes=1000):
    """Train REINFORCE agent"""

    env = gym.make(env_name)

    agent = REINFORCEAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=1e-3,
        gamma=0.99
    )

    episode_rewards = []
    losses = []

    print(f"Training REINFORCE on {env_name}...")

    for episode in tqdm(range(n_episodes)):
        state, _ = env.reset()
        done = False
        total_reward = 0

        # Collect full episode
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward)

            state = next_state
            total_reward += reward

        # Update policy after episode
        loss = agent.update_policy()

        episode_rewards.append(total_reward)
        losses.append(loss)

    env.close()

    # Plot
    plot_training_curve(episode_rewards, losses, "REINFORCE")

    return agent


def plot_training_curve(rewards, losses, algorithm_name):
    """Plot training progress"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    axes[0].plot(rewards, alpha=0.3)
    if len(rewards) >= 100:
        smoothed = [np.mean(rewards[max(0, i-100):i+1])
                   for i in range(len(rewards))]
        axes[0].plot(smoothed, linewidth=2, label='100-ep avg')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title(f'{algorithm_name} Training')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(losses)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Policy Loss')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{algorithm_name.lower()}_results.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    agent = train_reinforce(n_episodes=1000)
```

---

## Part 4: Actor-Critic Methods

### Theory: Best of Both Worlds

**Combines:**
- **Actor**: Policy network π_θ(a|s)
- **Critic**: Value network V_φ(s)

**Advantage:**
- Lower variance than REINFORCE
- Faster learning

**Update Rules:**
- Critic: minimize TD error δ = r + γV(s') - V(s)
- Actor: update using δ as baseline

### Implementation

```python
# actor_critic.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm

class ActorCriticNetwork(nn.Module):
    """Shared network for actor and critic"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCriticNetwork, self).__init__()

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        shared_features = self.shared(state)
        action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_probs, state_value


class A2CAgent:
    """Advantage Actor-Critic Agent"""

    def __init__(self, state_dim, action_dim, learning_rate=1e-3, gamma=0.99):
        self.ac_network = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.ac_network.parameters(), lr=learning_rate)
        self.gamma = gamma

    def choose_action(self, state):
        """Sample action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action_probs, _ = self.ac_network(state_tensor)

        action = torch.multinomial(action_probs, 1).item()
        return action

    def update(self, state, action, reward, next_state, done):
        """Actor-Critic update"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

        # Get predictions
        action_probs, state_value = self.ac_network(state_tensor)

        with torch.no_grad():
            _, next_state_value = self.ac_network(next_state_tensor)

        # Calculate TD error (advantage)
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * next_state_value.item()

        advantage = td_target - state_value.item()

        # Critic loss (value function)
        critic_loss = nn.MSELoss()(state_value, torch.FloatTensor([td_target]))

        # Actor loss (policy gradient with advantage)
        log_prob = torch.log(action_probs[0, action])
        actor_loss = -log_prob * advantage

        # Total loss
        loss = actor_loss + critic_loss

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


def train_a2c(env_name='CartPole-v1', n_episodes=500):
    """Train A2C agent"""

    env = gym.make(env_name)

    agent = A2CAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=1e-3,
        gamma=0.99
    )

    episode_rewards = []
    losses = []

    print(f"Training A2C on {env_name}...")

    for episode in tqdm(range(n_episodes)):
        state, _ = env.reset()
        total_reward = 0
        done = False
        episode_losses = []

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Update after each step (online learning)
            loss = agent.update(state, action, reward, next_state, done)
            episode_losses.append(loss)

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        losses.append(np.mean(episode_losses))

    env.close()

    plot_training_curve(episode_rewards, losses, "A2C")

    return agent


def plot_training_curve(rewards, losses, algorithm_name):
    """Plot results"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    axes[0].plot(rewards, alpha=0.3)
    if len(rewards) >= 100:
        smoothed = [np.mean(rewards[max(0, i-100):i+1])
                   for i in range(len(rewards))]
        axes[0].plot(smoothed, linewidth=2)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title(f'{algorithm_name} Training')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(losses)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Loss')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{algorithm_name.lower()}_results.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    agent = train_a2c(n_episodes=500)
```

---

## Part 5: Advanced Projects

### Project 1: LunarLander

```python
# Train on LunarLander-v2
env = gym.make('LunarLander-v2')

# Use DQN or A2C
agent = DQNAgent(
    state_dim=8,  # LunarLander state space
    action_dim=4,  # 4 discrete actions
    learning_rate=5e-4,
    gamma=0.99
)

# Train for 1000+ episodes
# Goal: Achieve score > 200
```

### Project 2: Atari Pong (using CleanRL)

```bash
# Install Atari environments
pip install gymnasium[atari]
pip install gymnasium[accept-rom-license]

# Use CleanRL for optimized implementation
git clone https://github.com/vwxyzjn/cleanrl.git
cd cleanrl

# Train DQN on Pong
python cleanrl/dqn_atari.py --env-id PongNoFrameskip-v4
```

**Key Components for Atari:**
1. **Frame preprocessing**: Grayscale, resize to 84x84
2. **Frame stacking**: Stack 4 frames for motion
3. **Convolutional layers**: Process image input
4. **Longer training**: 10M+ steps

---

## Comparison: Q-Learning vs DQN vs Policy Gradients

| Algorithm | State Space | Action Space | Sample Efficiency | Stability |
|-----------|-------------|--------------|-------------------|-----------|
| Q-Learning | Small/Discrete | Discrete | High | High |
| DQN | Large/Continuous | Discrete | Medium | Medium |
| REINFORCE | Any | Discrete/Continuous | Low | Low |
| A2C | Any | Discrete/Continuous | Medium | Medium |

**When to use:**
- **Q-Learning**: Small discrete problems (GridWorld, FrozenLake)
- **DQN**: Moderate state spaces, discrete actions (CartPole, Atari)
- **Policy Gradients**: Continuous actions, stochastic policies
- **A2C**: Balance of efficiency and stability

---

## Key Concepts Summary

### 1. Exploration vs Exploitation
- **Epsilon-greedy**: Random action with probability ε
- **Temperature**: Softmax with temperature parameter
- **Entropy regularization**: Encourage exploration in policy gradients

### 2. Credit Assignment
- **Temporal Difference (TD)**: Update using bootstrapped estimate
- **Monte Carlo**: Update using full episode return
- **n-step**: Hybrid approach

### 3. Variance Reduction
- **Baseline**: Subtract value function from returns
- **Advantage**: A(s,a) = Q(s,a) - V(s)
- **Generalized Advantage Estimation (GAE)**

### 4. Off-Policy vs On-Policy
- **Off-policy** (DQN): Learn from any experience
- **On-policy** (REINFORCE, A2C): Learn from current policy

---

## Resources

### Books
- [Sutton & Barto: Reinforcement Learning (Free PDF)](http://incompleteideas.net/book/RLbook2020.pdf) - The RL bible
- [Spinning Up in Deep RL (OpenAI)](https://spinningup.openai.com/) - Practical guide

### Courses & Tutorials
- [David Silver's RL Course](https://www.davidsilver.uk/teaching/) - Foundational lectures
- [Deep RL Course (Hugging Face)](https://huggingface.co/learn/deep-rl-course/) - Hands-on tutorials
- [Karpathy: Deep RL - Pong from Pixels](https://karpathy.github.io/2016/05/31/rl/) - Excellent practical walkthrough of Policy Gradients

### Implementations
- [CleanRL](https://github.com/vwxyzjn/cleanrl) - Clean, single-file implementations
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - Production-ready algorithms

### Papers
- [DQN Paper](https://arxiv.org/abs/1312.5602) - Playing Atari with Deep RL
- [PPO Paper](https://arxiv.org/abs/1707.06347) - Proximal Policy Optimization
- [A3C Paper](https://arxiv.org/abs/1602.01783) - Asynchronous Actor-Critic

---

## Next Steps

1. **Implement all 4 algorithms** - Get hands-on experience
2. **Compare performance** - Same environment, different algorithms
3. **Tune hyperparameters** - Learning rate, network size, buffer size
4. **Try harder environments** - LunarLander, MountainCar, Atari
5. **Read classic papers** - DQN, A3C, PPO, SAC
6. **Explore modern RL** - RLHF, offline RL, model-based RL

---

**Congratulations!** You now understand the core algorithms of deep reinforcement learning. You can train agents to solve complex sequential decision problems, from games to robotics. 🎮🤖
