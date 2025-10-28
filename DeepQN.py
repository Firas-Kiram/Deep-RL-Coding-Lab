
# Traditional Q-Learning fails in high-dimensional environments.
# instead of maintaining a Q-table, we approximate the Q-value function with a neural network.


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import gymnasium as gym


class DeepQNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 64):
        super(DeepQNetwork, self).__init__()
        self.network(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DeepQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64, buffer_size=10000,
                 batch_size=64, gamma=0.99, lr=1e-3, epsilon_start=1.0,
                 epsilon_end=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        #Init network
        self.policy_net = DeepQNetwork(state_size, action_size, hidden_size)
        self.target_net = DeepQNetwork(state_size, action_size, hidden_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net.to(self.device)
        self.target_net.to(self.device)

    def select_action(self, state):
        #Epsilon-greedy action selection
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item()
                return action
        else:
            return random.randrange(self.action_size)
        
    def update_epsilon(self):
        #Decay epsilon value
        self.epsilon = max(self.epsilon_end , self.epsilon * self.epsilon_decay)
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q_value = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_value * (1 - dones))

        #Compute loss and update
        loss = nn.MSEloss()(current_q_values.squeeze(),target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())



def test_dqn():

    env = gym('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DeepQNAgent(state_size, action_size)
    assert agent.state_size == 4 , "State size should be 4 for CartPole"
    assert agent.action_size == 2 , "Action size should be 2 for CartPole"


    #Test replay buffer ,
    state = env.reset()[0]
    action = agent.select_action(state)
    next_state, reward, done, _, _ = env.step(action)
    agent.memory.push(state, action, reward, next_state, done)
    assert len(agent.memory) == 1 , "Replay buffer should have 1 transition"


    #Test epsilon Decay
    initial_epsilon = agent.epsilon
    agent.update_epsilon
    assert agent.epsilon < initial_epsilon , "Epsilon should decay after update"

    #Training loop for one episode 
    episode_reward = 0
    state = env.reset()[0]
    for t in range(100):
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.memory.push(state, action, reward, next_state, done)
        loss = agent.train_step()
        state = next_state
        episode_reward += reward
        
        if len(agent.memory) >= agent.batch_size:
            loss = agent.train_step
            assert isinstance(loss , float), "Training loss should be a float"
        
        if done:
            break

        print("All Test Cases passed ")
        return agent, env
    

if __name__ == "__main__":
    agent, env = test_dqn()

    #Train for multiple episode
    num_episodes = 100
    target_update_frequency = 10

    for episode in range(num_episodes):
        state = env.reset()[0]
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state = reward, done, _, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(agent.memory) >= agent.batch_size:
                loss = agent.train_step()


            if episode % target_update_frequency == 0:
                agent.update_target_network()

            agent.update_epsilon()
            print(f"Episode {episode + 1}, Total Reward: {episode_reward}, Epsilon: {agent.epsilon:.3f}")