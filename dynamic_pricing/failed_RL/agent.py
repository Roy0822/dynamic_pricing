import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class PricingAgent:
    def __init__(self, config):
        self.config = config
        # 自動選擇 CPU 或 GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 建立 policy 與 target 網路，並移到 device
        self.policy_net = DQN(config.STATE_DIM, config.ACTION_DIM).to(self.device)
        self.target_net = DQN(config.STATE_DIM, config.ACTION_DIM).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LR)
        self.memory = deque(maxlen=config.BUFFER_SIZE)
        self.steps = 0

    def select_action(self, state, epsilon):
        # ε-greedy 探索
        if random.random() < epsilon:
            return random.randint(0, self.config.ACTION_DIM - 1)
        # 利用模式：將 state 移到 device
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.policy_net(state_t)
        return int(q_vals.argmax(dim=1).item())

    def store_transition(self, transition):
        # transition = (state, action, reward, next_state, done)
        self.memory.append(transition)

    def update_model(self):
        if len(self.memory) < self.config.BATCH_SIZE:
            return

        # 隨機取一批
        batch = random.sample(self.memory, self.config.BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 轉成 tensor 並搬到 device
        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        # 計算目標 Q 值
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.config.GAMMA * next_q

        # 計算當前 Q 值
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 計算並反向傳播損失
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 定期更新 target network
        if self.steps % self.config.TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.steps += 1
