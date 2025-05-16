import random
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config
from environment import PricingEnv
from agent import PricingAgent

def train():
    # --- 固定随机种子以便复现 ---
    np.random.seed(7542)
    random.seed(7542)
    torch.manual_seed(7542)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(7521)

    # --- 初始化环境与 agent ---
    config = Config()
    env = PricingEnv()
    agent = PricingAgent(config)
    device = agent.device

    # 将网络搬到 GPU
    agent.policy_net.to(device)
    agent.target_net.to(device)

    # --- 训练超参数 ---
    num_episodes = 10000
    max_steps = config.MAX_STEPS

    # --- 记录变量 ---
    rewards_history = []
    avg_rewards = []
    reward_stds = []
    epsilons = []
    action_counts = np.zeros(config.ACTION_DIM, dtype=int)

    # 固定一个 state 用于后续策略稳定性 & Q 值分布测试
    test_state = env.reset()

    # --- 主训练循环 ---
    epsilon = config.EPS_START
    for ep in tqdm(range(num_episodes), desc="Training Episodes"):
        state = env.reset()
        total_reward = 0.0

        epsilons.append(epsilon)

        for step in range(max_steps):
            # ε-greedy 选动作
            if random.random() < epsilon:
                action = random.randrange(config.ACTION_DIM)
            else:
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_vals = agent.policy_net(state_t)
                action = int(q_vals.argmax(dim=1).item())

            action_counts[action] += 1

            next_state, reward, done, _ = env.step(action)
            agent.store_transition((state, action, reward, next_state, float(done)))
            agent.update_model()

            state = next_state
            total_reward += reward
            if done:
                break

        epsilon = max(config.EPS_END, epsilon * config.EPS_DECAY)

        # 记录本集回报
        rewards_history.append(total_reward)
        window = rewards_history[-100:]
        avg_rewards.append(np.mean(window))
        reward_stds.append(np.std(window))

        if (ep + 1) % 50 == 0:
            print(f"\n=== Episode {ep+1} ===")
            print(f"Time period: {env.time_period:02d}:00h | Avg@100={avg_rewards[-1]:.2f} | Std@100={reward_stds[-1]:.2f} | ε={epsilon:.3f}")
            test_state = env.reset()
            s_t = torch.FloatTensor(test_state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_vals = agent.policy_net(s_t)[0].cpu().numpy()
            print("固定状态 Q 值分布:")
            for a, q in enumerate(q_vals):
                print(f"  Action {a} ({a*5}%): {q:.1f}")

    # --- 保存模型 ---
    torch.save(agent.policy_net.state_dict(), "pricing_model.pth")

    # --- 绘图 & 保存图像 ---

    # 1) Reward & 滑动平均
    fig1 = plt.figure(figsize=(8, 4))
    plt.plot(rewards_history, alpha=0.3, label="Episode Reward")
    plt.plot(avg_rewards, label="100-episode Avg")
    plt.title("Reward & 100-episode Avg")
    plt.xlabel("Episode"); plt.ylabel("Reward")
    plt.legend()
    fig1.savefig("reward_trend.png")

    # 2) Reward 标准差
    fig2 = plt.figure(figsize=(8, 4))
    plt.plot(reward_stds, color="orange", label="Std Dev@100")
    plt.title("Reward Std Dev")
    plt.xlabel("Episode"); plt.ylabel("Std Dev")
    plt.legend()
    fig2.savefig("reward_std.png")

    # 3) ε 衰减曲线
    fig3 = plt.figure(figsize=(8, 4))
    plt.plot(epsilons, color="green", label="Epsilon")
    plt.title("Exploration Rate")
    plt.xlabel("Episode"); plt.ylabel("ε")
    plt.legend()
    fig3.savefig("epsilon_decay.png")

    # 4) 动作选择分布
    fig4 = plt.figure(figsize=(8, 4))
    plt.bar(np.arange(config.ACTION_DIM)*5, action_counts, width=4)
    plt.title("Action Selection Distribution")
    plt.xlabel("Discount (%)"); plt.ylabel("Count")
    fig4.savefig("action_distribution.png")

    # 5) 固定 state 的 Q 值分布
    s = torch.FloatTensor(test_state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_vals = agent.policy_net(s)[0].cpu().numpy()
    fig5 = plt.figure(figsize=(6, 4))
    plt.bar(np.arange(config.ACTION_DIM)*5, q_vals, width=4)
    plt.title("Q-values on Fixed State")
    plt.xlabel("Discount (%)"); plt.ylabel("Q-value")
    fig5.savefig("q_values_fixed_state.png")

    plt.close('all')  # 关闭所有 Figure

if __name__ == "__main__":
    train()
