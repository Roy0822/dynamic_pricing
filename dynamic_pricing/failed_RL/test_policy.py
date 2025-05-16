# test_policy.py

import torch
import numpy as np
from environment import PricingEnv
from agent import PricingAgent
from config import Config

def test_policy(model_path: str):
    # 初始化
    config = Config()
    env = PricingEnv()
    agent = PricingAgent(config)
    agent.policy_net.load_state_dict(
        torch.load(model_path, map_location=agent.device)
    )
    agent.policy_net.eval()
    device = agent.device

    # 从环境获取官方时段定义
    peak_hours = []
    for start, end in env.operating_hours['peak']:
        peak_hours.extend(range(start, end))
    
    offpeak_hours = []
    for start, end in env.operating_hours['offpeak']:
        offpeak_hours.extend(range(start, end))

    closed_hours = []
    for start, end in env.operating_hours['closed']:
        closed_hours.extend(range(start, end))

    # 自定义测试场景 (覆盖所有时段类型)
    scenarios = [
        # Peak Hours
        {'time': 12, 'rating': 4.8, 'price': 110.0},
        {'time': 19, 'rating': 4.2, 'price': 95.0},
        
        # Offpeak Hours
        {'time': 11, 'rating': 4.5, 'price': 85.0},
        {'time': 15, 'rating': 3.8, 'price': 75.0},
        {'time': 20, 'rating': 4.0, 'price': 90.0},
        
        # Closed Hours
        {'time': 2,  'rating': 4.5, 'price': 100.0},
        {'time': 22, 'rating': 4.5, 'price': 100.0},
        
        # Edge Cases
        {'time': 10, 'rating': 5.0, 'price': 150.0},  # Offpeak起始
        {'time': 21, 'rating': 3.0, 'price': 60.0},   # Closed起始
    ]

    print("\n=== 手动场景测试 ===\n")
    for sc in scenarios:
        # 设置环境状态
        env.time_period = sc['time']
        env.rating = sc['rating']
        env.current_price = sc['price']
        env.price_history = [sc['price'], sc['price']*0.95]  # 模拟价格趋势
        
        # 生成状态向量
        state = env._get_state()
        
        # 模型推理
        with torch.no_grad():
            q_vals = agent.policy_net(
                torch.FloatTensor(state).unsqueeze(0).to(device)
            )[0].cpu().numpy()
        
        # 解析结果
        best_action = int(np.argmax(q_vals))
        best_discount = best_action * 5
        period_type = "尖峰" if sc['time'] in peak_hours else \
                     "离峰" if sc['time'] in offpeak_hours else "非营业"
        
        print(f"时间: {sc['time']:02d}:00 | 类型: {period_type}")
        print(f"评分: {sc['rating']:.1f}/5.0 | 当前价格: {sc['price']:.1f}")
        print(f"推荐折扣: {best_discount}%")
        print(f"Q值分布前3: {np.argsort(-q_vals)[:3]*5}%\n")

    # 全时段扫描测试
    print("\n=== 24小时全时段扫描 (评分=4.0, 价格=100) ===\n")
    discount_records = {h: [] for h in range(24)}
    
    for hour in range(24):
        # 设置环境
        env.time_period = hour
        env.rating = 4.0
        env.current_price = 100.0
        env.price_history = [100.0, 95.0]  # 模拟降价趋势
        
        # 获取状态
        state = env._get_state()
        
        # 模型推理
        with torch.no_grad():
            q_vals = agent.policy_net(
                torch.FloatTensor(state).unsqueeze(0).to(device)
            )[0].cpu().numpy()
        
        # 记录结果
        best_action = int(np.argmax(q_vals))
        discount_records[hour].append(best_action*5)

    # 打印时段统计
    print("\n=== 时段统计 ===")
    def print_stats(hours, name):
        discounts = [discount_records[h][0] for h in hours]
        valid_discounts = [d for h, d in zip(hours, discounts) if h not in closed_hours]
        print(f"{name}时段 ({len(valid_discounts)}小时):")
        print(f"  平均折扣: {np.mean(valid_discounts):.1f}%")
        print(f"  折扣分布: {np.unique(valid_discounts, return_counts=True)}\n")

    print_stats(peak_hours, "尖峰")
    print_stats(offpeak_hours, "离峰")
    
    # 非营业时段验证
    closed_discounts = [discount_records[h][0] for h in closed_hours]
    print(f"非营业时段 ({len(closed_hours)}小时):")
    print("  预期无意义折扣 (因不营业):", closed_discounts)

if __name__ == "__main__":
    test_policy("pricing_model.pth")