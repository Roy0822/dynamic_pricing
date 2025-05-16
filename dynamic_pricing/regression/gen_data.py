# data_generation.py
import numpy as np
import pandas as pd

def generate_pricing_data(n_samples=20000, seed=42, bonus_coef=1000.0):
    np.random.seed(seed)
    records = []

    # 预定义理想折扣（按业务需求设定）
    ideal_map = {
        11: 0.05, 12: 0.05, 13: 0.10,
        14: 0.20, 15: 0.30, 16: 0.30,
        17: 0.15, 18: 0.05, 19: 0.00,
        20: 0.10
    }
    max_discount = 0.95  # 折扣上限

    for _ in range(n_samples):
        # 随机特征
        hour     = np.random.randint(0, 24)
        is_peak   = 1 if hour in (11,12,13,17,18,19) else 0
        is_closed = 1 if hour < 10 or hour >= 21 else 0
        is_offpeak = 1 - (is_peak + is_closed)

        rating   = np.random.uniform(3.0, 5.0)
        price    = float(np.random.choice([80,100,120,150]))
        discount = float(np.random.choice(np.linspace(0, max_discount, 20)))
        adj_price = price * (1 - discount)

        # 基础需求（peak vs offpeak vs closed）
        if is_closed:
            base_demand = 0.0
        elif is_peak:
            base_demand = 100 + 30*(rating - 3.0)
        else:
            base_demand = 50  + 20*(rating - 3.0)

        # 非线性需求模型
        elasticity = 2.0 - discount
        ref_price  = 80.0
        demand = base_demand * np.exp(-elasticity*(adj_price - ref_price)/ref_price)
        demand += np.random.normal(0, 5)
        demand = float(np.clip(demand, 0, 200))

        # 利润（保证多数为正）
        cost   = price * 0.4
        profit = (adj_price - cost) * demand

        # 理想折扣 & 奖励项
        ideal_d = ideal_map.get(hour, 0.0)
        bonus   = bonus_coef * max(0.0, 1 - abs(discount - ideal_d) / max_discount)

        # 最终训练目标：profit_obj
        profit_obj = profit + bonus

        records.append([
            hour, is_peak, is_offpeak, is_closed,
            rating, price, discount, adj_price,
            demand, profit, profit_obj
        ])

    cols = [
        "hour","is_peak","is_offpeak","is_closed",
        "rating","price","discount","adjusted_price",
        "demand","profit","profit_obj"
    ]
    df = pd.DataFrame(records, columns=cols)
    df.to_csv("synthetic_pricing_data.csv", index=False)
    print("Saved synthetic_pricing_data_with_ideal.csv (with profit_obj)")

if __name__ == "__main__":
    generate_pricing_data()
