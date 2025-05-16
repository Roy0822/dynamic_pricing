import numpy as np
import matplotlib.pyplot as plt
import joblib

# 加载训练好的 XGBoost 模型（profit_obj 版）
model = joblib.load("xgb_profit_bonus_model.pkl")

# 定义营业时段分类
PEAK_HOURS    = set(range(11,14)) | set(range(17,20))
OFFPEAK_HOURS = set(range(10,21)) - PEAK_HOURS

def is_peak(hour):
    return 1 if hour in PEAK_HOURS else 0

def is_offpeak(hour):
    return 1 if hour in OFFPEAK_HOURS else 0

# 要比较的不同时间点（小时）
times = [12, 15, 18]  # 12:00 peak, 15:00 offpeak, 18:00 peak

# 固定的其他特征
rating = 4.0
price  = 100.0

# 折扣点
discounts = np.linspace(0, 0.95, 20)

# 画图
plt.figure(figsize=(8, 5))
for h in times:
    profits = []
    for d in discounts:
        x = np.array([[h, rating, price, d, is_peak(h), is_offpeak(h)]])
        profits.append(model.predict(x)[0])
    plt.plot(discounts * 100, profits, label=f"{h}:00")

plt.xlabel("Discount (%)")
plt.ylabel("Predicted profit_obj")
plt.title("Discount vs. Expected Profit at Different Times")
plt.legend()
plt.tight_layout()

# 保存图像文件
plt.savefig("discount_curves_by_time.png")
plt.show()
