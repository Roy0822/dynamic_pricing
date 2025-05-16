# test_regressor.py

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import suggest_discount  # 或者直接在此处 load 模型并实现 suggest_discount

def main():
    # 1) 读取模型
    model_path = "xgb_profit_bonus_model.pkl"
    print(f"Loading model from {model_path}...")
    # 如果你的 suggest_discount 已经封装好，这里直接导入即可
    # from model_utils import suggest_discount

    # 2) 定义营业时段与场景
    PEAK_HOURS    = set(range(11,14)) | set(range(17,20))
    OFFPEAK_HOURS = set(range(10,21)) - PEAK_HOURS

    ratings = [3.0, 4.0, 5.0]
    prices  = [80.0, 100.0, 120.0, 150.0]
    hours   = list(range(10,21))  # 10~20

    # 3) 收集推荐结果
    rows = []
    for rating in ratings:
        for price in prices:
            for hour in hours:
                d, p = suggest_discount(hour, rating, price)
                rows.append({
                    "hour": hour,
                    "rating": rating,
                    "price": price,
                    "discount_%": d*100,
                    "expected_obj": p
                })

    df = pd.DataFrame(rows)

    # 4) 打印一个示例小表
    print("\nSample recommendations:\n")
    print(df.head(10).to_string(index=False, float_format="{:.2f}".format))

    # 5) 绘图：不同评分下的平均折扣随小时变化
    plt.figure(figsize=(8,5))
    for rating in ratings:
        avg_ds = df[df["rating"]==rating].groupby("hour")["discount_%"].mean()
        plt.plot(avg_ds.index, avg_ds.values, marker='o', label=f"Rating={rating}")

    plt.xlabel("Hour of Day")
    plt.ylabel("Avg Recommended Discount (%)")
    plt.title("Avg Discount vs Hour\n(vary rating, avg over prices)")
    plt.xticks(hours)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("avg_discount_by_hour.png")
    print("Saved plot to avg_discount_by_hour.png")
    plt.show()

    # 6) 如果需要进一步分析，也可以画价格维度的折线图，或热力图等
    #    例如：固定 rating=4.0，画价格 vs 折扣 heatmap……

    # 7) 保存完整表格，供下游使用
    df.to_csv("all_discount_recommendations.csv", index=False)
    print("Saved full results to all_discount_recommendations.csv")

if __name__ == "__main__":
    main()
