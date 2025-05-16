import numpy as np

def preprocess_state(raw_data):
    """预处理实时数据为状态向量"""
    # raw_data应包含：当前小时、评分、历史销量
    hour_feature = raw_data['hour'] / 23.0
    rating_feature = raw_data['rating'] / 5.0
    demand_feature = np.mean(raw_data['sales_history'][-3:])/100.0
    return np.array([
        hour_feature,
        rating_feature,
        raw_data['price']/150.0,
        demand_feature,
        raw_data['rating'],
        np.tanh(len(raw_data['sales_history'])/10.0)
    ], dtype=np.float32)