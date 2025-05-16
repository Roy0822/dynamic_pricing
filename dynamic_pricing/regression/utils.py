# model_utils.py

import numpy as np
import joblib

MODEL_PATH = "xgb_profit_bonus_model.pkl"
_model = joblib.load(MODEL_PATH)

PEAK_HOURS    = set(range(11,14)) | set(range(17,20))
OFFPEAK_HOURS = set(range(10,21)) - PEAK_HOURS

def suggest_discount(hour: int, rating: float, price: float):
    is_peak    = 1 if hour in PEAK_HOURS else 0
    is_offpeak = 1 if hour in OFFPEAK_HOURS else 0

    best_d, best_obj = 0.0, -np.inf
    for a in range(20):
        d = a * 0.05
        x = np.array([[hour, rating, price, d, is_peak, is_offpeak]])
        val = _model.predict(x)[0]
        if val > best_obj:
            best_obj, best_d = val, d

    # 返回折扣与原始预计 profit（扣掉 bonus）
    # 如果需要真实 profit，可以再减回 bonus: best_obj - bonus_coef * best_d 
    return best_d, best_obj
