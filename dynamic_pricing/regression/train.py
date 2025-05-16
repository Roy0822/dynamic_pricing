# train_model.py

import pandas as pd
import joblib
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor

def main():
    df = pd.read_csv("synthetic_pricing_data.csv")

    feature_cols = ["hour","rating","price","discount","is_peak","is_offpeak"]
    target_col   = "profit_obj"   # 改成 profit_obj

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    use_gpu = torch.cuda.is_available()
    print("Use GPU:", use_gpu)

    params = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.1,
        "random_state": 42,
        "verbosity": 0
    }
    if use_gpu:
        params.update({
            "tree_method":"gpu_hist",
            "predictor":"gpu_predictor",
            "device":'cuda'
        })
    else:
        params.update({"tree_method":"hist"})
    
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"Test RMSE on profit_obj: {rmse:.2f}")

    joblib.dump(model, "xgb_profit_bonus_model.pkl")
    print("Saved xgb_profit_bonus_model.pkl")

if __name__=="__main__":
    main()
