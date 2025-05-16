import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Uncomment the one you want to use
#import lightgbm as lgb
import xgboost as xgb

# === Load CSV File ===
csv_path = "data.csv"  
df = pd.read_csv(csv_path, header = 0)

for cat in ["Location_Category", "Customer_Loyalty_Status", "Vehicle_Type", "Time_of_Booking"]:
    df[cat] = df[cat].astype("category")
X = df.iloc[:, :9]           # First 9 columns are features
y = df["Historical_Cost_of_Ride"]                  # Target column

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === LightGBM ===
#model = lgb.LGBMRegressor(
#    objective='regression',
#    n_estimators=100,
#    learning_rate=0.1
#)

# === XGBoost (Alternative) ===
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    enable_categorical=True,
    tree_method="hist"
)

# === Train Model ===
model.fit(X_train, y_train)

# === Predict and Evaluate ===
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"RMSE on test set: {rmse:.2f}")

# === Plot: Actual vs. Predicted ===
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs. Predicted Price")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.grid(True)
plt.tight_layout()
plt.savefig("prediction_plot.png", dpi=300, bbox_inches='tight')  # Save as PNG
#plt.show()

# === Plot: Feature Importance ===
plt.figure(figsize=(10, 6))
importances = model.feature_importances_
sns.barplot(x=importances, y=X.columns)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature.png", dpi=300, bbox_inches='tight')  # Save as PNG
#plt.show()
