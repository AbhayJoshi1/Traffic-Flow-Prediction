
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create folders
os.makedirs("graphs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("data/Metro_Interstate_Traffic_Volume.csv")
df["date_time"] = pd.to_datetime(df["date_time"])

# Extract time features
df["hour"] = df["date_time"].dt.hour
df["day_of_week"] = df["date_time"].dt.dayofweek
df["month"] = df["date_time"].dt.month
df["year"] = df["date_time"].dt.year

# Cyclical encoding
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

# Select features and target
X = df[[
    "temp", "rain_1h", "snow_1h", "clouds_all",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "month", "year", "weather_main"
]]

y = df["traffic_volume"]

# Encode categorical features
X = pd.get_dummies(X, columns=["weather_main"], dtype=int)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Final Linear Regression Results")
print("--------------------------------")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# Save trained model and feature columns
with open("models/traffic_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("models/feature_columns.pkl", "wb") as file:
    pickle.dump(X.columns.tolist(), file)

# Save actual vs predicted graph
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel("Actual Traffic Volume")
plt.ylabel("Predicted Traffic Volume")
plt.title("Actual vs Predicted Traffic Volume")
plt.tight_layout()
plt.savefig("graphs/final_actual_vs_predicted.png", dpi=300)
plt.show()
plt.close()

print("\nModel saved in models/traffic_model.pkl")
print("Feature columns saved in models/feature_columns.pkl")
