
import pickle
import numpy as np
import pandas as pd

# Load the saved model
with open("models/traffic_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("models/feature_columns.pkl", "rb") as file:
    feature_columns = pickle.load(file)

# Get input from user
print("TRAFFIC FLOW PREDICTION")
print("-----------------------")

date_time = pd.to_datetime(
    input("Enter date and time (YYYY-MM-DD HH:MM): ")
)

temp = float(input("Enter temperature in Kelvin (e.g. 288.15): "))
rain = float(input("Enter rainfall in mm (e.g. 0): "))
snow = float(input("Enter snowfall in mm (e.g. 0): "))
clouds = int(input("Enter cloud coverage (0-100): "))

weather = input(
    "Enter weather (Clear, Clouds, Rain, Drizzle, Mist, Snow, etc.): "
).strip()

# Extract time features
hour = date_time.hour
day_of_week = date_time.dayofweek
month = date_time.month
year = date_time.year

# Cyclical encoding
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
day_sin = np.sin(2 * np.pi * day_of_week / 7)
day_cos = np.cos(2 * np.pi * day_of_week / 7)

# Prepare input features
input_data = pd.DataFrame([{
    "temp": temp,
    "rain_1h": rain,
    "snow_1h": snow,
    "clouds_all": clouds,
    "hour_sin": hour_sin,
    "hour_cos": hour_cos,
    "day_sin": day_sin,
    "day_cos": day_cos,
    "month": month,
    "year": year,
    "weather_main": weather
}])

# Encode weather and align columns with training data
input_data = pd.get_dummies(
    input_data, columns=["weather_main"], dtype=int
)
input_data = input_data.reindex(
    columns=feature_columns, fill_value=0
)

# Predict traffic volume
prediction = model.predict(input_data)[0]

# Traffic volume cannot be negative
prediction = max(0, prediction)

print("\n-----------------------")
print(f"Predicted traffic volume: {prediction:.0f} vehicles per hour")
print("-----------------------")
