import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# 1. Load Data
# ---------------------------------------------------------
print("⏳ Loading data & Training 4 models (Max, Min, Rain, Hum)...")
df = pd.read_csv("vietnam_weather_final.csv")
df['time'] = pd.to_datetime(df['time'])

# Numeric fixes & Filling
for col in ['rain_sum', 'wind_speed_10m_max']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.ffill()

# Create Month feature
df['Month'] = df['time'].dt.month

# 2. Prepare Features & Targets
# ---------------------------------------------------------
# We now use Max and Min instead of Mean
feature_cols = ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'humidity_avg', 'Month']

# Create 4 Targets (Next Day's Max, Min, Rain, Humidity)
df['Target_Max']  = df.groupby('city')['temperature_2m_max'].shift(-1)
df['Target_Min']  = df.groupby('city')['temperature_2m_min'].shift(-1)
df['Target_Rain'] = df.groupby('city')['precipitation_sum'].shift(-1)
df['Target_Hum']  = df.groupby('city')['humidity_avg'].shift(-1)

df = df.dropna()

# One-Hot Encoding for City
X = pd.get_dummies(df[feature_cols + ['city']], columns=['city'], drop_first=True)
model_columns = X.columns 

# 3. Train the "Four Horsemen" Models
# ---------------------------------------------------------
model_max = joblib.load('model_max_temp.joblib')
model_min = joblib.load('model_min_temp.joblib')
model_rain = joblib.load('model_rain.joblib')
model_hum = joblib.load('model_hum.joblib')
# 4. Prediction Function (Recursive)
# ---------------------------------------------------------
def predict_next_7_days(city_name, start_date_str):
    start_date = pd.to_datetime(start_date_str)
    
    # Check if city/date exists
    row = df[(df['city'] == city_name) & (df['time'] == start_date)]
    
    if row.empty:
        print(f"❌ Error: No data found for {city_name} on {start_date_str}")
        return

    # Prepare first input
    current_input = pd.get_dummies(row[feature_cols + ['city']], columns=['city'], drop_first=True)
    current_input = current_input.reindex(columns=model_columns, fill_value=0)
    
    print(f"\n🔮 7-Day Forecast for {city_name} starting {start_date_str}")
    print("="*75)
    # New header with High/Low
    print(f"{'Date':<12} | {'High (°C)':<10} | {'Low (°C)':<10} | {'Rain (mm)':<10} | {'Hum (%)':<8} | {'Status'}")
    print("-" * 75)

    current_date = start_date
    
    for i in range(1, 8):
        # Predict all 4 variables
        pred_max  = model_max.predict(current_input)[0]
        pred_min  = model_min.predict(current_input)[0]
        pred_rain = max(0, model_rain.predict(current_input)[0])
        pred_hum  = min(100, max(0, model_hum.predict(current_input)[0]))

        # Simple Logic check: Max must be >= Min
        if pred_min > pred_max:
            pred_max, pred_min = pred_min, pred_max

        # Status logic
        status = "Sunny ☀️"
        if pred_rain > 5: status = "Rainy 🌧️"
        elif pred_rain > 0.5: status = "Drizzle 🌦️"
        elif pred_hum > 90: status = "Cloudy ☁️"

        # Print
        next_date = current_date + pd.Timedelta(days=1)
        print(f"{next_date.strftime('%Y-%m-%d'):<12} | {pred_max:^10.1f} | {pred_min:^10.1f} | {pred_rain:^10.1f} | {pred_hum:^8.1f} | {status}")

        # Update input for the next loop (Recursive Step)
        current_input['temperature_2m_max'] = pred_max
        current_input['temperature_2m_min'] = pred_min
        current_input['precipitation_sum'] = pred_rain
        current_input['humidity_avg'] = pred_hum
        current_input['Month'] = next_date.month
        
        current_date = next_date

# 5. User Input
# ---------------------------------------------------------
try:
    print("\n--- Enter Details to Predict ---")
    user_city = input("City (e.g., Hanoi): ").strip()
    user_date = input("Date (YYYY-MM-DD): ").strip()
    
    predict_next_7_days(user_city, user_date)

except Exception as e:
    print("Something went wrong:", e)