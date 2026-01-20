import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor

# 1. TẢI DỮ LIỆU
print("⏳ Loading data...")
try:
    # Use your latest file
    df = pd.read_csv("resource/vietnam_weather_full_filled.csv") 
except:
    df = pd.read_csv("resource/vietnam_weather_final.csv")

df['time'] = pd.to_datetime(df['time'])
df['Month'] = df['time'].dt.month

# Standardize City Names
city_map = {
    'Huế': 'Hue', 'Cà Mau': 'Ca Mau', 'Đà Nẵng': 'Da Nang', 
    'Đà Lạt': 'Da Lat', 'Hà Nội': 'Hanoi', 
    'TP. Hồ Chí Minh': 'Ho Chi Minh City', 'Hồ Chí Minh': 'Ho Chi Minh City'
}
df['city'] = df['city'].replace(city_map)

# Handle numeric columns
for col in ['precipitation_sum', 'humidity_avg', 'pressure_avg']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# ---------------------------------------------------------
# 2. CREATE TARGETS (NEW LOGIC)
# ---------------------------------------------------------
print("🎯 Creating Next Day Targets (Grouped by City)...")

# We group by city so "Tomorrow" for Hanoi doesn't grab data from "Ho Chi Minh City"
df['Target_NextDay_Max'] = df.groupby('city')['temperature_2m_max'].shift(-1)
df['Target_NextDay_Min'] = df.groupby('city')['temperature_2m_min'].shift(-1)

# Drop rows where we don't have a "Tomorrow" (the last date of each city)
df = df.dropna(subset=['Target_NextDay_Max', 'Target_NextDay_Min'])

# 3. DEFINE INPUT FEATURES
# Note: Removed 'mean' temp as it is redundant
features = [
    'temperature_2m_max', 'temperature_2m_min',
    'precipitation_sum', 'humidity_avg', 'pressure_avg', 'Month'
]

# One-Hot Encoding for City
X = pd.get_dummies(df[features + ['city']], columns=['city'], drop_first=True)

# Save column list for the App
joblib.dump(X.columns, 'model_columns.joblib')

# 4. TRAIN MODELS
print("🚀 Training Models...")

# Model 1: Predict Next Day MAX
print("   -> Training Max Temp Model...")
model_max = GradientBoostingRegressor(n_estimators=100, random_state=42)
model_max.fit(X, df['Target_NextDay_Max'])  # <--- Uses new column
joblib.dump(model_max, 'models/model_max.joblib')

# Model 2: Predict Next Day MIN
print("   -> Training Min Temp Model...")
model_min = GradientBoostingRegressor(n_estimators=100, random_state=42)
model_min.fit(X, df['Target_NextDay_Min'])  # <--- Uses new column
joblib.dump(model_min, 'models/model_min.joblib')

print("✅ DONE! Models saved.")

# ---------------------------------------------------------
# 5. FINAL CHECK
# ---------------------------------------------------------
print("\n--- 🔎 FINAL VERIFICATION ---")
sample_row = X.iloc[[0]] # Take first row

pred_max = model_max.predict(sample_row)[0]
pred_min = model_min.predict(sample_row)[0]

print(f"Test Input (Month {sample_row['Month'].values[0]})")
print(f"Predicted Next Day Max: {pred_max:.2f}°C")
print(f"Predicted Next Day Min: {pred_min:.2f}°C")

if pred_max > pred_min:
    print("✅ SUCCESS: The models are distinct (Max > Min).")
else:
    print("❌ ERROR: Max is lower than Min (Check data).")