import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor

# 1. LOAD THE NEW "SUPER DATASET"
print("⏳ Loading Complete Dataset...")
try:
    # We use the file you just created with the "fix_data.py" script
    df = pd.read_csv("vietnam_weather_full_filled.csv")
    print(f"   -> Loaded {len(df)} rows of data (2009-2025).")
except FileNotFoundError:
    print("❌ Error: 'vietnam_weather_full_filled.csv' not found.")
    print("   Please run the 'fix_data.py' script first!")
    exit()

# 2. PREPARE DATA
print("🔧 Preparing data...")
df['time'] = pd.to_datetime(df['time'])
df['Month'] = df['time'].dt.month

# Standardize City Names (Just to be safe)
city_map = {
    'Huế': 'Hue', 'Cà Mau': 'Ca Mau', 'Đà Nẵng': 'Da Nang', 
    'Đà Lạt': 'Da Lat', 'Hà Nội': 'Hanoi', 
    'TP. Hồ Chí Minh': 'Ho Chi Minh City', 'Hồ Chí Minh': 'Ho Chi Minh City'
}
df['city'] = df['city'].replace(city_map)

# Clean Numeric Columns (Ensure no text strings in numbers)
for col in ['precipitation_sum', 'wind_speed_10m_max', 'humidity_avg', 'pressure_avg']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# 3. CREATE TARGETS (Teach the AI to look at "Tomorrow")
# ---------------------------------------------------------
# We create 3 separate targets for our 3 experts
df['Target_Temp'] = df.groupby('city')['temperature_2m_mean'].shift(-1)
df['Target_Rain'] = df.groupby('city')['precipitation_sum'].shift(-1)
df['Target_Hum']  = df.groupby('city')['humidity_avg'].shift(-1)

# Drop the last row of each city (because it has no "Tomorrow")
df = df.dropna()

# 4. DEFINE FEATURES (Inputs)
# ---------------------------------------------------------
# Note: We use Humidity and Pressure now because your new file HAS them!
features = [
    'temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min',
    'precipitation_sum', 'humidity_avg', 'pressure_avg', 'Month'
]

# One-Hot Encode (Convert City to numbers)
X = pd.get_dummies(df[features + ['city']], columns=['city'], drop_first=True)

# Save the column list (CRITICAL for the App to work)
joblib.dump(X.columns, 'model_columns.joblib')
print("   -> Feature columns saved.")

# 5. TRAIN THE 3 EXPERTS
print("🚀 Training the AI Team...")

# --- Expert 1: Temperature ---
print("   -> Training Temperature Model...")
model_temp = GradientBoostingRegressor(n_estimators=100, random_state=42)
model_temp.fit(X, df['Target_Temp'])
joblib.dump(model_temp, 'model_temp.joblib')

# --- Expert 2: Rain ---
print("   -> Training Rain Model...")
model_rain = GradientBoostingRegressor(n_estimators=100, random_state=42)
model_rain.fit(X, df['Target_Rain'])
joblib.dump(model_rain, 'model_rain.joblib')

# --- Expert 3: Humidity ---
print("   -> Training Humidity Model...")
model_hum = GradientBoostingRegressor(n_estimators=100, random_state=42)
model_hum.fit(X, df['Target_Hum'])
joblib.dump(model_hum, 'model_hum.joblib')

print("✅ DONE! The AI Team is fully trained and saved.")