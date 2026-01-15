import pandas as pd
import joblib

# 1. LOAD MODEL
print("⚡ Loading model...")
try:
    model = joblib.load("weather_predictor.joblib")
except FileNotFoundError:
    print("❌ Error: 'weather_predictor.joblib' not found.")
    exit()

# 2. LOAD DATA
print("📂 Loading data...")
try:
    # Try the updated file first, fallback to original
    df = pd.read_csv("vietnam_weather_updated.csv")
except:
    df = pd.read_csv("vietnam_weather_final.csv")

# === FIX 1: CALCULATE MONTH ===
df['time'] = pd.to_datetime(df['time'])
df['Month'] = df['time'].dt.month

# === FIX 2: STANDARDIZE CITY NAMES (Removes Accents) ===
# This aligns "Huế" -> "Hue", "Cà Mau" -> "Ca Mau" to match the trained model.
city_map = {
    'Huế': 'Hue', 
    'Cà Mau': 'Ca Mau', 
    'Đà Nẵng': 'Da Nang', 
    'Đà Lạt': 'Da Lat',
    'Hà Nội': 'Hanoi',
    'TP. Hồ Chí Minh': 'Ho Chi Minh City',
    'Hồ Chí Minh': 'Ho Chi Minh City'
}
df['city'] = df['city'].replace(city_map)
print("✅ City names cleaned (Accents removed).")

# 3. DEFINE FEATURES & STRUCTURE
features = [
    'temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min',
    'precipitation_sum', 'humidity_avg', 'pressure_avg', 'Month'
]

# --- SAFETY STEP: REBUILD COLUMN STRUCTURE ---
# Now that city names are fixed, this will generate "city_Hue" (Correct) instead of "city_Huế" (Wrong)
temp_df = df[features + ['city']].copy()
temp_X = pd.get_dummies(temp_df, columns=['city'], drop_first=True)
model_columns = temp_X.columns 
print(f"✅ Model Structure Ready: Expecting {len(model_columns)} columns.")

# 4. PREDICTION FUNCTION
def predict_7_days_temp_only(city_name, start_date_str):
    start_date = pd.to_datetime(start_date_str)
    
    # Clean the user input city too
    city_name = city_map.get(city_name, city_name) 

    # Get Data for Day 0
    row = df[(df['city'] == city_name) & (df['time'] == start_date)]
    
    if row.empty:
        print(f"❌ Error: No data found for {city_name} on {start_date_str}")
        print(f"   (Available cities: {df['city'].unique()})")
        return

    # Prepare Input
    current_input = row.copy()
    
    # One-Hot Encode
    input_df = pd.get_dummies(current_input[features + ['city']], columns=['city'], drop_first=True)
    
    # ALIGN COLUMNS (The final lock)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    print(f"\n7-Day Temperature Forecast for {city_name}")
    print("="*40)
    print(f"{'Date':<12} | {'Avg Temp (°C)':<15}")
    print("-" * 40)

    current_date = start_date
    
    for i in range(1, 8):
        # Predict
        pred_mean = model.predict(input_df)[0]
        
        # Print
        next_date = current_date + pd.Timedelta(days=1)
        print(f"{next_date.strftime('%Y-%m-%d'):<12} | {pred_mean:^15.2f}")
        
        # Update Inputs
        old_mean = input_df['temperature_2m_mean'].values[0]
        diff = pred_mean - old_mean
        
        input_df['temperature_2m_mean'] = pred_mean
        input_df['temperature_2m_max'] += diff
        input_df['temperature_2m_min'] += diff
        input_df['Month'] = next_date.month
        
        current_date = next_date

# 5. RUN
try:
    c = input("City (e.g., Hue, Hanoi): ")
    d = input("Date (2009-2021): ") 
    predict_7_days_temp_only(c, d)
except Exception as e:
    print("Error:", e)