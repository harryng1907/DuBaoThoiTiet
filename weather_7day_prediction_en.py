import pandas as pd
import joblib

# 1. LOAD MODEL
chosen = input("Chon mo hinh:")
match chosen:
    case "1":
        model_name = "Decision_Tree"
    case "2":
        model_name = "Gradient_Boosting"
    case "3":
        model_name = "Linear_Regression"
    case "4":
        model_name = "Random_Forest"
    case "5":
        model_name = "Ridge_Regression"


filename = f"models/model_{model_name}.joblib"
try:
    model = joblib.load(filename)
except FileNotFoundError:
    print(f"❌ Error: {filename} not found.")
    exit()

df = pd.read_csv("resource/vietnam_weather_final.csv")

df['time'] = pd.to_datetime(df['time'])
df['Month'] = df['time'].dt.month
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

features = [
    'temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min',
    'precipitation_sum', 'humidity_avg', 'pressure_avg', 'Month'
]

temp_df = df[features + ['city']].copy()
temp_X = pd.get_dummies(temp_df, columns=['city'], drop_first=True)
model_columns = temp_X.columns 

# 4. PREDICTION FUNCTION
def predict_7_days_temp_only(city_name, start_date_str):
    start_date = pd.to_datetime(start_date_str)
    city_name = city_map.get(city_name, city_name) # Clean input city 
    row = df[(df['city'] == city_name) & (df['time'] == start_date)] #Data Day 0
    current_input = row.copy()
    input_df = pd.get_dummies(current_input[features + ['city']], columns=['city'], drop_first=True)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    print(f"\nDu bao nhiet do cho 7 ngay tiep theo o {city_name}:")
    print("="*40)
    print(f"{'Ngay':<12} | {'Nhiet Do Trung Binh (°C)':<15}")
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
    c = input("Chon Thanh Pho (e.g., Hue, Hanoi): ")
    d = input("Chon Ngay (2009-2021): ") 
    predict_7_days_temp_only(c, d)
except Exception as e:
    print("Error:", e)