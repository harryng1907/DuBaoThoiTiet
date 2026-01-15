import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib


# 1. Tải dữ liệu
print("⏳ Đang tải dữ liệu...")
df = pd.read_csv("vietnam_weather_final.csv")

# 2. Làm sạch và Chuẩn bị dữ liệu
df['time'] = pd.to_datetime(df['time'])


cols_to_fix = ['rain_sum', 'wind_speed_10m_max']
for col in cols_to_fix:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.fillna(method='ffill')

# 3. Tạo Mục tiêu
df['Target_NextDay_Temp'] = df.groupby('city')['temperature_2m_mean'].shift(-1)
df = df.dropna()

df['Month'] = df['time'].dt.month

feature_cols = [
    'temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min',
    'precipitation_sum', 'humidity_avg', 'pressure_avg', 'Month'
]

# Chuyển đổi
X = pd.get_dummies(df[feature_cols + ['city']], columns=['city'], drop_first=True)
y = df['Target_NextDay_Temp']

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Đang huấn luyện trên {len(X_train)} dòng, Kiểm thử trên {len(X_test)} dòng.")
print("-" * 50)

# Định nghĩa Mô hình
models = {
    "Hồi quy Tuyến tính (Linear Regression)": LinearRegression(),
    "Hồi quy Ridge (Ridge Regression)": Ridge(),
    "Cây Quyết định (Decision Tree)": DecisionTreeRegressor(max_depth=10),
    "Rừng Ngẫu nhiên (Random Forest)": RandomForestRegressor(n_estimators=50, random_state=42),
    "Tăng cường Gradient (Gradient Boosting)": GradientBoostingRegressor(random_state=42)
}

# Vòng lặp Huấn luyện
results = []

for name, model in models.items():
    # Huấn luyện
    model.fit(X_train, y_train)
    
    # Dự đoán thử
    predictions = model.predict(X_test)
    
    # Đánh giá
    mae = mean_absolute_error(y_test, predictions) # Sai số bao nhiêu độ?
    r2 = r2_score(y_test, predictions)             # Độ chính xác tổng thể (0 đến 1)
    
    results.append([name, mae, r2])
    print(f"{name} đã xong.")

# 8. In Bảng Kết quả
# ---------------------------------------------------------
results_df = pd.DataFrame(results, columns=["Mô hình", "MAE", "R2"])
results_df = results_df.sort_values(by="MAE", ascending=True)

print("\n" + "="*60)
print("BẢNG XẾP HẠNG MÔ HÌNH ")
print("="*60)
print(results_df.to_string(index=False))

# Tìm mô hình tốt nhất dựa trên sai số thấp nhất (MAE)
best_result = min(results, key=lambda x: x[1]) 
best_model_name = best_result[0]
best_mae = best_result[1]

print(f"\nMô hình #1 là: {best_model_name}")
print(f"   Với sai số trung bình (MAE): {best_mae:.2f}°C")

best_model = models[best_model_name]
print("\n Huấn luyện mô hình")

best_model.fit(X, y)

filename = "weather_predictor.joblib"
joblib.dump(best_model, filename)

print("\nĐã Hoàn Thành")