import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor

# 1. TẢI DỮ LIỆU
print("⏳ Dang tai du lieu...")
try:
    # Dùng file dữ liệu đầy đủ nhất bạn có
    df = pd.read_csv("resource/vietnam_weather_full_filled.csv") 
except:
    df = pd.read_csv("resource/vietnam_weather_final.csv")

df['time'] = pd.to_datetime(df['time'])
df['Month'] = df['time'].dt.month

# Chuẩn hóa tên thành phố
city_map = {
    'Huế': 'Hue', 'Cà Mau': 'Ca Mau', 'Đà Nẵng': 'Da Nang', 
    'Đà Lạt': 'Da Lat', 'Hà Nội': 'Hanoi', 
    'TP. Hồ Chí Minh': 'Ho Chi Minh City', 'Hồ Chí Minh': 'Ho Chi Minh City'
}
df['city'] = df['city'].replace(city_map)

# Xử lý số liệu
for col in ['precipitation_sum', 'humidity_avg', 'pressure_avg']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# 2. TẠO MỤC TIÊU (TARGETS)
# Chúng ta muốn dự báo Max và Min của NGÀY MAI
df['Target_Max'] = df.groupby('city')['temperature_2m_max'].shift(-1)
df['Target_Min'] = df.groupby('city')['temperature_2m_min'].shift(-1)
df = df.dropna()

# 3. ĐỊNH NGHĨA ĐẦU VÀO (FEATURES)
# Lưu ý: Cấu trúc cột phải khớp với những gì App sẽ gửi vào
features = [
    'temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min',
    'precipitation_sum', 'humidity_avg', 'pressure_avg', 'Month'
]

X = pd.get_dummies(df[features + ['city']], columns=['city'], drop_first=True)

# Lưu danh sách cột để App dùng
joblib.dump(X.columns, 'model_columns.joblib')

# 4. HUẤN LUYỆN 2 MÔ HÌNH RIÊNG BIỆT
print("🚀 Dang huan luyen...")

# Mô hình 1: Chuyên gia Max Temp
print("   -> Training Max Temp...")
model_max = GradientBoostingRegressor(n_estimators=100, random_state=42)
model_max.fit(X, df['Target_Max'])
joblib.dump(model_max, 'models/model_max.joblib') # Lưu vào thư mục models

# Mô hình 2: Chuyên gia Min Temp
print("   -> Training Min Temp...")
model_min = GradientBoostingRegressor(n_estimators=100, random_state=42)
model_min.fit(X, df['Target_Min'])
joblib.dump(model_min, 'models/model_min.joblib') # Lưu vào thư mục models

print("✅ XONG! Da co 2 file: model_max.joblib va model_min.joblib")