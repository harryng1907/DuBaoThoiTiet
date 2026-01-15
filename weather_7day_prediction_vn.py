import pandas as pd
import joblib

# 1. TẢI MÔ HÌNH
print("⚡ Đang tải mô hình...")
try:
    model = joblib.load("weather_predictor.joblib")
except FileNotFoundError:
    print("❌ Lỗi: Không tìm thấy file 'weather_predictor.joblib'.")
    exit()

# 2. TẢI DỮ LIỆU
print("📂 Đang tải dữ liệu...")
try:
    # Thử file mới cập nhật trước, nếu không có thì dùng file cũ
    df = pd.read_csv("vietnam_weather_updated.csv")
except:
    df = pd.read_csv("vietnam_weather_final.csv")

# === SỬA LỖI 1: TÍNH TOÁN CỘT THÁNG (MONTH) ===
df['time'] = pd.to_datetime(df['time'])
df['Month'] = df['time'].dt.month

# === SỬA LỖI 2: CHUẨN HÓA TÊN THÀNH PHỐ (Bỏ dấu tiếng Việt) ===
# Bước này giúp đồng bộ "Huế" -> "Hue", "Cà Mau" -> "Ca Mau" để khớp với lúc huấn luyện.
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
print("✅ Đã chuẩn hóa tên thành phố (Đã bỏ dấu).")

# 3. ĐỊNH NGHĨA CÁC ĐẶC TRƯNG & CẤU TRÚC
features = [
    'temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min',
    'precipitation_sum', 'humidity_avg', 'pressure_avg', 'Month'
]

# --- BƯỚC AN TOÀN: TÁI TẠO CẤU TRÚC CỘT ---
# Tạo lại cấu trúc cột y hệt lúc huấn luyện để tránh lỗi thiếu cột thành phố
temp_df = df[features + ['city']].copy()
temp_X = pd.get_dummies(temp_df, columns=['city'], drop_first=True)
model_columns = temp_X.columns 
print(f"✅ Cấu trúc mô hình sẵn sàng: Mong đợi {len(model_columns)} cột đầu vào.")

# 4. HÀM DỰ BÁO
def predict_7_days_temp_only(city_name, start_date_str):
    start_date = pd.to_datetime(start_date_str)
    
    # Tự động sửa tên thành phố nếu người dùng nhập có dấu
    city_name = city_map.get(city_name, city_name) 

    # Lấy dữ liệu của ngày bắt đầu (Day 0)
    row = df[(df['city'] == city_name) & (df['time'] == start_date)]
    
    if row.empty:
        print(f"❌ Lỗi: Không tìm thấy dữ liệu cho {city_name} vào ngày {start_date_str}")
        print(f"   (Các thành phố có sẵn: {df['city'].unique()})")
        return

    # Chuẩn bị dữ liệu đầu vào
    current_input = row.copy()
    
    input_df = pd.get_dummies(current_input[features + ['city']], columns=['city'], drop_first=True)
    
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    print(f"\n🔮 Dự báo nhiệt độ 7 ngày tới tại {city_name}")
    print("="*45)
    print(f"{'Ngày':<12} | {'Nhiệt độ TB (°C)':<18}")
    print("-" * 45)

    current_date = start_date
    
    for i in range(1, 8):
        # 1. Dự báo nhiệt độ trung bình ngày mai
        pred_mean = model.predict(input_df)[0]
        
        # 2. In kết quả
        next_date = current_date + pd.Timedelta(days=1)
        print(f"{next_date.strftime('%Y-%m-%d'):<12} | {pred_mean:^18.2f}")
        
        # 3. Cập nhật đầu vào
        old_mean = input_df['temperature_2m_mean'].values[0]
        diff = pred_mean - old_mean
        
        # Cập nhật các chỉ số
        input_df['temperature_2m_mean'] = pred_mean
        input_df['temperature_2m_max'] += diff
        input_df['temperature_2m_min'] += diff
        input_df['Month'] = next_date.month
        
        current_date = next_date

# 5. CHẠY CHƯƠNG TRÌNH
try:
    c = input("Nhập tên thành phố (vd: Hue, Hanoi): ")
    d = input("Nhập ngày bắt đầu (YYYY-MM-DD): ") 
    predict_7_days_temp_only(c, d)
except Exception as e:
    print("Có lỗi xảy ra:", e)