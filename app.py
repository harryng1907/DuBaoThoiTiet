from flask import Flask, render_template, request
import pandas as pd
import joblib
import datetime
import os
import random
import requests

app = Flask(__name__)

# --- 1. CẤU HÌNH: TẢI MÔ HÌNH ---
print("⚡ Dang khoi dong he thong...")

try:
    model_columns = joblib.load("models/model_columns.joblib")
    # Tải 2 mô hình chuyên gia Max/Min (Global variables)
    model_max = joblib.load("models/model_max.joblib")
    model_min = joblib.load("models/model_min.joblib")
    print("✅ Da tai mo hinh Max & Min thanh cong.")
except Exception as e:
    print(f"❌ Loi: Khong tim thay file mo hinh ({e}). Hay chay script training truoc!")
    exit()

# Tọa độ các thành phố
city_coords = {
    'Hanoi': {'lat': 21.0285, 'lon': 105.8542},
    'Hue': {'lat': 16.4637, 'lon': 107.5909},
    'Da Nang': {'lat': 16.0544, 'lon': 108.2022},
    'Ho Chi Minh City': {'lat': 10.8231, 'lon': 106.6297},
    'Can Tho': {'lat': 10.0452, 'lon': 105.7469},
    'Da Lat': {'lat': 11.9404, 'lon': 108.4583},
    'Vinh': {'lat': 18.6733, 'lon': 105.6869}
}

# --- 2. HÀM LẤY DỮ LIỆU THẬT (LIVE API) ---
def get_live_weather(city_name):
    coords = city_coords.get(city_name)
    if not coords:
        return None, "Khong tim thay toa do thanh pho nay."

    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={coords['lat']}&longitude={coords['lon']}&current=temperature_2m,relative_humidity_2m,rain,surface_pressure,wind_speed_10m&daily=temperature_2m_max,temperature_2m_min&timezone=auto"
        
        response = requests.get(url)
        data = response.json()
        
        current = data['current']
        daily = data['daily']
        
        return {
            'mean': current['temperature_2m'],
            'max': daily['temperature_2m_max'][0], 
            'min': daily['temperature_2m_min'][0],
            'rain': current['rain'],
            'hum': current['relative_humidity_2m'],
            'press': current['surface_pressure']
        }, None
    except Exception as e:
        return None, f"Loi ket noi API: {str(e)}"

# --- 3. BỘ MÁY DỰ BÁO ---
# (Đã sửa: Bỏ tham số model_key thừa vì chúng ta dùng model_max/min toàn cục)
def calculate_forecast(start_data, city, start_date):
    results = []
    
    current_input = {
        'temperature_2m_mean': float(start_data['mean']),
        'temperature_2m_max': float(start_data['max']),
        'temperature_2m_min': float(start_data['min']),
        'precipitation_sum': float(start_data['rain']),
        'humidity_avg': float(start_data['hum']),
        'pressure_avg': float(start_data['press']),
        'Month': pd.to_datetime(start_date).month,
    }
    
    for col in model_columns:
        if 'city_' in col:
            current_input[col] = 1 if col == f"city_{city}" else 0

    input_df = pd.DataFrame([current_input])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    current_date_obj = pd.to_datetime(start_date)

    for i in range(1, 8):
        pred_max = model_max.predict(input_df)[0]
        pred_min = model_min.predict(input_df)[0]
        pred_mean = (pred_max + pred_min) / 2
        
        next_date = current_date_obj + datetime.timedelta(days=1)
        results.append({
            "date": next_date.strftime('%d-%m-%Y'),
            "max": round(pred_max, 1),
            "min": round(pred_min, 1),
            "mean": round(pred_mean, 1)
        })
        
        input_df['temperature_2m_mean'] = pred_mean
        input_df['temperature_2m_max'] = pred_max
        input_df['temperature_2m_min'] = pred_min
        input_df['Month'] = next_date.month
        
        current_date_obj = next_date
        
    return results, None

# --- 4. HÀM PHỤ TRỢ (HELPER) ---
# QUAN TRỌNG: Không đặt @app.route ở đây!
def run_dashboard_logic(city):
    # 1. Get Live Data
    today_str = datetime.date.today().strftime('%Y-%m-%d')
    live_data, err = get_live_weather(city)
    
    if err: return None, None, err
    
    # 2. Predict Forecast
    start_data = {
        'mean': live_data['mean'],
        'max': live_data['max'],
        'min': live_data['min'],
        'rain': live_data['rain'],
        'hum': live_data['hum'],
        'press': live_data['press']
    }
    
    # SỬA LỖI: Chỉ truyền 3 tham số (bỏ model_key)
    forecast, err = calculate_forecast(start_data, city, today_str)
    return live_data, forecast, err

# --- 5. GIAO DIỆN WEB (MAIN ROUTE) ---
@app.route('/', methods=['GET', 'POST'])
def index():
    # Mặc định
    city = 'Ho Chi Minh City'
    
    # Nếu người dùng chọn thành phố khác
    if request.method == 'POST':
        city = request.form.get('city')
    
    # Random Ad Image
    ad_images = ['monkey.gif','vietnam.png','mu.png','target.png'] # Đảm bảo file tồn tại trong static/
    random_ad = random.choice(ad_images)

    # AUTO-RUN: Chạy logic
    current_weather, forecast, error = run_dashboard_logic(city)
    
    # Tạo tên file ảnh thành phố
    city_img_name = city.lower().replace(" ", "") + ".jpg"
    print(city_img_name)

    nice_date = datetime.date.today().strftime('%d/%m/%Y')
    return render_template('index2.html', 
                           city=city, 
                           weather=current_weather, 
                           forecast=forecast, 
                           error=error,
                           city_image=city_img_name,
                           ad_image=random_ad,
                           date_display=nice_date)

if __name__ == '__main__':
    app.run(debug=True)