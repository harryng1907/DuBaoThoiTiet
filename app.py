from flask import Flask, render_template, request
import pandas as pd
import joblib
import datetime
import os
import random
import requests

app = Flask(__name__)

# --- 1. CONFIGURATION: LOAD MODELS ---
print("⚡ Starting system...")

try:
    model_columns = joblib.load("models/model_columns.joblib")
    # Load the 2 expert models (Global variables)
    model_max = joblib.load("models/model_max.joblib")
    model_min = joblib.load("models/model_min.joblib")
    print("✅ Max & Min models loaded successfully.")
except Exception as e:
    print(f"❌ Error: Model files not found ({e}). Please run the training script first!")
    exit()

# City Coordinates
city_coords = {
    'Hanoi': {'lat': 21.0285, 'lon': 105.8542},
    'Hue': {'lat': 16.4637, 'lon': 107.5909},
    'Da Nang': {'lat': 16.0544, 'lon': 108.2022},
    'Ho Chi Minh City': {'lat': 10.8231, 'lon': 106.6297},
    'Can Tho': {'lat': 10.0452, 'lon': 105.7469},
    'Da Lat': {'lat': 11.9404, 'lon': 108.4583},
    'Vinh': {'lat': 18.6733, 'lon': 105.6869}
}

# --- 2. GET REAL DATA (LIVE API) ---
def get_live_weather(city_name):
    coords = city_coords.get(city_name)
    if not coords:
        return None, "Coordinates not found for this city."

    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={coords['lat']}&longitude={coords['lon']}&current=temperature_2m,relative_humidity_2m,rain,surface_pressure,wind_speed_10m&daily=temperature_2m_max,temperature_2m_min&timezone=auto"
        
        response = requests.get(url)
        data = response.json()
        
        current = data['current']
        daily = data['daily']
        
        return {
            'mean': current['temperature_2m'], # Used for display only
            'max': daily['temperature_2m_max'][0], 
            'min': daily['temperature_2m_min'][0],
            'rain': current['rain'],
            'hum': current['relative_humidity_2m'],
            'press': current['surface_pressure']
        }, None
    except Exception as e:
        return None, f"API Connection Error: {str(e)}"

# --- 3. FORECAST ENGINE ---
def calculate_forecast(start_data, city, start_date):
    results = []
    
    # 1. Prepare Input for Model
    current_input = {
        'temperature_2m_max': float(start_data['max']),
        'temperature_2m_min': float(start_data['min']),
        'precipitation_sum': float(start_data['rain']),
        'humidity_avg': float(start_data['hum']),
        'pressure_avg': float(start_data['press']),
        'Month': pd.to_datetime(start_date).month,
    }
    
    # Add City One-Hot Encoding
    for col in model_columns:
        if 'city_' in col:
            current_input[col] = 1 if col == f"city_{city}" else 0

    input_df = pd.DataFrame([current_input])
    
    # ENSURE columns match model exactly (removes any accidental extra columns)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    if 'temperature_2m_mean' in input_df.columns:
        input_df = input_df.drop(columns=['temperature_2m_mean'])
    
    current_date_obj = pd.to_datetime(start_date)

    # 2. Loop for 7 Days
    for i in range(1, 8):
        # Predict
        pred_max = model_max.predict(input_df)[0]
        pred_min = model_min.predict(input_df)[0]
        
        # Calculate Mean ONLY for display (do not feed back to model)
        pred_mean = (pred_max + pred_min) / 2
        
        next_date = current_date_obj + datetime.timedelta(days=1)
        results.append({
            "date": next_date.strftime('%d-%m-%Y'),
            "max": round(pred_max, 1),
            "min": round(pred_min, 1),
            "mean": round(pred_mean, 1)
        })
        
        # Update inputs for the NEXT loop iteration
        input_df['temperature_2m_max'] = pred_max
        input_df['temperature_2m_min'] = pred_min
        input_df['Month'] = next_date.month
        
        # ❌ DELETED LINE: input_df['temperature_2m_mean'] = pred_mean 
        # (We removed this because the model doesn't use it anymore)
        
        current_date_obj = next_date
        
    return results, None

# --- 4. HELPER FUNCTION ---
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
    
    forecast, err = calculate_forecast(start_data, city, today_str)
    return live_data, forecast, err

# --- 5. WEB ROUTES ---
@app.route('/', methods=['GET', 'POST'])
def index():
    city = 'Ho Chi Minh City'
    
    if request.method == 'POST':
        city = request.form.get('city')
    
    ad_images = ['monkey.gif','vietnam.png','mu.png','target.png','ad1.jpg', 'ad2.jpg']
    random_ad = random.choice(ad_images)

    current_weather, forecast, error = run_dashboard_logic(city)
    
    city_img_name = city.lower().replace(" ", "") + ".jpg"

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