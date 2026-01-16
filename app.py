from flask import Flask, render_template, request
import pandas as pd
import joblib
import datetime
import os
import random

app = Flask(__name__)

# --- 1. SETUP: LOAD DATA & DEFINE COLUMNS ---
print("⚡ Loading Data & Configuring Columns...")
try:
    # Load Data (using your specific path)
    df = pd.read_csv("resource/vietnam_weather_final.csv")
    df['time'] = pd.to_datetime(df['time'])
    df['Month'] = df['time'].dt.month

    # Standardize Cities
    city_map = {
        'Huế': 'Hue', 'Cà Mau': 'Ca Mau', 'Đà Nẵng': 'Da Nang', 
        'Đà Lạt': 'Da Lat', 'Hà Nội': 'Hanoi', 
        'TP. Hồ Chí Minh': 'Ho Chi Minh City', 'Hồ Chí Minh': 'Ho Chi Minh City'
    }
    df['city'] = df['city'].replace(city_map)

    # Define Features (Exact same as your script)
    features = [
        'temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min',
        'precipitation_sum', 'humidity_avg', 'pressure_avg', 'Month'
    ]

    # LEARN COLUMN STRUCTURE (Crucial Step)
    # We simulate the training step to get the exact list of 16 columns (including cities)
    temp_df = df[features + ['city']].copy()
    temp_X = pd.get_dummies(temp_df, columns=['city'], drop_first=True)
    model_columns = temp_X.columns 
    print(f"✅ Data Ready. Model expects {len(model_columns)} columns.")

except Exception as e:
    print(f"❌ Critical Error loading data: {e}")
    exit()

# --- 2. SETUP: LOAD ALL MODELS ---
print("⚡ Loading Models...")
models_dict = {}
model_files = {
    "1": "Decision_Tree",
    "2": "Gradient_Boosting",
    "3": "Linear_Regression",
    "4": "Random_Forest",
    "5": "Ridge_Regression"
}

for key, name in model_files.items():
    path = f"models/model_{name}.joblib"
    if os.path.exists(path):
        models_dict[key] = joblib.load(path)
        print(f"   -> Loaded {name}")
    else:
        print(f"   ⚠️ Warning: {name} not found in 'models/' folder.")

# --- 3. PREDICTION ENGINE ---
def calculate_forecast(model_key, start_data, city, start_date):
    # 1. Select the correct model
    if model_key not in models_dict:
        return None, "Model file not found. Did you train it?"
    
    model = models_dict[model_key]

    results = []
    
    # 2. Prepare Initial Input
    # Note: We use the user's manual inputs for Rain/Hum/Pressure
    current_input = {
        'temperature_2m_mean': float(start_data['mean']),
        'temperature_2m_max': float(start_data['max']),
        'temperature_2m_min': float(start_data['min']),
        'precipitation_sum': float(start_data['rain']), # User Input
        'humidity_avg': float(start_data['hum']),       # User Input
        'pressure_avg': float(start_data['press']),     # User Input
        'Month': pd.to_datetime(start_date).month,
        'city': city
    }
    
    # Create DataFrame & Align Columns
    input_df = pd.DataFrame([current_input])
    input_df = pd.get_dummies(input_df, columns=['city'], drop_first=True)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    current_date_obj = pd.to_datetime(start_date)

    # 3. Prediction Loop (Your Exact Logic)
    for i in range(1, 8):
        # Predict Mean Temp
        pred_mean = model.predict(input_df)[0]
        
        # Calculate Date
        next_date = current_date_obj + datetime.timedelta(days=1)
        
        # Store Result
        results.append({
            "date": next_date.strftime('%Y-%m-%d'),
            "temp": round(pred_mean, 2)
        })
        
        # Update Inputs (Recursive Logic)
        old_mean = input_df['temperature_2m_mean'].values[0]
        diff = pred_mean - old_mean
        
        input_df['temperature_2m_mean'] = pred_mean
        input_df['temperature_2m_max'] += diff
        input_df['temperature_2m_min'] += diff
        input_df['Month'] = next_date.month
        
        # Note: Rain/Hum/Pressure stay constant based on user input (as per your script logic)
        
        current_date_obj = next_date
        
    return results, None

# --- 4. WEB ROUTES ---
@app.route('/', methods=['GET', 'POST'])
def index():
    # Defaults
    form_data = {
        'city': 'Hanoi', 'date': '2021-01-01', 'model': '2',
        'mean': '', 'max': '', 'min': '', 
        'rain': '0.0', 'hum': '80.0', 'press': '1010.0'
    }
    forecast = None
    error = None
    message = "Chọn Mô hình và số liệu."

    #load hinh anh
    bg_images = ['danang.jpg', 'hanoi.jpg', 'hue.jpg', 'saigon.jpg']
    selected_bg = random.choice(bg_images) # Pick one randomly

    
    if request.method == 'POST':
        action = request.form.get('action')
        
        # Capture all inputs
        form_data.update({
            'city': request.form.get('city'),
            'date': request.form.get('date'),
            'model': request.form.get('model'),
            'mean': request.form.get('mean'),
            'max': request.form.get('max'),
            'min': request.form.get('min'),
            'rain': request.form.get('rain'),
            'hum': request.form.get('hum'),
            'press': request.form.get('press')
        })

        # LOGIC: Load History
        if action == 'load':
            target_date = pd.to_datetime(form_data['date'])
            city_clean = city_map.get(form_data['city'], form_data['city'])
            
            row = df[(df['city'] == city_clean) & (df['time'] == target_date)]
            
            if not row.empty:
                form_data['mean'] = row.iloc[0]['temperature_2m_mean']
                form_data['max'] = row.iloc[0]['temperature_2m_max']
                form_data['min'] = row.iloc[0]['temperature_2m_min']
                form_data['rain'] = row.iloc[0]['precipitation_sum']
                form_data['hum'] = row.iloc[0]['humidity_avg']
                # Handle pressure if missing in CSV
                p = row.iloc[0]['pressure_avg']
                form_data['press'] = p if pd.notnull(p) else 1010.0
                message = "Đã tải được thông tin trong ngày"
            else:
                form_data.update({'mean': 25, 'max': 30, 'min': 20})
                message = "Không tìm được thông tin trong ngày. Đã tải mặc định"

        # LOGIC: Predict
        elif action == 'predict':
            try:
                forecast, err = calculate_forecast(
                    form_data['model'], form_data, 
                    form_data['city'], form_data['date']
                )
                if err: error = err
                else: message = f"Dự đoán thời tiết thành công bằng {model_files[form_data['model']]}!"
            except Exception as e:
                error = f"Error: {str(e)}"

    return render_template('index.html', form=form_data, forecast=forecast, 
                       message=message, error=error, 
                       bg_image=selected_bg) # ✅ Correct

if __name__ == '__main__':
    app.run(debug=True)