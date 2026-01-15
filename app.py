from flask import Flask, render_template, request
import pandas as pd
import joblib
import datetime

app = Flask(__name__)

# --- 1. SETUP: LOAD MODEL & DATA ---
print("⚡ Loading AI Brain...")
try:
    model = joblib.load("weather_predictor.joblib")
    
    # Load Data
    try:
        df = pd.read_csv("vietnam_weather_updated.csv")
        print("   -> Loaded updated data.")
    except:
        df = pd.read_csv("vietnam_weather_final.csv")
        print("   -> Loaded original data.")
        
    df['time'] = pd.to_datetime(df['time'])
    df['Month'] = df['time'].dt.month

    # Standardize City Names
    city_map = {
        'Huế': 'Hue', 'Cà Mau': 'Ca Mau', 'Đà Nẵng': 'Da Nang', 
        'Đà Lạt': 'Da Lat', 'Hà Nội': 'Hanoi', 
        'TP. Hồ Chí Minh': 'Ho Chi Minh City', 'Hồ Chí Minh': 'Ho Chi Minh City'
    }
    df['city'] = df['city'].replace(city_map)
    
    # Features List
    features = ['temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min',
                'precipitation_sum', 'humidity_avg', 'pressure_avg', 'Month']
    
    # === THE FIX IS HERE ===
    # OLD BUGGY LINE: temp_df = df[features + ['city']].iloc[:5].copy()
    # NEW CORRECT LINE: We use the WHOLE dataframe so it learns ALL cities (Ca Mau, Hue, etc.)
    temp_df = df[features + ['city']].copy()
    # =======================
    
    temp_X = pd.get_dummies(temp_df, columns=['city'], drop_first=True)
    model_columns = temp_X.columns 
    print(f"✅ System Ready. Learned {len(model_columns)} feature columns.")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# --- 2. THE CORE PREDICTION ENGINE ---
def calculate_forecast(start_data, city, start_date):
    results = []
    
    # Prepare the initial input
    current_input = {
        'temperature_2m_mean': float(start_data['mean']),
        'temperature_2m_max': float(start_data['max']),
        'temperature_2m_min': float(start_data['min']),
        'precipitation_sum': float(start_data['rain']),
        'humidity_avg': float(start_data['hum']),
        'pressure_avg': float(start_data['press']),
        'Month': pd.to_datetime(start_date).month,
        'city': city
    }
    
    # Create DataFrame & One-Hot Encode
    input_df = pd.DataFrame([current_input])
    input_df = pd.get_dummies(input_df, columns=['city'], drop_first=True)
    
    # ALIGN COLUMNS: This fills in 0s for missing cities (e.g., city_Ca Mau = 0)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    current_date_obj = pd.to_datetime(start_date)

    # Loop for 7 days
    for i in range(1, 8):
        # Predict Tomorrow's Mean
        pred_mean = model.predict(input_df)[0]
        
        # Calculate Date
        next_date = current_date_obj + datetime.timedelta(days=1)
        
        # Store result
        results.append({
            "date": next_date.strftime('%Y-%m-%d'),
            "temp": round(pred_mean, 2)
        })
        
        # UPDATE INPUTS FOR NEXT LOOP
        old_mean = input_df['temperature_2m_mean'].values[0]
        diff = pred_mean - old_mean
        
        input_df['temperature_2m_mean'] = pred_mean
        input_df['temperature_2m_max'] += diff
        input_df['temperature_2m_min'] += diff
        input_df['Month'] = next_date.month
        
        current_date_obj = next_date
        
    return results

# --- 3. WEB ROUTES ---
@app.route('/', methods=['GET', 'POST'])
def index():
    # Default values
    form_data = {
        'city': 'Hanoi', 'date': '2025-01-01',
        'mean': '', 'max': '', 'min': '', 
        'rain': '', 'hum': '', 'press': ''
    }
    forecast = None
    message = "Welcome! Load data or enter values manually."

    if request.method == 'POST':
        city = request.form.get('city')
        date = request.form.get('date')
        action = request.form.get('action')

        form_data.update({
            'city': city, 'date': date,
            'mean': request.form.get('mean'),
            'max': request.form.get('max'),
            'min': request.form.get('min'),
            'rain': request.form.get('rain'),
            'hum': request.form.get('hum'),
            'press': request.form.get('press')
        })

        if action == 'load':
            target_date = pd.to_datetime(date)
            city_clean = city_map.get(city, city)
            
            row = df[(df['city'] == city_clean) & (df['time'] == target_date)]
            
            if not row.empty:
                form_data['mean'] = row.iloc[0]['temperature_2m_mean']
                form_data['max'] = row.iloc[0]['temperature_2m_max']
                form_data['min'] = row.iloc[0]['temperature_2m_min']
                form_data['rain'] = row.iloc[0]['precipitation_sum']
                form_data['hum'] = row.iloc[0]['humidity_avg']
                # Handle missing pressure
                press = row.iloc[0]['pressure_avg']
                form_data['press'] = press if pd.notnull(press) else 1010
                
                message = "✅ Data loaded! You can now predict."
            else:
                # Use defaults if date not found
                form_data.update({'mean': 25, 'max': 30, 'min': 20, 'rain': 0, 'hum': 80, 'press': 1010})
                message = "⚠️ Date not in history. Defaults loaded."

        elif action == 'predict':
            try:
                forecast = calculate_forecast(form_data, city, date)
                message = "🚀 Forecast generated successfully!"
            except Exception as e:
                message = f"❌ Error: {str(e)}"

    return render_template('index.html', form=form_data, forecast=forecast, message=message)

if __name__ == '__main__':
    app.run(debug=True)