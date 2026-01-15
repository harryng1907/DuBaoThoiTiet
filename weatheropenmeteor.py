import requests
import pandas as pd
import time

# 1. The Setup
# ---------------------------------------------------------
# Your selected Top Cities (I kept your list exactly as is)
top_cities = [ 
    "Hanoi", "Hai Phong", "Vinh", "Hue", "Nha Trang",
    "Da Lat", "Buon Ma Thuot", "Ho Chi Minh City", "Can Tho", "Ca Mau"
]

manual = { "Vinh": {"lat": 18.67337, "lon": 105.69232, "name" : "Vinh"}}

start_date = "2021-01-01"
end_date = "2026-01-14"

# CHANGE: We use "daily" instead of "hourly" to get 1 row per day
weather_params = {
    "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,rain_sum,wind_speed_10m_max",
    "timezone": "auto",
    "start_date": start_date,
    "end_date": end_date
}

all_data = []

print(f"Starting DAILY data collection for {len(top_cities)} cities...")

# 2. The Loop
# ---------------------------------------------------------
for city in top_cities:
    try:
        if city in manual:
            print(f"📍 Using Manual Override for {city}...")
            loc = manual[city]
            lat = loc["lat"]
            lon = loc["lon"]
            real_name = loc["name"]
        else:
        # STEP A: Geocoding
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=10&language=en&format=json"
            
            geo_response = requests.get(geo_url, timeout=10).json()

            if "results" not in geo_response:
                print(f"⚠️ Could not find location for: {city}")
                continue
                
            # Filter: Vietnam Only
            vietnam_results = [
                item for item in geo_response["results"] 
                if item.get("country_code") == "VN"
            ]

            if not vietnam_results:
                print(f"⚠️ Found '{city}' but not in Vietnam.")
                continue

            # Select: Highest Population
            best_match = sorted(vietnam_results, key=lambda x: x.get("population", 0), reverse=True)[0]
            
            real_name = best_match["name"]
            lat = best_match["latitude"]
            lon = best_match["longitude"]

            print(f"Found: {real_name}")

        # --- Step B: Get Weather Data ---
        weather_url = "https://archive-api.open-meteo.com/v1/archive"
        params = weather_params.copy()
        params["latitude"] = lat
        params["longitude"] = lon
        
        print(f"   ⏳ Requesting daily data for {real_name}...")
        response = requests.get(weather_url, params=params, timeout=60)
        
        if response.status_code != 200:
            print(f"❌ Error {response.status_code}: {response.text}")
            continue

        data_json = response.json()
        
        # CHANGE: We look for "daily" key now, not "hourly"
        if "daily" not in data_json:
            print(f"❌ No daily data found for {city}")
            continue

        daily_data = data_json["daily"]
        
        df_city = pd.DataFrame(daily_data)
        df_city["city"] = city             
        df_city["official_name"] = real_name
        
        all_data.append(df_city)
        
        time.sleep(1) # Shorter sleep is fine because data is smaller

    except Exception as e:
        print(f"❌ Critical Error with {city}: {e}")

# 3. Combine and Save
# ---------------------------------------------------------
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Save to CSV
    filename = "vietnam_weather_morerecent2.csv"
    final_df.to_csv(filename, index=False)
    print("------------------------------------------------")
    print(f"✅ Success! Saved {len(final_df)} rows to '{filename}'")
else:
    print("No data collected.")