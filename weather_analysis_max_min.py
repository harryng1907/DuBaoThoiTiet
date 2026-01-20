import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. SETUP & DATA PREP
# ---------------------------------------------------------
print("⏳ Loading data...")
try:
    df = pd.read_csv("resource/vietnam_weather_full_filled.csv")
except:
    df = pd.read_csv("resource/vietnam_weather_final.csv")

df['time'] = pd.to_datetime(df['time'])
df['Month'] = df['time'].dt.month

# Standardize City Names
city_map = {
    'Huế': 'Hue', 'Cà Mau': 'Ca Mau', 'Đà Nẵng': 'Da Nang', 
    'Đà Lạt': 'Da Lat', 'Hà Nội': 'Hanoi', 
    'TP. Hồ Chí Minh': 'Ho Chi Minh City', 'Hồ Chí Minh': 'Ho Chi Minh City'
}
df['city'] = df['city'].replace(city_map)

# Numeric conversion
for col in ['rain_sum', 'wind_speed_10m_max', 'precipitation_sum', 'humidity_avg', 'pressure_avg']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# ---------------------------------------------------------
# 2. CREATE TARGETS
# ---------------------------------------------------------
print("🎯 Creating Targets for Next Day Max & Min...")
df['Target_NextDay_Max'] = df.groupby('city')['temperature_2m_max'].shift(-1)
df['Target_NextDay_Min'] = df.groupby('city')['temperature_2m_min'].shift(-1)
df = df.dropna(subset=['Target_NextDay_Max', 'Target_NextDay_Min'])

# Select Features
features = [
    'temperature_2m_max', 'temperature_2m_min',
    'precipitation_sum', 'humidity_avg', 'pressure_avg', 'Month'
]
X = pd.get_dummies(df[features + ['city']], columns=['city'], drop_first=True)

# Split X (Random State 42 ensures consistency)
X_train, X_test, _, _ = train_test_split(X, df['Target_NextDay_Max'], test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 3. ANALYSIS LOOP
# ---------------------------------------------------------
models = {} 

for model_type in ['max', 'min']:
    print(f"\n" + "="*40)
    print(f"📊 ANALYZING MODEL: {model_type.upper()}")
    print("="*40)

    # 1. Load Model
    filename = f"models/model_{model_type}.joblib"
    try:
        model = joblib.load(filename)
        models[model_type] = model 
        print(f"✅ Loaded {filename}")
    except FileNotFoundError:
        print(f"❌ Error: {filename} not found. Skipping...")
        continue

    # 2. Get Target
    target_col = f'Target_NextDay_{model_type.capitalize()}'
    y = df[target_col]
    
    # Split y with same random_state
    _, _, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Predict
    y_pred = model.predict(X_test)

    # ------------------- CHARTS -------------------
    
    # Chart 1: Feature Importance
    plt.figure(figsize=(10, 5))
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)[-10:]
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(X.columns)[sorted_idx])
    plt.title(f'Top Factors for Next Day {model_type.upper()}')
    plt.tight_layout()
    plt.show()

    # Chart 2: Correlation Matrix (Specific to this Target)
    plt.figure(figsize=(10, 8))
    # Combine features + the specific target column for correlation
    numeric_df = df[features + [target_col]].corr()
    sns.heatmap(numeric_df, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Correlation Matrix: Features vs Next Day {model_type.upper()}')
    plt.tight_layout()
    plt.show()

    # Chart 3: Scatter Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel(f'Actual {model_type.upper()}')
    plt.ylabel(f'Predicted {model_type.upper()}')
    plt.title(f'Accuracy Scatter: {model_type.upper()}')
    plt.show()

    # Chart 4: Confusion Matrix (5 Bins)
    # Define 5 bins suitable for Vietnam weather
    bins = [-np.inf, 18, 24, 28, 33, np.inf]
    labels = ['Very Cool (<18)', 'Cool (18-24)', 'Pleasant (24-28)', 'Hot (28-33)', 'Very Hot (>33)']

    y_test_cat = pd.cut(y_test, bins=bins, labels=labels)
    y_pred_cat = pd.cut(y_pred, bins=bins, labels=labels)

    # Generate Matrix
    cm = confusion_matrix(y_test_cat, y_pred_cat, labels=labels)
    
    # Display
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='d', xticks_rotation='vertical')
    plt.title(f'Confusion Matrix (5 Bins) - {model_type.upper()}')
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# 4. FINAL DUAL PREDICTION
# ---------------------------------------------------------
print("\n" + "="*40)
print("🤖 FINAL TEST CASE (Next Day Forecast)")
print("="*40)

if 'max' in models and 'min' in models:
    sample_idx = 0 
    sample_row = X_test.iloc[[sample_idx]]
    
    actual_max = df.loc[sample_row.index, 'Target_NextDay_Max'].values[0]
    actual_min = df.loc[sample_row.index, 'Target_NextDay_Min'].values[0]

    pred_max = models['max'].predict(sample_row)[0]
    pred_min = models['min'].predict(sample_row)[0]

    print(f"Input Data (Today):")
    print(f"  - Today Max: {sample_row['temperature_2m_max'].values[0]}°C")
    print(f"  - Today Min: {sample_row['temperature_2m_min'].values[0]}°C")
    print("-" * 20)
    print(f"🔮 PREDICTION FOR TOMORROW:")
    print(f"  - MAX Temp: {pred_max:.2f}°C  (Actual: {actual_max:.2f}) -> Diff: {abs(pred_max - actual_max):.2f}")
    print(f"  - MIN Temp: {pred_min:.2f}°C  (Actual: {actual_min:.2f}) -> Diff: {abs(pred_min - actual_min):.2f}")
    
    if pred_max > pred_min:
        print("\n✅ Logic Check Passed: Max is higher than Min.")
    else:
        print("\n❌ Logic Check Failed: Max is lower than Min!")
else:
    print("Could not run final test: One or both models missing.")