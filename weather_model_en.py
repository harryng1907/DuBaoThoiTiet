import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
from sklearn.multioutput import MultiOutputRegressor

# Load Data
print("Loading data...")
df = pd.read_csv("vietnam_weather_final.csv")

# Data Cleaning & Prep
df['time'] = pd.to_datetime(df['time'])

# Fix columns that might be read as text (Object) instead of numbers
cols_to_fix = ['rain_sum', 'wind_speed_10m_max']
for col in cols_to_fix:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing values (if any) with 0 or mean
df = df.fillna(method='ffill')

# Create the "Target"
df['Target_NextDay_Temp'] = df.groupby('city')['temperature_2m_mean'].shift(-1)

df = df.dropna()

# Select Features vs Target
df['Month'] = df['time'].dt.month

feature_cols = [
    'temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min',
    'precipitation_sum', 'humidity_avg', 'pressure', 'Month'
]

# Convert 'city' to numbers
X = pd.get_dummies(df[feature_cols + ['city']], columns=['city'], drop_first=True)
y = df['Target_NextDay_Temp']

# Split Data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} rows, Testing on {len(X_test)} rows.")
print("-" * 50)

# Define the Models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Decision Tree": DecisionTreeRegressor(max_depth=10),
    "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = []

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    predictions = model.predict(X_test)
    
    # Evaluate
    mae = mean_absolute_error(y_test, predictions) 
    r2 = r2_score(y_test, predictions)           
    
    results.append([name, mae, r2])
    print(f"{name} finished.")

# Print Results Sorted by Accuracy
# ---------------------------------------------------------
results_df = pd.DataFrame(results, columns=["Model", "MAE (Lower is Better)", "R2 Score (Higher is Better)"])
results_df = results_df.sort_values(by="R2 Score (Higher is Better)", ascending=True)

print("\n" + "="*50)
print("FINAL SCOREBOARD")
print("="*50)
print(results_df.to_string(index=False))

best_result = min(results, key=lambda x: x[1]) 
best_model_name = best_result[0]
best_mae = best_result[1]
print(f"\nThe best model is  {best_model_name} with MAE: {best_mae:.2f}°C")


best_model = models[best_model_name]

super_model = MultiOutputRegressor(best_model(random_state=42))
super_model.fit(X, y)

joblib.dump(super_model,'weather_model.joblib')
joblib.dump(X.columns,'weather_columns.joblib')
print("done!")