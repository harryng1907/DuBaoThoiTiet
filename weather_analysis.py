import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Load & Prepare Data
# ---------------------------------------------------------
print("⏳ Loading data and training model...")
df = pd.read_csv("vietnam_weather_final.csv")
df['time'] = pd.to_datetime(df['time'])

# Numeric conversion
for col in ['rain_sum', 'wind_speed_10m_max']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.fillna(method='ffill')

# Create Target (Next Day Temp)
df['Target_NextDay_Temp'] = df.groupby('city')['temperature_2m_mean'].shift(-1)
df = df.dropna()

# Select Features
df['Month'] = df['time'].dt.month
feature_cols = [
    'temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min',
    'precipitation_sum', 'humidity_avg', 'pressure_avg', 'Month'
]
X = pd.get_dummies(df[feature_cols + ['city']], columns=['city'], drop_first=True)
y = df['Target_NextDay_Temp']

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = joblib.load("weather_predictor.joblib")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("✅ Model trained. Generating charts...")

# ---------------------------------------------------------
# CHART 1: Feature Importance (What matters most?)
# ---------------------------------------------------------
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)[-10:] # Top 10 features

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X.columns)[sorted_idx])
plt.title('Top 10 Factors Affecting Weather Prediction')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# CHART 2: Correlation Heatmap (Relationships)
# ---------------------------------------------------------
plt.figure(figsize=(10, 8))
numeric_df = df[feature_cols + ['Target_NextDay_Temp']]
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# CHART 3: Scatter Plot (Accuracy Check)
# ---------------------------------------------------------
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.3, color='blue', label='Prediction')
# Draw a perfect red line (If dots are on this line, prediction is perfect)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Match')
plt.xlabel('Actual Temperature')
plt.ylabel('Predicted Temperature')
plt.title('Actual vs Predicted Temperature')
plt.legend()
plt.show()

# ---------------------------------------------------------
# CHART 4: Confusion Matrix (Categorized)
# ---------------------------------------------------------
# We convert numbers to categories: <20 (Cool), 20-28 (Pleasant), >28 (Hot)
bins = [-np.inf, 20, 28, np.inf]
labels = ['Cool (<20)', 'Pleasant (20-28)', 'Hot (>28)']

y_test_cat = pd.cut(y_test, bins=bins, labels=labels)
y_pred_cat = pd.cut(y_pred, bins=bins, labels=labels)

cm = confusion_matrix(y_test_cat, y_pred_cat, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

print("\n--- Confusion Matrix (Accuracy by Category) ---")
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix: Weather Categories')
plt.show()

# ---------------------------------------------------------
# EXTRA: Predict for a specific scenario
# ---------------------------------------------------------
print("\n--- Example Prediction ---")

sample_row = X_test.iloc[0]
sample_actual = y_test.iloc[0]
sample_pred = model.predict([sample_row])[0]

print(f"Test Case: Row #{sample_row.name}")
print(f"Actual Tomorrow Temp:    {sample_actual:.2f}°C")
print(f"Model Predicted Temp:    {sample_pred:.2f}°C")
print(f"Difference:              {abs(sample_actual - sample_pred):.2f}°C")