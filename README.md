# DuBaoThoiTiet
Du Bao Thoi Tiet trong 7 ngay tiep theo theo Mo hinh, Thanh pho, ngay va cac dieu kien thoi tiet cua ngay duoc chon.

Do Chinh Xac (lay tu weather_model_train.py):
Mô hình | MAE (°C) | R2
--- | --- | ---
Gradient Boosting | 0.6181 | 94.45%
Linear Regression | 0.6214 | 94.22%
Ridge Regression | 0.6214 | 94.22%
Random Forest | 0.6239 | 94.43%
Decision Tree | 0.6521 | 93.67%

du lieu mo hinh duoc cap nhat den 14/1/2026

Top Predictors: Current day temperatures are the strongest predictors for next day's temperature.temperature_2m_mean ($0.97$)temperature_2m_min ($0.93$)temperature_2m_max ($0.92$)
Inverse Relationships: pressure_avg ($-0.60$) and humidity_avg ($-0.39$) have moderate negative correlations, meaning as pressure/humidity drops, temperature tends to rise.
Multicollinearity Warning: The input temperature variables (min, max, mean) are extremely highly correlated with each other ($>0.95$). You might consider removing one or two to avoid redundancy in linear models.
Weak Features: precipitation_sum and Month show negligible correlation to the target.

- loai bo nhitet do ttrung binh 