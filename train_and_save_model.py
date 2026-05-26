"""
train_and_save_model.py
-----------------------
Yeh script chalao apne laptop pe — terminal ya VS Code mein.
Yeh script Flight_Fare.xlsx padhega, model train karega,
aur flight_fare_model.pkl + encoders.pkl tumhare folder mein save kar dega.

Run command:
    python train_and_save_model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import pickle

print("=" * 50)
print("  Flight Fare — Model Training Script")
print("  Author: Amir Nawed")
print("=" * 50)
print()

# ── Step 1: Load Data ──────────────────────────────────────────────────────────
print("📂 Loading dataset...")
df = pd.read_excel("Flight_Fare.xlsx")
print(f"   Loaded! Shape: {df.shape}")

# ── Step 2: Clean ──────────────────────────────────────────────────────────────
print("🧹 Cleaning data...")
df = df.drop_duplicates()
df = df.dropna()
print(f"   After cleaning: {df.shape}")

# ── Step 3: Feature Engineering ───────────────────────────────────────────────
print("⚙️  Feature engineering...")

# Date
df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y')
df['Journey_Day']     = df['Date_of_Journey'].dt.day
df['Journey_Month']   = df['Date_of_Journey'].dt.month
df.drop(columns=['Date_of_Journey'], inplace=True)

# Departure time
df['Dep_Hour'] = pd.to_datetime(df['Dep_Time']).dt.hour
df['Dep_Min']  = pd.to_datetime(df['Dep_Time']).dt.minute
df.drop(columns=['Dep_Time'], inplace=True)

# Arrival time
df['Arrival_Time_Clean'] = df['Arrival_Time'].apply(lambda x: x.split(' ')[0])
df['Arrival_Hour'] = pd.to_datetime(df['Arrival_Time_Clean']).dt.hour
df['Arrival_Min']  = pd.to_datetime(df['Arrival_Time_Clean']).dt.minute
df.drop(columns=['Arrival_Time', 'Arrival_Time_Clean'], inplace=True)

# Duration to minutes
def convert_duration(duration):
    hours = 0; minutes = 0
    if 'h' in duration:
        parts = duration.split('h')
        hours = int(parts[0].strip())
        if 'm' in parts[1]:
            minutes = int(parts[1].replace('m', '').strip())
    elif 'm' in duration:
        minutes = int(duration.replace('m', '').strip())
    return hours * 60 + minutes

df['Duration_Minutes'] = df['Duration'].apply(convert_duration)
df.drop(columns=['Duration'], inplace=True)

# Stops to number
stops_mapping = {'non-stop':0, '1 stop':1, '2 stops':2, '3 stops':3, '4 stops':4}
df['Total_Stops'] = df['Total_Stops'].map(stops_mapping)

# Drop Route
df.drop(columns=['Route'], inplace=True)

print("   Done!")

# ── Step 4: Encoding ───────────────────────────────────────────────────────────
print("🔢 Encoding categorical columns...")
cat_cols = ['Airline', 'Source', 'Destination', 'Additional_Info']
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"   {col} encoded")

# ── Step 5: Split ──────────────────────────────────────────────────────────────
X = df.drop(columns=['Price'])
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n📊 Train size: {X_train.shape} | Test size: {X_test.shape}")

# ── Step 6: Train XGBoost (best params from tuning) ───────────────────────────
print("\n🚀 Training XGBoost model...")
print("   (using best parameters found during tuning)")

# These are the best params found by RandomizedSearchCV
model = XGBRegressor(
    n_estimators     = 300,
    max_depth        = 7,
    learning_rate    = 0.1,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 5,
    random_state     = 42,
    verbosity        = 0
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n✅ Training complete!")
print(f"   R2 Score        : {r2:.4f}  ({r2*100:.1f}% accuracy)")
print(f"   Avg Error (MAE) : ₹{mae:.2f}")

# ── Step 7: Save model and encoders ───────────────────────────────────────────
print("\n💾 Saving model and encoders...")

with open("flight_fare_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("   ✅ flight_fare_model.pkl saved!")
print("   ✅ encoders.pkl saved!")
print()
print("=" * 50)
print("  Done! Now run:  streamlit run app.py")
print("=" * 50)
