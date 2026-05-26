# flight-fare-prediction

# ✈️ Flight Fare Prediction using Machine Learning



> **Predict Indian domestic flight ticket prices using Machine Learning.**  
> Built with XGBoost, scikit-learn, and deployed as a Streamlit web app.

---

## 📌 Project Overview

Flight ticket prices change constantly based on many factors — airline, route, departure time, duration, number of stops, and more. This project builds a Machine Learning model that predicts the price of a flight ticket based on these features.

This is a **Supervised Machine Learning — Regression** problem where the goal is to predict a continuous numeric value (price in INR).

---

## 🎯 Problem Statement

**Given flight details such as airline, source, destination, departure time, duration, and number of stops — predict the ticket price (in INR).**

---

## 📊 Dataset Details

| Feature | Details |
|---------|---------|
| **Source** | Kaggle — Indian Flight Fare Dataset |
| **Rows** | 10,683 |
| **Columns** | 11 |
| **Target Column** | `Price` (flight ticket price in INR) |
| **Missing Values** | 2 rows |
| **Duplicate Rows** | 220 rows |
| **Problem Type** | Regression |

### 📋 Columns Description

| Column | Description |
|--------|-------------|
| `Airline` | Name of the airline (IndiGo, Air India, etc.) |
| `Date_of_Journey` | Date of travel (DD/MM/YYYY) |
| `Source` | Departure city |
| `Destination` | Arrival city |
| `Route` | Flight path with intermediate stops |
| `Dep_Time` | Departure time (HH:MM) |
| `Arrival_Time` | Arrival time (HH:MM) |
| `Duration` | Total flight duration (e.g. "2h 50m") |
| `Total_Stops` | Number of stops (non-stop, 1 stop, etc.) |
| `Additional_Info` | Extra info (meal, baggage, layover, etc.) |
| `Price` | 🎯 **Target** — ticket price in INR |

---

## 🔁 Project Workflow

Raw Data (.xlsx)
      ↓
Data Cleaning        → Remove 220 duplicates, drop 2 missing rows
      ↓
Feature Engineering  → Extract Day/Month from Date, Hour/Min from Time,
                       Duration → total minutes, Stops → numbers
      ↓
Encoding             → LabelEncoder for Airline, Source, Destination, Additional_Info
      ↓
Train 5 Models       → Linear Regression, Decision Tree, Random Forest,
                       Extra Trees, Gradient Boosting
      ↓
Compare Models       → MAE, RMSE, R2 Score
      ↓
Hyperparameter Tuning → RandomizedSearchCV on RF + XGBoost
      ↓
Best Model Selected  → XGBoost (Tuned) — R2: 88.7%, MAE: ₹740
      ↓
Save Model           → flight_fare_model.pkl + encoders.pkl
      ↓
Streamlit Web App    → User inputs → Predict → Show price

---

## 🧹 Data Cleaning & Feature Engineering

| Column | Problem | Solution |
|--------|---------|----------|
| `Date_of_Journey` | String format "24/03/2019" | Extracted Journey_Day and Journey_Month as numbers |
| `Dep_Time` | String "22:20" | Extracted Dep_Hour and Dep_Min as numbers |
| `Arrival_Time` | String with extra date "01:10 22 Mar" | Split to get only HH:MM, then extracted Hour and Min |
| `Duration` | String "2h 50m" in 3 different formats | Custom function → converted to total minutes |
| `Total_Stops` | Text "non-stop", "1 stop" etc. | Manual mapping → 0, 1, 2, 3, 4 |
| `Route` | Redundant — already in Source + Destination + Stops | Dropped the column |
| `Airline`, `Source`, `Destination`, `Additional_Info` | Text — model needs numbers | LabelEncoder (saved for app use) |
| Missing Values (2 rows) | 0.02% of data | Dropped with dropna() |
| Duplicate Rows (220) | Could cause overfitting | Removed with drop_duplicates() |

---

## 🤖 Models Trained

| Model | R2 Score | MAE (₹) | Notes |
|-------|----------|---------|-------|
| Linear Regression | ~0.60 | ~2800 | Too simple for non-linear price patterns |
| Decision Tree | ~0.80 | ~1100 | Good but prone to overfitting |
| Random Forest | ~0.88 | ~697 | Strong baseline — ensemble of 100 trees |
| Extra Trees | ~0.87 | ~720 | Similar to RF, slightly more random |
| Gradient Boosting | ~0.86 | ~780 | Sequential trees — powerful but slow |
| Random Forest (Tuned) | ~0.888 | ~710 | After RandomizedSearchCV |
| ✅ XGBoost (Tuned) | 0.887 | ₹740 | Final model — best overall |

### Why XGBoost?
- Builds trees sequentially — each tree fixes mistakes of the previous one
- Optimized version of Gradient Boosting — faster and more accurate
- Industry standard for tabular price prediction problems
- Lightweight model file (~1.7 MB vs 82 MB for Random Forest)

---

## 📈 Model Performance

Final Model  : XGBoost (Tuned)
R2 Score     : 0.887   → model explains 88.7% of price variation
MAE          : ₹740    → on average, prediction is only ₹740 off
RMSE         : ~₹1400

### 🔑 Top Features (by importance)
1. Duration — longer flights cost more
2. Total Stops — more stops = generally higher price
3. Airline — business class airlines charge significantly more
4. Departure Hour — morning/night flights tend to be cheaper
5. Journey Month — peak season months have higher prices

---

## 🌐 Streamlit Web App

### App Features
- ✅ Airline, Source, Destination dropdowns (only valid routes shown)
- ✅ Date picker (past dates disabled)
- ✅ Departure time picker
- ✅ Duration input → Arrival time auto-calculated
- ✅ Input validation (same city, zero duration checks)
- ✅ Clean result display with input summary table
- ✅ Model accuracy disclaimer shown

### Supported Routes

| From | To |
|------|----|
| Banglore | Delhi, New Delhi |
| Chennai | Kolkata |
| Delhi | Cochin |
| Kolkata | Banglore |
| Mumbai | Hyderabad |

---

## 🗂️ Project Structure

flight-fare-prediction/
│
├── Flight_Fare_Prediction_Amir_Nawed.ipynb   ← Main Jupyter notebook
├── app.py                                     ← Streamlit web app
├── Flight_Fare.xlsx                           ← Dataset
├── flight_fare_model.pkl                      ← Trained XGBoost model
├── encoders.pkl                               ← LabelEncoders for categories
├── requirements.txt                           ← Python dependencies
└── README.md                                  ← This file

---

## ⚙️ How to Run Locally

### Step 1 — Clone the repository
git clone https://github.com/amirnawed1/flight-fare-prediction.git
cd flight-fare-prediction

### Step 2 — Install dependencies
pip install -r requirements.txt

### Step 3 — Train and save the model (first time only)
python train_and_save_model.py

### Step 4 — Run the Streamlit app
streamlit run app.py

App will open at http://localhost:8501

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push all files to a GitHub repository
2. Go to https://share.streamlit.io
3. Login with GitHub
4. Click "New app" → Select your repo → Select app.py
5. Click Deploy — live link in 2-3 minutes!

---

## 📦 Requirements

streamlit
pandas
numpy
scikit-learn
xgboost
openpyxl

Install all:
pip install -r requirements.txt

---

## 💡 Key Learnings

- Feature Engineering is more important than model selection
- Saving encoders alongside the model is critical for production
- RandomizedSearchCV is much faster than GridSearchCV
- XGBoost gives comparable accuracy with 50x smaller model file
- End-to-end ML pipeline = Data → Clean → Engineer → Encode → Train → Tune → Save → Deploy

---

## 👨‍💻 Author

Amir Nawed
Machine Learning | Data Science

---

## 📄 License

This project is open source and available under the MIT License.
