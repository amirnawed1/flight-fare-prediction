"""
✈️ Flight Fare Prediction App
Author: Amir Nawed
Description: Streamlit web app to predict flight ticket prices using a trained ML model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import date, time, datetime, timedelta

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Flight Fare Predictor",
    page_icon="✈️",
    layout="centered"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
    <style>
        .stApp { background-color: #f0f4f8; }

        .main-title {
            text-align: center;
            color: #1a1a2e;
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0px;
        }
        .sub-title {
            text-align: center;
            color: #555;
            font-size: 1rem;
            margin-bottom: 25px;
        }
        .result-box {
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            color: white;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            margin-top: 20px;
        }
        .result-box h2 { font-size: 1.1rem; color: #aaa; margin-bottom: 5px; }
        .result-box h1 { font-size: 2.8rem; color: #00d4ff; font-weight: 800; }

        .section-header {
            font-size: 1rem;
            font-weight: 600;
            color: #333;
            margin-top: 10px;
            border-left: 4px solid #00d4ff;
            padding-left: 8px;
        }
        .arrival-box {
            background-color: #e8f4fd;
            border-left: 4px solid #00d4ff;
            padding: 10px 15px;
            border-radius: 5px;
            font-size: 1rem;
            color: #1a1a2e;
            margin-top: 8px;
        }
        .warning-box {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px 15px;
            border-radius: 5px;
            font-size: 0.9rem;
            color: #856404;
            margin-top: 8px;
        }
    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# VALID SOURCE → DESTINATION COMBINATIONS
# These are the only routes present in our training data.
# Model will give unreliable results for other combinations.
# ─────────────────────────────────────────────
VALID_ROUTES = {
    "Banglore" : ["Delhi", "New Delhi"],
    "Chennai"  : ["Kolkata"],
    "Delhi"    : ["Cochin"],
    "Kolkata"  : ["Banglore"],
    "Mumbai"   : ["Hyderabad"],
}

# ─────────────────────────────────────────────
# ADDITIONAL INFO — clean display names + tooltips
# Dataset had "No Info" and "No info" as two separate values — we treat both as same
# We show user-friendly labels but encode correctly behind the scenes
# ─────────────────────────────────────────────
ADDITIONAL_INFO_OPTIONS = {
    "No Info (Nothing special)"             : "No info",
    "In-flight meal not included"           : "In-flight meal not included",
    "No check-in baggage included"          : "No check-in baggage included",
    "Business class"                        : "Business class",
    "Red-eye flight (late night / early AM)": "Red-eye flight",
    "1 Short layover"                       : "1 Short layover",
    "1 Long layover"                        : "1 Long layover",
    "2 Long layovers"                       : "2 Long layover",
    "Change airports during journey"        : "Change airports",
}
# Display label → actual encoder value
# User sees clean label, we pass correct value to encoder

# ─────────────────────────────────────────────
# LOAD MODEL AND ENCODERS
# ─────────────────────────────────────────────
@st.cache_resource
def load_model_and_encoders():
    with open("flight_fare_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

try:
    model, encoders = load_model_and_encoders()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<p class="main-title">✈️ Flight Fare Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Enter your flight details to get an estimated ticket price</p>', unsafe_allow_html=True)
st.markdown("---")

if not model_loaded:
    st.error("⚠️ Model files not found! Please make sure 'flight_fare_model.pkl' and 'encoders.pkl' are in the same folder as app.py")
    st.stop()


# ─────────────────────────────────────────────
# SECTION 1 — FLIGHT DETAILS
# ─────────────────────────────────────────────
st.markdown('<p class="section-header">✈️ Flight Details</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    airline = st.selectbox(
        "Airline",
        options=encoders['Airline'].classes_.tolist(),
        help="💡 'Jet Airways Business' and 'Vistara Premium economy' are premium/business class — expect higher prices"
    )

with col2:
    # Show source options
    source = st.selectbox(
        "From (Source City)",
        options=list(VALID_ROUTES.keys())
    )

# Destination depends on source — only show valid destinations for selected source
valid_destinations = VALID_ROUTES[source]
destination = st.selectbox(
    "To (Destination City)",
    options=valid_destinations,
    help="Only valid routes from your selected source city are shown"
)




# ─────────────────────────────────────────────
# SECTION 2 — JOURNEY DETAILS
# ─────────────────────────────────────────────
st.markdown('<p class="section-header">📅 Journey Details</p>', unsafe_allow_html=True)

col4, col5 = st.columns(2)

with col4:
    # FIX: min_value = today — user cannot select past dates
    journey_date = st.date_input(
        "Date of Journey",
        value=date.today(),
        min_value=date.today(),   # cannot select past dates!
        help="Select your travel date. Past dates are not allowed."
    )

with col5:
    total_stops = st.selectbox(
        "Total Stops",
        options=["non-stop", "1 stop", "2 stops", "3 stops", "4 stops"],
        help="How many stops does this flight have? Non-stop = direct flight"
    )


# ─────────────────────────────────────────────
# SECTION 3 — DEPARTURE TIME & DURATION
# ─────────────────────────────────────────────
st.markdown('<p class="section-header">🕐 Departure Time & Duration</p>', unsafe_allow_html=True)

col6, col7 = st.columns(2)

with col6:
    dep_time = st.time_input(
        "🛫 Departure Time",
        value=time(8, 0),
        help="What time does your flight depart? (24-hour format — e.g. 14:30 = 2:30 PM)"
    )
    dep_hour = dep_time.hour    # model needs hour as a number
    dep_min  = dep_time.minute  # model needs minute as a number

with col7:
    duration_hours   = st.number_input("Flight Duration — Hours",   min_value=0, max_value=48, value=2)
    duration_minutes = st.number_input("Flight Duration — Minutes", min_value=0, max_value=59, value=30)

# ── Auto-calculate Arrival Time ────────────────────────────────────────────
# User does NOT enter arrival time — we calculate it automatically
# Arrival = Departure Time + Duration
total_duration_mins = (duration_hours * 60) + duration_minutes

if total_duration_mins > 0:
    dep_datetime     = datetime.combine(journey_date, dep_time)
    arrival_datetime = dep_datetime + timedelta(minutes=total_duration_mins)
    arrival_hour     = arrival_datetime.hour
    arrival_min      = arrival_datetime.minute
    arrival_display  = arrival_datetime.strftime("%I:%M %p")
    next_day_note    = " (next day ✈)" if arrival_datetime.date() != journey_date else ""

    st.markdown(
        f'<div class="arrival-box">🛬 Estimated Arrival Time: <strong>{arrival_display}{next_day_note}</strong>'
        f' &nbsp;—&nbsp; auto-calculated from departure + duration</div>',
        unsafe_allow_html=True
    )
else:
    # Duration is 0 — set arrival same as departure
    arrival_hour = dep_hour
    arrival_min  = dep_min


# ─────────────────────────────────────────────
# SECTION 4 — ADDITIONAL INFO
# ─────────────────────────────────────────────
st.markdown('<p class="section-header">ℹ️ Additional Flight Info</p>', unsafe_allow_html=True)

selected_label = st.selectbox(
    "Any extra info about this flight?",
    options=list(ADDITIONAL_INFO_OPTIONS.keys()),
    help="Select the best matching option. If nothing applies, choose 'No Info'."
)
# Map display label → actual encoder value
additional_info_value = ADDITIONAL_INFO_OPTIONS[selected_label]

# Handle "No Info" / "No info" duplicate in encoder
# Dataset had both — we always use "No info" (lowercase i) as default
if additional_info_value == "No info" and "No info" not in encoders['Additional_Info'].classes_:
    additional_info_value = "No Info"


# ─────────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────────
st.markdown("---")
predict_btn = st.button("🔮 Predict Flight Price", use_container_width=True)


# ─────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────
if predict_btn:

    # ── Validations ───────────────────────────
    errors = []

    if source == destination:
        errors.append("Source and Destination cannot be the same city.")

    if total_duration_mins == 0:
        errors.append("Flight duration cannot be 0. Please enter how long the flight takes.")

    if errors:
        for err in errors:
            st.error(f"⚠️ {err}")

    else:
        # ── Encode categorical inputs ──────────
        airline_encoded  = encoders['Airline'].transform([airline])[0]
        source_encoded   = encoders['Source'].transform([source])[0]
        dest_encoded     = encoders['Destination'].transform([destination])[0]
        add_info_encoded = encoders['Additional_Info'].transform([additional_info_value])[0]

        # ── Convert stops to number ────────────
        stops_mapping = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}
        stops_num = stops_mapping[total_stops]

        # ── Extract day and month ──────────────
        journey_day   = journey_date.day
        journey_month = journey_date.month

        # ── Build input DataFrame ──────────────
        # Column order must EXACTLY match how model was trained!
        input_data = pd.DataFrame([[
            airline_encoded,
            source_encoded,
            dest_encoded,
            stops_num,
            add_info_encoded,
            journey_day,
            journey_month,
            dep_hour,
            dep_min,
            arrival_hour,   # auto-calculated
            arrival_min,    # auto-calculated
            total_duration_mins
        ]], columns=[
            'Airline', 'Source', 'Destination', 'Total_Stops', 'Additional_Info',
            'Journey_Day', 'Journey_Month', 'Dep_Hour', 'Dep_Min',
            'Arrival_Hour', 'Arrival_Min', 'Duration_Minutes'
        ])

        # ── Predict! ───────────────────────────
        predicted_price = model.predict(input_data)[0]

        # ── Show result ────────────────────────
        st.markdown(f"""
            <div class="result-box">
                <h2>Estimated Flight Price</h2>
                <h1>₹ {predicted_price:,.0f}</h1>
                <p style="color:#aaa; font-size:0.85rem;">
                    {airline} &nbsp;|&nbsp; {source} → {destination} &nbsp;|&nbsp; {total_stops}
                </p>
            </div>
        """, unsafe_allow_html=True)

        # ── Show summary table ─────────────────
        st.markdown("#### 📋 Your Flight Summary")
        summary_df = pd.DataFrame({
            "Detail": [
                "Airline", "From", "To", "Stops",
                "Journey Date", "Departure Time", "Arrival Time",
                "Duration", "Extra Info"
            ],
            "Value": [
                airline,
                source,
                destination,
                total_stops,
                journey_date.strftime("%d %B %Y"),
                dep_time.strftime("%I:%M %p"),
                arrival_display + next_day_note,
                f"{duration_hours}h {duration_minutes}m",
                selected_label
            ]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # ── Disclaimer ─────────────────────────
        st.caption(
            "📌 Note: This is an ML-based estimate. "
            "Actual prices may vary based on demand, booking time, and availability. "
            f"Model accuracy: ~88.7% R2 Score, avg error ±₹740."
        )


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#aaa; font-size:0.8rem;'>"
    "Flight Fare Prediction | ML Project by Amir Nawed | XGBoost Model"
    "</p>",
    unsafe_allow_html=True
)
