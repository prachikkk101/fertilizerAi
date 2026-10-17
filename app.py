from pathlib import Path
import sys
import time
import re

import pandas as pd
import streamlit as st

# Folder layout expected:
# FertilizerAI/                  <- outer project folder
#   app.py                       <- this file
#   hardware/main1.py
#   hardware/sensor_log.csv
#   FertilizerAI/src/models/predict.py

APP_ROOT = Path(__file__).resolve().parent
HARDWARE_DIR = APP_ROOT / "hardware"
CSV_PATH = HARDWARE_DIR / "sensor_log.csv"
PYTHON_PROJECT_DIR = APP_ROOT / "FertilizerAI"

if str(PYTHON_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_PROJECT_DIR))

try:
    from src.models.predict import predict_fertilizer
except Exception as import_error:
    predict_fertilizer = None
    PREDICT_IMPORT_ERROR = import_error
else:
    PREDICT_IMPORT_ERROR = None

st.set_page_config(
    page_title="Fertilizer AI Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
      .hero {padding: 1.25rem 1.4rem; border-radius: 16px;
             background: linear-gradient(115deg, #123b2a, #22764c); color: white;
             margin-bottom: 1rem;}
      .hero h1 {margin: 0; font-size: 2rem;}
      .hero p {margin: .45rem 0 0 0; opacity: .9;}
      div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22);
          padding: .8rem .9rem; border-radius: 12px;}
      .small-note {color: #777; font-size: .85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🌱 Fertilizer AI Dashboard</h1>
      <p>Live sensor readings · On-demand fertilizer prediction · Sensor history</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Normalise headers so variations like "Temperature (C)" and "Temperature_C"
# can be recognised without editing the receiver's CSV writer.
def normalize_header(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())

COLUMN_ALIASES = {
    "timestamp": ["timestamp", "datetime", "date_time", "time", "received_at", "received"],
    "moisture": ["moisture_raw", "moistureraw", "soil_moisture_raw", "moisture", "soilmoisture"],
    "moisture_voltage": ["moisture_voltage", "moisturevoltage", "voltage", "moisture_v"],
    "temperature": ["temperature", "temperature_c", "temperaturec", "temp", "temp_c"],
    "humidity": ["humidity", "humidity_rh", "humidity_pct", "humidity_percent", "relative_humidity", "relativehumidity"],
    "nitrogen": ["nitrogen", "nitrogen_n", "n"],
    "phosphorus": ["phosphorus", "phosphorous", "phosphorus_p", "p"],
    "potassium": ["potassium", "potassium_k", "k"],
}


CSV_COLUMNS = [
    "Timestamp",
    "Moisture_Raw",
    "Moisture_Voltage",
    "Temperature",
    "Humidity",
    "Nitrogen",
    "Phosphorus",
    "Potassium",
]


def read_sensor_csv():
    """Read the receiver's headerless CSV, tolerating concurrent appends.

    The current hardware/main1.py writes eight fields per row without a header:
    timestamp, moisture raw, moisture voltage, temperature, humidity, N, P, K.
    Passing header=None prevents pandas from mistakenly treating the first
    sensor packet as the CSV's column names.
    """
    if not CSV_PATH.exists():
        return None, f"Sensor CSV not found: {CSV_PATH}"
    last_error = None
    for _ in range(3):
        try:
            data = pd.read_csv(CSV_PATH, header=None, names=CSV_COLUMNS)
            if data.empty:
                return data, "CSV exists, but no sensor records have been saved yet."

            # Be tolerant if a header row is added to the CSV in a future version.
            first_value = normalize_header(data.iloc[0]["Timestamp"])
            if first_value in {"timestamp", "datetime", "date", "time", "receivedat"}:
                data = data.iloc[1:].reset_index(drop=True)

            # Keep sensor columns numeric so metrics/charts and prediction checks work.
            for column in CSV_COLUMNS[1:]:
                data[column] = pd.to_numeric(data[column], errors="coerce")

            if data.empty:
                return data, "CSV exists, but no sensor records have been saved yet."
            return data, None
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError, ValueError) as exc:
            last_error = exc
            time.sleep(0.12)
    return None, f"Could not read the sensor CSV: {last_error}"


def find_column(data, logical_name):
    aliases = {normalize_header(x) for x in COLUMN_ALIASES[logical_name]}
    for column in data.columns:
        if normalize_header(column) in aliases:
            return column
    return None


def latest_value(data, logical_name):
    column = find_column(data, logical_name)
    if column is None:
        return None
    value = data.iloc[-1][column]
    if pd.isna(value):
        return None
    return value


def as_number(value):
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def display_number(value, suffix="", digits=2):
    numeric = as_number(value)
    return "—" if numeric is None else f"{numeric:.{digits}f}{suffix}"


def sensor_snapshot(data):
    return {
        "timestamp": latest_value(data, "timestamp"),
        "moisture": latest_value(data, "moisture"),
        "moisture_voltage": latest_value(data, "moisture_voltage"),
        "temperature": latest_value(data, "temperature"),
        "humidity": latest_value(data, "humidity"),
        "nitrogen": latest_value(data, "nitrogen"),
        "phosphorus": latest_value(data, "phosphorus"),
        "potassium": latest_value(data, "potassium"),
    }


# Context fields that are not currently supplied by the sensors are selected here.
with st.sidebar:
    st.header("🌾 Farm details")
    district = st.text_input("District", value="Vellore")
    soil_color = st.selectbox("Soil color", ["Red", "Black", "Brown", "Alluvial", "Other"], index=0)
    ph_value = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=900.0, step=10.0)
    crop = st.selectbox("Crop", ["Rice", "Maize", "Wheat", "Cotton", "Sugarcane", "Groundnut", "Other"], index=0)
    growth_stage = st.selectbox("Growth stage", ["Germination", "Vegetative", "Flowering", "Fruiting", "Maturity"], index=1)
    st.divider()
    st.caption("The receiver must be running in another PowerShell window for live data.")

st.subheader("Live sensor readings")
st.caption("This panel refreshes automatically about every 2 seconds. Prediction runs only when you press the button.")

@st.fragment(run_every="2s")
def live_sensor_panel():
    data, error = read_sensor_csv()
    if error:
        st.info(error)
        return
    snapshot = sensor_snapshot(data)
    stamp = snapshot["timestamp"]
    if stamp is not None:
        st.caption(f"Latest CSV record: {stamp}")
    else:
        st.caption(f"Showing latest of {len(data)} saved record(s).")

    metrics = [
        ("Temperature", snapshot["temperature"], " °C"),
        ("Humidity", snapshot["humidity"], " %RH"),
        ("Soil moisture raw", snapshot["moisture"], ""),
        ("Moisture voltage", snapshot["moisture_voltage"], " V"),
        ("Nitrogen (N)", snapshot["nitrogen"], ""),
        ("Phosphorus (P)", snapshot["phosphorus"], ""),
        ("Potassium (K)", snapshot["potassium"], ""),
    ]
    # Seven metrics laid out compactly.
    row1 = st.columns(4)
    row2 = st.columns(3)
    for index, (label, value, suffix) in enumerate(metrics):
        target = row1[index] if index < 4 else row2[index - 4]
        target.metric(label, display_number(value, suffix))

    numeric_columns = []
    for logical_name in ("temperature", "humidity", "moisture", "nitrogen", "phosphorus", "potassium"):
        col = find_column(data, logical_name)
        if col is not None and pd.api.types.is_numeric_dtype(data[col]):
            numeric_columns.append(col)
    if numeric_columns and len(data) > 1:
        st.markdown("**Recent sensor trend**")
        chart_data = data[numeric_columns].tail(60).copy()
        chart_data = chart_data.apply(pd.to_numeric, errors="coerce")
        st.line_chart(chart_data)

live_sensor_panel()

st.divider()
st.subheader("AI fertilizer prediction")
st.write("Click once to use the newest saved sensor record. The AI will not run again until you click the button again.")

if PREDICT_IMPORT_ERROR is not None:
    st.error(
        "Could not import the prediction module. Check that the project contains "
        "FertilizerAI/src/models/predict.py and its model files. "
        f"Details: {PREDICT_IMPORT_ERROR}"
    )

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_prediction_time" not in st.session_state:
    st.session_state.last_prediction_time = None

if st.button("🔬 Predict Fertilizer", type="primary", use_container_width=True, disabled=(predict_fertilizer is None)):
    data, error = read_sensor_csv()
    if error or data is None or data.empty:
        st.error(error or "No sensor record is available yet. Start the receiver first.")
    else:
        snapshot = sensor_snapshot(data)
        required_sensor_values = {
            "Temperature": snapshot["temperature"],
            "Nitrogen": snapshot["nitrogen"],
            "Phosphorus": snapshot["phosphorus"],
            "Potassium": snapshot["potassium"],
        }
        missing = [name for name, value in required_sensor_values.items() if as_number(value) is None]
        if missing:
            st.error(
                "The CSV is missing usable values for: " + ", ".join(missing) +
                ". Check the CSV header and make sure main1.py has received data. "
                f"Detected CSV columns: {', '.join(map(str, data.columns))}"
            )
        else:
            model_input = {
                "District_Name": district,
                "Soil_color": soil_color,
                "Nitrogen": as_number(snapshot["nitrogen"]),
                "Phosphorus": as_number(snapshot["phosphorus"]),
                "Potassium": as_number(snapshot["potassium"]),
                "pH": float(ph_value),
                "Rainfall": float(rainfall),
                "Temperature": as_number(snapshot["temperature"]),
                "Crop": crop,
                "Growth_Stage": growth_stage,
            }
            try:
                with st.spinner("Running the fertilizer model once..."):
                    result = predict_fertilizer(model_input)
                st.session_state.last_prediction = result
                st.session_state.last_prediction_time = time.strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.prediction_input_snapshot = model_input
                st.success("Prediction completed.")
            except Exception as exc:
                st.error(f"Prediction failed: {type(exc).__name__}: {exc}")

result = st.session_state.last_prediction
if result:
    st.markdown("### Latest prediction")
    if st.session_state.last_prediction_time:
        st.caption(f"Generated at {st.session_state.last_prediction_time}")
    st.metric("Recommended fertilizer", result.get("Recommended_Fertilizer", "—"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Primary amount", f"{result.get('Primary_Amount_kg_per_hectare', '—')} kg/hectare")
    c2.metric("Water", f"{result.get('Water_Liters', '—')} liters")
    c3.metric("Pump 1", f"{result.get('Pump1_ml', '—')} ml")
    c4, c5 = st.columns(2)
    c4.metric("Pump 2", f"{result.get('Pump2_ml', '—')} ml")
    c5.metric("Pump 3", f"{result.get('Pump3_ml', '—')} ml")
    with st.expander("Show exact values sent to the model"):
        st.json(st.session_state.get("prediction_input_snapshot", {}))
        st.json(result)

st.divider()
st.caption(
    "Local dashboard only: open it on the same computer as the sensor receiver. "
    "The model outputs need agronomic validation and pump calibration before they are used for automatic dosing."
)
