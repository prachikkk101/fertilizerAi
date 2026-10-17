<!-- ============================================================ -->
<!--         FERTILIZER AI — TECHNICAL DISCLOSURE DOCUMENT        -->
<!-- ============================================================ -->

<div align="center">

# 🌱 FERTILIZER AI
### An AI-Powered Real-Time Soil Nutrient Monitoring & Precision Fertilizer Recommendation System

---

**Publication Reference:** FertilizerAI-001  
**Filing Date:** October 2026  
**Inventors / Authors:** Prachikk101 et al.  
**Institution:** Vellore, Tamil Nadu, India  
**Status:** Prototype — Proof of Concept  

---

</div>

---

## ABSTRACT

The present invention discloses a system and method for precision agricultural fertilizer recommendation using a wireless, multi-sensor IoT hardware node coupled with a machine-learning prediction engine and a real-time interactive web dashboard. The system continuously measures in-situ soil and ambient parameters — including Nitrogen (N), Phosphorus (P), Potassium (K) concentrations, soil moisture (raw ADC and voltage), ambient temperature, and relative humidity — via sensor arrays interfaced to a Raspberry Pi Pico W microcontroller. Sensor data is transmitted over a UDP/Wi-Fi link to a host computer running a Python-based receiver (`main1.py`) that logs all readings to a timestamped CSV flat file. A Streamlit dashboard (`app.py`) reads the live CSV, renders real-time metric cards and trend charts refreshed every 2 seconds, and on-demand invokes a trained ML model (`predict_fertilizer`) to recommend the optimal fertilizer type, application quantity (kg/hectare), water volume, and three-pump dosing volumes. The invention addresses the limitation of generic, non-site-specific fertilizer application by providing crop-type, growth-stage, soil-color, pH, rainfall, and district-aware predictions, demonstrably reducing nutrient over-application and crop loss risk.

---

## TABLE OF CONTENTS

1. [Technical Field](#1-technical-field)
2. [Background of the Invention](#2-background-of-the-invention)
3. [Summary of the Invention](#3-summary-of-the-invention)
4. [System Architecture](#4-system-architecture)
5. [Detailed Description of Embodiments](#5-detailed-description-of-embodiments)
   - 5.1 [Hardware Node — Sensor Array & Wireless Transmitter](#51-hardware-node--sensor-array--wireless-transmitter)
   - 5.2 [Physical Prototype — Field Deployment Unit](#52-physical-prototype--field-deployment-unit)
   - 5.3 [Host Receiver & Data Pipeline](#53-host-receiver--data-pipeline)
   - 5.4 [Live Dashboard — Sensor Monitoring Panel](#54-live-dashboard--sensor-monitoring-panel)
   - 5.5 [Live Dashboard — Historical Trend Analysis](#55-live-dashboard--historical-trend-analysis)
   - 5.6 [AI Prediction Engine](#56-ai-prediction-engine)
6. [Data Protocol & CSV Schema](#6-data-protocol--csv-schema)
7. [Machine Learning Model Interface](#7-machine-learning-model-interface)
8. [Novelty & Claims](#8-novelty--claims)
9. [Bill of Materials (BOM)](#9-bill-of-materials-bom)
10. [Installation & Operation](#10-installation--operation)
11. [File Structure](#11-file-structure)
12. [Limitations & Future Work](#12-limitations--future-work)
13. [License](#13-license)

---

## 1. TECHNICAL FIELD

The present disclosure relates to the field of **precision agriculture**, and more specifically to an **edge-computing IoT system** that integrates multi-parameter soil sensing, wireless data telemetry, time-series logging, and machine-learning-based fertilizer prescription into a unified, locally-running platform operable without internet connectivity.

---

## 2. BACKGROUND OF THE INVENTION

Conventional fertilizer application in smallholder and commercial farming is guided primarily by seasonal schedules or generic soil test results obtained through laboratory analysis. Such approaches suffer from: (i) temporal lag between soil sampling and laboratory reporting (typically 5–14 days); (ii) inability to capture spatial or temporal variability within a single field; (iii) generic recommendations that do not account for real-time crop growth stage, ambient micro-climate, or soil moisture dynamics; and (iv) over-application of chemical fertilizers resulting in environmental run-off, soil acidification, and increased input cost.

Existing IoT-based soil monitoring systems are predominantly data collection platforms without integrated decision-support, requiring the farmer to interpret raw sensor values. Conversely, AI-based recommendation engines typically operate on static, offline datasets without real-time sensor integration.

The invention described herein addresses this gap by providing a **closed-loop pipeline** where real-time sensor readings directly feed an AI recommendation engine, and the farmer receives actionable, crop-specific dosing instructions with sub-minute latency.

---

## 3. SUMMARY OF THE INVENTION

The invention provides:

1. A **wireless soil sensor node** (Raspberry Pi Pico W + NPK sensor + DHT sensor + resistive soil moisture probe) that continuously samples soil and ambient conditions and transmits JSON packets over UDP/Wi-Fi.
2. A **host receiver daemon** (`main1.py`) that listens on UDP port 8000, parses incoming JSON, appends structured records to a rolling CSV log, and optionally runs AI inference on each packet.
3. A **Streamlit web dashboard** (`app.py`) providing: live auto-refreshing sensor metrics (every 2 s), 60-point trend line charts, configurable farm context (district, soil color, pH, rainfall, crop, growth stage), and an on-demand AI fertilizer prediction button.
4. A **trained ML prediction module** (`src/models/predict.py`) that accepts 10 agronomic features and outputs fertilizer type, primary dosage, water volume, and three-pump dispense volumes.

---

## 4. SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HARDWARE LAYER                               │
│                                                                     │
│   ┌──────────────────┐     ┌────────────────┐    ┌──────────────┐  │
│   │  NPK Soil Sensor  │────▶│ Pico W (Wi-Fi) │───▶│  DHT Sensor  │  │
│   └──────────────────┘     │  MicroPython   │    └──────────────┘  │
│   ┌──────────────────┐     │                │    ┌──────────────┐  │
│   │ Moisture Probe   │────▶│  JSON Encoder  │    │  ADC Moisture│  │
│   └──────────────────┘     └───────┬────────┘    └──────────────┘  │
│                                    │ UDP Packet (Port 8000)         │
└────────────────────────────────────┼────────────────────────────────┘
                                     │ Wi-Fi (LAN)
┌────────────────────────────────────▼────────────────────────────────┐
│                         SOFTWARE LAYER                              │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │           main1.py  —  UDP Receiver Daemon                   │  │
│   │  • socket.recvfrom(4096)                                     │  │
│   │  • json.loads() → validate → write CSV row                   │  │
│   │  • Optional: call predict_fertilizer() per packet            │  │
│   └──────────────────────────────────┬───────────────────────────┘  │
│                                      │ sensor_log.csv               │
│   ┌──────────────────────────────────▼───────────────────────────┐  │
│   │           app.py  —  Streamlit Dashboard                     │  │
│   │  • pd.read_csv() → @st.fragment(run_every=2s)                │  │
│   │  • 7-metric card display                                     │  │
│   │  • 60-point rolling line chart                               │  │
│   │  • Sidebar: District, Soil Color, pH, Rainfall, Crop, Stage  │  │
│   │  • Button → predict_fertilizer(model_input) → result cards   │  │
│   └──────────────────────────────────┬───────────────────────────┘  │
│                                      │                              │
│   ┌──────────────────────────────────▼───────────────────────────┐  │
│   │        src/models/predict.py  —  ML Inference Engine         │  │
│   │  • Input: 10 features (N, P, K, Temp, pH, Rainfall, …)      │  │
│   │  • Output: Fertilizer type, kg/ha, Water L, Pump ml ×3      │  │
│   └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. DETAILED DESCRIPTION OF EMBODIMENTS

### 5.1 Hardware Node — Sensor Array & Wireless Transmitter

The hardware sensor node comprises a **Raspberry Pi Pico W** microcontroller programmed in MicroPython. The node integrates the following sensor peripherals:

| Sensor | Parameter Measured | Interface |
|---|---|---|
| RS485 NPK Soil Sensor | Nitrogen, Phosphorus, Potassium (mg/kg) | UART / RS485 |
| DHT11 / DHT22 | Ambient Temperature (°C), Relative Humidity (%RH) | 1-Wire Digital |
| Resistive Moisture Probe | Soil moisture (raw ADC counts, 0–65535) | ADC |
| Resistive Moisture Probe | Soil moisture voltage (V) | ADC (converted) |

The Pico W encodes all sampled values into a single JSON object and transmits it via **UDP broadcast** to the host computer's IP address on port 8000. The wireless link uses the station's local Wi-Fi network (2.4 GHz, 802.11n), requiring no additional hardware or cloud relay.

**Sample transmitted JSON packet:**
```json
{
  "moisture_raw": 472,
  "moisture_voltage": 0.02,
  "temperature": 30.69,
  "humidity": 69.08,
  "nitrogen": 3.0,
  "phosphorus": 4.0,
  "potassium": 11.0
}
```

---

### 5.2 Physical Prototype — Field Deployment Unit

> **Figure 1 — Sensor Node Prototype (Soil Insertion Unit)**

The first embodiment of the physical hardware is illustrated below. The prototype uses a recycled PET bottle as a waterproof enclosure/planter. The Pico W board is mounted at the top of the bottle, with sensor probe wires running down into the soil medium. This form factor enables rapid field deployment and testing without custom PCB fabrication.

![Figure 1 — Sensor Node prototype](sensor_hardware_prototype_1792245145996.jpg)

*Figure 1: The sensor node comprises a Pico W microcontroller (black rectangular board, top) mounted on an inverted PET bottle acting as a soil container. Conductive sensor probes penetrate the soil medium to measure NPK concentrations and moisture content. A USB cable supplies power from the host machine. The sensor continuously broadcasts JSON telemetry over the LAN Wi-Fi interface.*

---

### 5.3 Breadboard Wiring Prototype (Full Circuit Assembly)

> **Figure 2 — Full Circuit Assembly**

![Figure 2 — Full circuit assembly](circuit_wiring_diagram_1792245166726.jpg)

*Figure 2: The complete circuit assembly shows the Raspberry Pi Pico W (center board) mounted on a solderless breadboard with all sensor connections established via Dupont jumper wires. Color coding: Red = 3.3V/5V VCC rails; Black = GND; Blue = data lines; Yellow = ADC inputs; White = UART Tx/Rx. The DHT sensor module (top-left) and NPK RS485 soil sensor module connect via UART with an RS485-to-TTL adapter. The resistive moisture probe pair connects to two ADC pins on the Pico W.*

**Pin Mapping (Pico W):**

| GPIO Pin | Sensor Connection | Signal Type |
|---|---|---|
| GP0 (UART0 TX) | RS485 adapter TX | UART |
| GP1 (UART0 RX) | RS485 adapter RX | UART |
| GP15 | DHT11/22 data | 1-Wire |
| GP26 (ADC0) | Moisture probe signal | Analog |
| GP27 (ADC1) | Moisture voltage reference | Analog |
| 3V3 | VCC rails | Power |
| GND | Common ground | Power |

---

### 5.4 Host Receiver & Data Pipeline

The host receiver (`hardware/main1.py`) is a Python 3 daemon that:

1. Opens a **UDP socket** bound to `0.0.0.0:8000`, accepting packets from any local network source.
2. On each received packet: decodes UTF-8 bytes → parses JSON → extracts the seven sensor fields → writes a timestamped row to `hardware/sensor_log.csv`.
3. Validates NPK and temperature values before triggering ML inference; logs prediction results to `stdout`.
4. Implements graceful shutdown on `KeyboardInterrupt` with socket cleanup.

The receiver is designed to be run in a **separate terminal window** from the Streamlit dashboard, enabling concurrent writing (receiver) and reading (dashboard) of the CSV file. File-level concurrency is handled via retry logic in the dashboard's `read_sensor_csv()` function (up to 3 read attempts with 120 ms delays).

---

### 5.5 Live Dashboard — Sensor Monitoring Panel

> **Figure 3 — Live Sensor Readings Panel**

![Figure 3 — Live sensor readings panel](dashboard_live_readings_1792245233408.jpg)

*Figure 3: The Streamlit dashboard live sensor panel at localhost:8501. The dark green hero banner identifies the application. Seven metric tiles are arranged in a 4+3 grid showing: Temperature 30.69°C, Humidity 69.08%RH, Soil Moisture Raw 472.00, Moisture Voltage 0.02V, Nitrogen 3.00, Phosphorus 4.00, Potassium 11.00. The panel auto-refreshes every 2 seconds via @st.fragment(run_every="2s"). The left sidebar provides editable farm context: District (Vellore), Soil Color (Red), pH, Rainfall, Crop (Rice), Growth Stage (Vegetative).*

---

### 5.6 Live Dashboard — Historical Trend Analysis

> **Figure 4 — Sensor Trend Chart & NPK Summary**

![Figure 4 — NPK summary and sensor trend chart](dashboard_ai_prediction_1792245250311.jpg)

*Figure 4: The lower portion of the dashboard renders a 60-point rolling line chart (st.line_chart) plotting the most recent 60 sensor records for all numeric columns: Humidity, Soil Moisture Raw, Nitrogen, Phosphorus, and Potassium. This time-series view enables detection of moisture trends, NPK depletion patterns, and sensor anomalies. Below the chart, the "AI Fertilizer Prediction" section is prominently displayed with a call-to-action button.*

---

### 5.7 AI Prediction Engine

The prediction engine is encapsulated in `FertilizerAI/src/models/predict.py` and exposes a single entry point:

```python
def predict_fertilizer(input_dict: dict) -> dict:
    """
    Parameters
    ----------
    input_dict : dict
        Keys: District_Name, Soil_color, Nitrogen, Phosphorus, Potassium,
              pH, Rainfall, Temperature, Crop, Growth_Stage

    Returns
    -------
    dict
        Keys: Recommended_Fertilizer, Primary_Amount_kg_per_hectare,
              Water_Liters, Pump1_ml, Pump2_ml, Pump3_ml
    """
```

**Model Input Features:**

| Feature | Source | Type |
|---|---|---|
| `Nitrogen` | NPK Soil Sensor (live) | Float |
| `Phosphorus` | NPK Soil Sensor (live) | Float |
| `Potassium` | NPK Soil Sensor (live) | Float |
| `Temperature` | DHT Sensor (live) | Float |
| `District_Name` | User sidebar input | Categorical |
| `Soil_color` | User sidebar input | Categorical |
| `pH` | User sidebar input | Float |
| `Rainfall` | User sidebar input | Float (mm) |
| `Crop` | User sidebar input | Categorical |
| `Growth_Stage` | User sidebar input | Categorical |

**Model Output:**

| Output | Description | Unit |
|---|---|---|
| `Recommended_Fertilizer` | Fertilizer product name | String |
| `Primary_Amount_kg_per_hectare` | Primary nutrient mass | kg/hectare |
| `Water_Liters` | Irrigation/dilution volume | Liters |
| `Pump1_ml` | Pump 1 dispense volume | mL |
| `Pump2_ml` | Pump 2 dispense volume | mL |
| `Pump3_ml` | Pump 3 dispense volume | mL |

---

## 6. DATA PROTOCOL & CSV SCHEMA

### UDP Packet Format

- **Transport:** UDP (connectionless, broadcast-compatible)
- **Port:** 8000
- **Encoding:** UTF-8 JSON
- **Max Packet Size:** 4096 bytes
- **Send Frequency:** Configurable on Pico W (typical: 2–5 seconds)

### CSV Log Schema (`hardware/sensor_log.csv`)

```
timestamp,moisture_raw,moisture_voltage,temperature,humidity,nitrogen,phosphorus,potassium
2026-10-10 12:34:46,472,0.02,30.69,69.08,3.0,4.0,11.0
2026-10-10 12:34:48,466,0.02,30.68,69.09,3.0,4.0,11.0
```

| Column | Data Type | Description |
|---|---|---|
| `timestamp` | `YYYY-MM-DD HH:MM:SS` | Host receipt time |
| `moisture_raw` | Integer | Raw ADC reading (0–65535) |
| `moisture_voltage` | Float | Computed from ADC (0.0–3.3 V) |
| `temperature` | Float | Celsius |
| `humidity` | Float | Percent relative humidity |
| `nitrogen` | Float | Soil N concentration |
| `phosphorus` | Float | Soil P concentration |
| `potassium` | Float | Soil K concentration |

---

## 7. MACHINE LEARNING MODEL INTERFACE

The prediction module is decoupled from the sensor pipeline and loaded once at dashboard startup. Import errors are caught gracefully; if the model files are absent, the prediction button is disabled and an informative error is displayed.

```python
# Graceful import with error surfacing
try:
    from src.models.predict import predict_fertilizer
except Exception as import_error:
    predict_fertilizer = None
    PREDICT_IMPORT_ERROR = import_error
```

Prediction is triggered only when:
1. A valid sensor CSV record exists
2. Nitrogen, Phosphorus, Potassium, and Temperature are non-null numeric values
3. The user presses the **"🔬 Predict Fertilizer"** button

Results are stored in `st.session_state` and persist across the 2-second auto-refresh cycle.

---

## 8. NOVELTY & CLAIMS

**Claim 1:** A system comprising a wireless IoT sensor node transmitting NPK, temperature, humidity, and soil moisture data as structured JSON over UDP, wherein the receiving host simultaneously logs data to a CSV and serves a real-time web dashboard from the same file without a shared-memory bus or database intermediary.

**Claim 2:** A method of auto-refreshing a sensor dashboard using Streamlit's fragment-based partial rendering (`@st.fragment(run_every=N)`) with file-level retry logic tolerating concurrent write-append operations from a separate process.

**Claim 3:** A fertilizer recommendation engine wherein model input is a hybrid of real-time sensor values and user-supplied agronomic context (crop type, growth stage, soil color, district, pH, rainfall), enabling growth-stage-aware and region-aware dosing instructions not achievable from sensor data alone.

**Claim 4:** A column-aliasing scheme for CSV normalization (`COLUMN_ALIASES`, `normalize_header`) that enables the dashboard to parse sensor log files regardless of column naming conventions applied by different hardware firmware versions.

**Claim 5:** An architecture wherein the same ML prediction function (`predict_fertilizer`) is callable from both the receiver daemon (automated, per-packet inference) and the web dashboard (on-demand, button-triggered inference) without code duplication.

---

## 9. BILL OF MATERIALS (BOM)

| Component | Specification | Qty | Purpose |
|---|---|---|---|
| Raspberry Pi Pico W | RP2040 + CYW43439 Wi-Fi | 1 | Main MCU + wireless |
| NPK Soil Sensor | RS485, 3-in-1 NPK | 1 | Soil nutrient sensing |
| RS485 to TTL Module | MAX485 | 1 | UART level converter |
| DHT11 or DHT22 | ±2°C / ±5%RH | 1 | Temperature & humidity |
| Resistive Moisture Probe | Capacitive-resistive pair | 1 | Soil moisture |
| Breadboard | 830-point solderless | 1 | Prototyping |
| Dupont Jumper Wires | M-M, M-F, F-F assorted | ~30 | Connections |
| PET Bottle (1.5L) | Recycled | 1 | Field enclosure |
| USB-A to Micro-USB Cable | 1m | 1 | Power & serial |
| Host Computer | Windows 10/11, Python 3.10+ | 1 | Receiver + dashboard |

**Estimated BOM Cost (INR):** ₹1,200 – ₹1,800 (excluding host computer)

---

## 10. INSTALLATION & OPERATION

### Prerequisites

```bash
python --version   # Python 3.10 or newer required
pip install streamlit pandas
```

### Step 1 — Flash Pico W

1. Flash MicroPython firmware to the Pico W.
2. Upload the sensor transmission script via Thonny IDE.
3. Configure the target IP address (host computer's LAN IP) and port `8000` in the Pico W script.
4. Power the Pico W; it will begin transmitting sensor data over Wi-Fi.

### Step 2 — Start the Receiver (Terminal 1)

```powershell
cd FertilizerAI\hardware
python main1.py
```

### Step 3 — Launch the Dashboard (Terminal 2)

```powershell
cd FertilizerAI
streamlit run ..\app.py
```

Navigate to `http://localhost:8501` in a browser.

### Step 4 — Configure Farm Details

Use the **left sidebar** to set District, Soil Color, pH, Rainfall, Crop, and Growth Stage.

### Step 5 — Run AI Prediction

Press **"🔬 Predict Fertilizer"** to invoke the ML model on the latest sensor snapshot.

---

## 11. FILE STRUCTURE

```
FertilizerAI/                          ← Outer project root
├── app.py                             ← Streamlit dashboard entrypoint
├── create_project.py                  ← Project scaffolding utility
├── hardware/
│   ├── main1.py                       ← UDP receiver daemon
│   └── sensor_log.csv                 ← Live sensor log
└── FertilizerAI/                      ← Python ML project package
    ├── README.md                      ← This document
    ├── requirements.txt
    ├── api/                           ← REST API layer (future)
    ├── configs/                       ← Model & pipeline configurations
    ├── data/                          ← Raw and processed datasets
    ├── docs/                          ← Additional documentation
    ├── logs/                          ← Training and inference logs
    ├── models/                        ← Serialized model artifacts
    ├── notebooks/                     ← Exploratory data analysis
    ├── outputs/                       ← Prediction output files
    ├── src/
    │   ├── data/                      ← Data loading utilities
    │   ├── features/                  ← Feature engineering
    │   ├── models/
    │   │   └── predict.py             ← predict_fertilizer() entry point
    │   ├── pipelines/                 ← Training pipelines
    │   ├── utils/                     ← Shared utilities
    │   └── visualization/             ← Chart generation
    └── tests/                         ← Unit and integration tests
```

---

## 12. LIMITATIONS & FUTURE WORK

| Limitation | Proposed Resolution |
|---|---|
| CSV file-level concurrency (no locking) | Migrate to SQLite WAL mode or InfluxDB |
| Single sensor node (no spatial coverage) | Deploy sensor mesh across field zones |
| Offline prediction only | REST API endpoint + mobile app integration |
| No pump actuator integration | GPIO-controlled peristaltic pump driver on Pico W |
| No OTA firmware updates | MicroPython OTA update via HTTP |
| Prediction requires manual button press | Threshold-triggered auto-prediction |
| No user authentication on dashboard | Streamlit-Authenticator or reverse proxy auth |
| Model not publicly validated | Multi-season field trials across Vellore district |

---

## 13. LICENSE

```
Copyright (c) 2026 Prachikk101 and Contributors

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<div align="center">

**— END OF TECHNICAL DISCLOSURE DOCUMENT —**

*FertilizerAI · Precision Agriculture through Real-Time Soil Intelligence*  
*Vellore, Tamil Nadu, India · October 2026*

</div>
