
import csv
import json
import os
import socket
import sys
import traceback
from datetime import datetime

# =====================================================
# PROJECT PATHS
# =====================================================
HARDWARE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(
    os.path.join(HARDWARE_DIR, "..", "FertilizerAI")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.predict import predict_fertilizer

# =====================================================
# CONFIGURATION
# =====================================================
HOST = "0.0.0.0"
PORT = 8000

LOG_FILE = os.path.join(HARDWARE_DIR, "sensor_log.csv")

# Replace these example values with actual field information.
DISTRICT = "Vellore"
SOIL_COLOR = "Red"
CROP = "Rice"
GROWTH_STAGE = "Vegetative"
PH_VALUE = 6.5
RAINFALL_MM = 900

FIELDS = [
    "moisture_raw",
    "moisture_voltage",
    "temperature",
    "humidity",
    "nitrogen",
    "phosphorus",
    "potassium",
]


# =====================================================
# HELPER FUNCTIONS
# =====================================================
def fmt(value, spec=".2f"):
    if value is None:
        return "N/A"
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def number_or_none(value):
    if value is None:
        return None
    try:
        number = float(value)
        if not (number == number and abs(number) != float("inf")):
            return None
        return number
    except (TypeError, ValueError):
        return None


# =====================================================
# MAIN RECEIVER
# =====================================================
def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        sock.bind((HOST, PORT))
    except OSError as error:
        print(f"ERROR: Cannot listen on UDP port {PORT}")
        print(error)
        sock.close()
        return

    print("=" * 55)
    print("FERTILIZER AI - LIVE SENSOR RECEIVER")
    print("=" * 55)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Listening on UDP port: {PORT}")
    print(f"CSV file: {LOG_FILE}")
    print("Waiting for Pico W sensor data...")
    print("Press Ctrl+C to stop.")
    print("=" * 55)

    try:
        new_file = (
            not os.path.exists(LOG_FILE)
            or os.path.getsize(LOG_FILE) == 0
        )

        with open(
            LOG_FILE, "a", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)

            if new_file:
                writer.writerow(["timestamp"] + FIELDS)
                file.flush()

            while True:
                packet, addr = sock.recvfrom(4096)

                try:
                    sensor_data = json.loads(packet.decode("utf-8"))
                except (ValueError, UnicodeDecodeError) as error:
                    print(f"Invalid JSON from {addr}: {error}")
                    continue

                if not isinstance(sensor_data, dict):
                    print("Ignoring packet: JSON must be an object.")
                    continue

                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                print("\n" + "-" * 55)
                print(f"[{now}] Data received from {addr[0]}")
                print("-" * 55)

                print("Moisture raw:",
                      fmt(sensor_data.get("moisture_raw"), "g"))
                print("Moisture voltage:",
                      fmt(sensor_data.get("moisture_voltage"), ".3f"), "V")
                print("Temperature:",
                      fmt(sensor_data.get("temperature")), "C")
                print("Humidity:",
                      fmt(sensor_data.get("humidity")), "%RH")
                print("Nitrogen:",
                      fmt(sensor_data.get("nitrogen")))
                print("Phosphorus:",
                      fmt(sensor_data.get("phosphorus")))
                print("Potassium:",
                      fmt(sensor_data.get("potassium")))

                # Save readings
                writer.writerow(
                    [now] + [sensor_data.get(field) for field in FIELDS]
                )
                file.flush()
                print("CSV: Sensor data saved.")

                # Prepare ML input
                nitrogen = number_or_none(sensor_data.get("nitrogen"))
                phosphorus = number_or_none(sensor_data.get("phosphorus"))
                potassium = number_or_none(sensor_data.get("potassium"))
                temperature = number_or_none(sensor_data.get("temperature"))

                print("\n===== FERTILIZER AI PREDICTION =====")

                if any(
                    value is None or value <= 0
                    for value in (nitrogen, phosphorus, potassium)
                ):
                    print("Prediction skipped: NPK readings are missing, "
                          "zero, or invalid.")
                    continue

                if temperature is None:
                    print("Prediction skipped: temperature is invalid.")
                    continue

                model_input = {
                    "District_Name": DISTRICT,
                    "Soil_color": SOIL_COLOR,
                    "Nitrogen": nitrogen,
                    "Phosphorus": phosphorus,
                    "Potassium": potassium,
                    "pH": PH_VALUE,
                    "Rainfall": RAINFALL_MM,
                    "Temperature": temperature,
                    "Crop": CROP,
                    "Growth_Stage": GROWTH_STAGE,
                }

                try:
                    prediction = predict_fertilizer(model_input)

                    if not isinstance(prediction, dict):
                        print("Unexpected model result:", repr(prediction))
                        continue

                    output_fields = {
                        "Recommended fertilizer": "Recommended_Fertilizer",
                        "Primary amount (kg/hectare)": (
                            "Primary_Amount_kg_per_hectare"
                        ),
                        "Water (liters)": "Water_Liters",
                        "Pump 1 (ml)": "Pump1_ml",
                        "Pump 2 (ml)": "Pump2_ml",
                        "Pump 3 (ml)": "Pump3_ml",
                    }

                    for label, key in output_fields.items():
                        print(f"{label}: {prediction.get(key, 'N/A')}")

                    print("ML prediction completed successfully.")

                except FileNotFoundError as error:
                    print("ML prediction failed: required file not found.")
                    print("Missing file:", error.filename or str(error))
                    print("Check file paths in src/models/predict.py")
                    traceback.print_exc()

                except Exception:
                    print("ML prediction failed:")
                    traceback.print_exc()

    except KeyboardInterrupt:
        print("\nReceiver stopped by user.")

    finally:
        sock.close()
        print("UDP socket closed.")


if __name__ == "__main__":
    main()