import json


# ==========================================
# SENSOR INPUT VALIDATION LIMITS
# ==========================================

LIMITS = {

    "Nitrogen": {
        "min": 0,
        "max": 200
    },

    "Phosphorus": {
        "min": 0,
        "max": 150
    },

    "Potassium": {
        "min": 0,
        "max": 200
    },

    "pH": {
        "min": 3.5,
        "max": 9.5
    },

    "Rainfall": {
        "min": 0,
        "max": 5000
    },

    "Temperature": {
        "min": 0,
        "max": 60
    }

}



# ==========================================
# MIXING OUTPUT LIMITS
# ==========================================

MIXING_LIMITS = {

    "Primary_Amount_kg_per_hectare": {

        "min": 0,
        "max": 500

    },


    "Water_Liters": {

        "min": 0,
        "max": 5000

    },


    "Pump1_ml": {

        "min": 0,
        "max": 1000

    },


    "Pump2_ml": {

        "min": 0,
        "max": 1000

    },


    "Pump3_ml": {

        "min": 0,
        "max": 1000

    }

}



# ==========================================
# VALIDATE SENSOR INPUT
# ==========================================

def validate_sensor_input(data):


    errors = []



    for feature, limits in LIMITS.items():


        if feature in data:


            value = data[feature]


            if value < limits["min"] or value > limits["max"]:

                errors.append(

                    f"{feature} value {value} is outside valid range "

                    f"({limits['min']} - {limits['max']})"

                )



    return errors




# ==========================================
# VALIDATE AI OUTPUT
# ==========================================

def validate_prediction_output(result):


    errors = []



    for parameter, limits in MIXING_LIMITS.items():


        if parameter in result:


            value = result[parameter]


            if value < limits["min"] or value > limits["max"]:


                errors.append(

                    f"{parameter} {value} exceeds safe limit "

                    f"({limits['min']} - {limits['max']})"

                )



    # Check total pump volume


    total_pump = (

        result.get("Pump1_ml",0)

        +

        result.get("Pump2_ml",0)

        +

        result.get("Pump3_ml",0)

    )



    if total_pump > 2000:


        errors.append(

            f"Total pump output too high: {total_pump} ml"

        )



    return errors




# ==========================================
# FINAL VALIDATION
# ==========================================

def validate_prediction(sensor_input, prediction):


    report = {


        "status": "SAFE",

        "sensor_errors": [],

        "prediction_errors": []

    }



    sensor_errors = validate_sensor_input(
        sensor_input
    )


    prediction_errors = validate_prediction_output(
        prediction
    )



    report["sensor_errors"] = sensor_errors

    report["prediction_errors"] = prediction_errors



    if sensor_errors or prediction_errors:


        report["status"] = "WARNING"



    return report




# ==========================================
# TEST
# ==========================================


if __name__ == "__main__":



    sensor_data = {


        "Nitrogen":80,

        "Phosphorus":40,

        "Potassium":50,

        "pH":6.5,

        "Rainfall":900,

        "Temperature":28

    }



    ai_output = {


        "Primary_Amount_kg_per_hectare":78.75,

        "Water_Liters":252.6,

        "Pump1_ml":393.77,

        "Pump2_ml":286.42,

        "Pump3_ml":25.45

    }



    result = validate_prediction(

        sensor_data,

        ai_output

    )



    print(
        json.dumps(
            result,
            indent=4
        )
    )