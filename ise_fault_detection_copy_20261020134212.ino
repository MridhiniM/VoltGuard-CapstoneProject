#include <Arduino.h>
#include <MycilaPZEM.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <math.h>

// ============================================================
// PROJECT: VOLTGUARD - ESP32 + PZEM-004T V4.0 + DS18B20
//
// PZEM LIBRARY: MycilaPZEM 8.0.5
//
// PZEM UART:
//   PZEM TX -> ESP32 GPIO 26
//   PZEM RX -> ESP32 GPIO 27
//
// DS18B20:
//   Red    -> 3V3
//   Yellow -> GPIO 4
//   Black  -> GND
//   4.7k resistor between DATA and 3V3
//
// USB SERIAL:
//   115200 baud
//
// DATA FORMAT:
//   DATA,V,I,P,PF,Wh,Hz,VA,VAr,millis,temperature_C
//
// INTERVAL:
//   2000 milliseconds
// ============================================================


// ============================================================
// 1. CONFIGURATION
// ============================================================

Mycila::PZEM pzem;

constexpr int PZEM_RX_PIN = 26;
constexpr int PZEM_TX_PIN = 27;

constexpr int DS18B20_PIN = 4;

constexpr unsigned long READ_INTERVAL_MS = 2000;


// ============================================================
// 2. DS18B20 OBJECTS
// ============================================================

OneWire oneWire(DS18B20_PIN);

DallasTemperature temperatureSensor(&oneWire);


// Most DS18B20 sensors support 12-bit conversion.
// At 12-bit resolution, conversion can take up to 750 ms.


// ============================================================
// 3. TIMING AND MEASUREMENT VARIABLES
// ============================================================

unsigned long lastReadTime = 0;

float temperatureC = NAN;


// ============================================================
// 4. SETUP
// ============================================================

void setup() {

  // USB serial connection to the computer.
  Serial.begin(115200);

  delay(1500);

  Serial.println("# ========================================");
  Serial.println("# VOLTGUARD SENSOR LOGGER");
  Serial.println("# ESP32 + PZEM-004T V4.0 + DS18B20");
  Serial.println("# MycilaPZEM 8.0.5");
  Serial.println("# ========================================");

  Serial.printf(
    "# PZEM RX: GPIO %d\n",
    PZEM_RX_PIN
  );

  Serial.printf(
    "# PZEM TX: GPIO %d\n",
    PZEM_TX_PIN
  );

  Serial.printf(
    "# DS18B20 DATA: GPIO %d\n",
    DS18B20_PIN
  );

  Serial.printf(
    "# Reading interval: %lu ms\n",
    READ_INTERVAL_MS
  );

  Serial.println(
    "# Record format: DATA,V,I,P,PF,Wh,Hz,VA,VAr,millis,temperature_C"
  );


  // ==========================================================
  // 5. INITIALIZE DS18B20
  // ==========================================================

  temperatureSensor.begin();

  const int sensorCount =
    temperatureSensor.getDeviceCount();

  Serial.printf(
    "# DS18B20 devices detected: %d\n",
    sensorCount
  );

  if (sensorCount == 0) {

    Serial.println(
      "# WARNING: No DS18B20 detected. Check wiring and pull-up resistor."
    );

  } else {

    Serial.println(
      "# DS18B20 initialization successful."
    );

  }


  // ==========================================================
  // 6. INITIALIZE PZEM CALLBACK
  // ==========================================================

  pzem.setCallback(
    [](Mycila::PZEM::EventType eventType,
       const Mycila::PZEM::Data& data) {

      // ------------------------------------------------------
      // SUCCESSFUL PZEM READING
      // ------------------------------------------------------

      if (
        eventType == Mycila::PZEM::EventType::EVT_READ
      ) {

        // Read the latest temperature measurement.
        //
        // NAN is emitted if the sensor reading is invalid.

        float temp = temperatureC;

        if (!isfinite(temp)) {
          temp = NAN;
        }

        // IMPORTANT:
        // Do not print diagnostics inside the DATA record.
        //
        // Original ten fields are retained in the same order.
        // Temperature is appended as field eleven.

        Serial.printf(
          "DATA,%.2f,%.4f,%.2f,%.4f,%lu,%.2f,%.2f,%.2f,%lu,%.2f\n",

          data.voltage,

          data.current,

          data.activePower,

          data.powerFactor,

          (unsigned long)data.activeEnergy,

          data.frequency,

          data.apparentPower,

          data.reactivePower,

          (unsigned long)millis(),

          temp
        );
      }


      // ------------------------------------------------------
      // PZEM TIMEOUT
      // ------------------------------------------------------

      else if (
        eventType ==
        Mycila::PZEM::EventType::EVT_READ_TIMEOUT
      ) {

        Serial.println(
          "# ERROR: PZEM response timeout"
        );
      }


      // ------------------------------------------------------
      // PZEM INVALID RESPONSE
      // ------------------------------------------------------

      else if (
        eventType ==
        Mycila::PZEM::EventType::EVT_READ_ERROR
      ) {

        Serial.println(
          "# ERROR: Invalid PZEM response"
        );
      }

    }
  );


  // ==========================================================
  // 7. INITIALIZE PZEM UART
  // ==========================================================

  pzem.begin(
    Serial2,
    PZEM_RX_PIN,
    PZEM_TX_PIN
  );

  Serial.println(
    "# PZEM initialization attempted."
  );

  Serial.println(
    "# Waiting for measurements..."
  );

}


// ============================================================
// 8. MAIN LOOP
// ============================================================

void loop() {

  const unsigned long currentMillis = millis();


  // ----------------------------------------------------------
  // UPDATE TEMPERATURE
  // ----------------------------------------------------------

  // requestTemperatures() blocks while the DS18B20 converts.
  // At 12-bit resolution, this can take up to 750 ms.

  static unsigned long lastTemperatureRead = 0;

  constexpr unsigned long TEMPERATURE_INTERVAL_MS = 2000;

  if (
    currentMillis - lastTemperatureRead >=
    TEMPERATURE_INTERVAL_MS
  ) {

    lastTemperatureRead = currentMillis;

    temperatureSensor.requestTemperatures();

    const float measuredTemperature =
      temperatureSensor.getTempCByIndex(0);

    if (
      measuredTemperature != DEVICE_DISCONNECTED_C &&
      isfinite(measuredTemperature) &&
      measuredTemperature >= -55.0 &&
      measuredTemperature <= 125.0
    ) {

      temperatureC = measuredTemperature;

    } else {

      temperatureC = NAN;

      Serial.println(
        "# ERROR: Invalid DS18B20 temperature reading"
      );

    }

  }


  // ----------------------------------------------------------
  // READ PZEM EVERY TWO SECONDS
  // ----------------------------------------------------------

  if (
    currentMillis - lastReadTime >=
    READ_INTERVAL_MS
  ) {

    lastReadTime = currentMillis;

    const bool success = pzem.read();

    if (!success) {

      Serial.println(
        "# ERROR: PZEM read request failed"
      );

    }

  }

}