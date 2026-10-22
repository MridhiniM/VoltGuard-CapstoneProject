#include <Arduino.h>
#include <MycilaPZEM.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <math.h>
#include <stdio.h>
// ============================================================
// VOLTGUARD - Single ESP32 sensor reader + MQTT publisher
//
// Hardware:
//   ESP32 Dev Module
//   PZEM-004T V4.0
//   DS18B20 temperature sensor
//
// Connections:
//   PZEM TX      -> ESP32 GPIO 26 (RX)
//   PZEM RX      -> ESP32 GPIO 27 (TX)
//   DS18B20 DATA -> ESP32 GPIO 4
//   DS18B20 pull-up resistor: 4.7k ohm from DATA to 3V3
//
// Relay:
//   Songle relay GPIO 25 is not configured or changed.
//
// USB CSV output remains compatible with the Python logger:
//   DATA,V,I,P,PF,Wh,Hz,VA,VAr,millis,temperature_C
//
// MQTT topic:
//   voltguard/telemetry
//
// MQTT client ID:
//   VoltGuardESP32
// ============================================================
// ===================== NETWORK SETTINGS =====================
const char *WIFI_SSID = "Phil_Dunphy";
const char *WIFI_PASSWORD = "Krupanidhi2002";
// Debian machine running Mosquitto
IPAddress MQTT_BROKER_IP(192, 168, 29, 46);
const uint16_t MQTT_BROKER_PORT = 1883;
const char *MQTT_TOPIC = "voltguard/telemetry";
// Fixed MQTT client ID
const char *MQTT_CLIENT_ID = "VoltGuardESP32";
// =============================================================
Mycila::PZEM pzem;
// Hardware pins
constexpr int PZEM_RX_PIN = 26;
constexpr int PZEM_TX_PIN = 27;
constexpr int DS18B20_PIN = 4;
// Timing
constexpr unsigned long READ_INTERVAL_MS = 2000;
constexpr unsigned long TEMPERATURE_INTERVAL_MS = 2000;
constexpr unsigned long WIFI_RETRY_INTERVAL_MS = 10000;
constexpr unsigned long MQTT_RETRY_INTERVAL_MS = 5000;
// Sensors and network clients
OneWire oneWire(DS18B20_PIN);
DallasTemperature temperatureSensor(&oneWire);
WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);
// Timers
unsigned long lastReadTime = 0;
unsigned long lastTemperatureRead = 0;
unsigned long lastWiFiAttempt = 0;
unsigned long lastMqttAttempt = 0;
// Temperature
float temperatureC = NAN;
// Measurement availability
bool newMeasurementAvailable = false;
// Latest PZEM readings
float latestVoltage = NAN;
float latestCurrent = NAN;
float latestActivePower = NAN;
float latestPowerFactor = NAN;
float latestFrequency = NAN;
float latestApparentPower = NAN;
float latestReactivePower = NAN;
unsigned long latestEnergyWh = 0;
unsigned long latestUptimeMs = 0;
// ============================================================
// MQTT ERROR DIAGNOSTICS
// ============================================================
void printMqttError(int state) {
  Serial.printf("# MQTT connection failed; state=%d\n", state);
  switch (state) {
    case -4:
      Serial.println("# Meaning: MQTT connection timed out.");
      Serial.println("# Check broker IP, network connectivity and firewall.");
      break;
    case -3:
      Serial.println("# Meaning: MQTT connection lost.");
      break;
    case -2:
      Serial.println("# Meaning: TCP connection to MQTT broker failed.");
      Serial.println("# Check that Mosquitto is listening on port 1883.");
      break;
    case -1:
      Serial.println("# Meaning: MQTT client is disconnected.");
      break;
    case 1:
      Serial.println("# Meaning: MQTT protocol version rejected.");
      break;
    case 2:
      Serial.println("# Meaning: MQTT client ID rejected.");
      Serial.println("# This sketch uses the fixed ID VoltGuardESP32.");
      break;
    case 3:
      Serial.println("# Meaning: MQTT broker unavailable.");
      break;
    case 4:
      Serial.println("# Meaning: MQTT credentials rejected.");
      break;
    case 5:
      Serial.println("# Meaning: MQTT client not authorized.");
      break;
    default:
      Serial.println("# Meaning: Unrecognized MQTT connection state.");
      break;
  }
}
// ============================================================
// WI-FI CONNECTION
// ============================================================
void startWiFi() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }
  Serial.printf("# Connecting to Wi-Fi: %s\n", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  lastWiFiAttempt = millis();
}
void maintainWiFi() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }
  if (millis() - lastWiFiAttempt >= WIFI_RETRY_INTERVAL_MS) {
    startWiFi();
  }
}
// ============================================================
// MQTT CONNECTION
// ============================================================
void maintainMqtt() {
  if (WiFi.status() != WL_CONNECTED) {
    return;
  }
  if (mqttClient.connected()) {
    return;
  }
  if (millis() - lastMqttAttempt < MQTT_RETRY_INTERVAL_MS) {
    return;
  }
  lastMqttAttempt = millis();
  Serial.println();
  Serial.println("# ----------------------------------------");
  Serial.println("# Attempting MQTT connection");
  Serial.printf("# Broker IP: %s\n",
                MQTT_BROKER_IP.toString().c_str());
  Serial.printf("# Broker port: %u\n", MQTT_BROKER_PORT);
  Serial.printf("# Client ID: %s\n", MQTT_CLIENT_ID);
  if (mqttClient.connect(MQTT_CLIENT_ID)) {
    Serial.println("# MQTT CONNECTED SUCCESSFULLY!");
    Serial.printf("# Publishing to topic: %s\n", MQTT_TOPIC);
  } else {
    printMqttError(mqttClient.state());
    Serial.println("# Will retry MQTT connection in 5 seconds.");
  }
  Serial.println("# ----------------------------------------");
  Serial.println();
}
// ============================================================
// MQTT TELEMETRY PUBLISHER
// ============================================================
void publishLatestMeasurement() {
  if (!newMeasurementAvailable) {
    return;
  }
  if (!mqttClient.connected()) {
    return;
  }
  char payload[320];
  int length = 0;
  // Build valid JSON.
  // An unavailable DS18B20 reading is represented as JSON null.
  if (isfinite(temperatureC)) {
    length = snprintf(
      payload,
      sizeof(payload),
      "{\"voltage\":%.2f,"
      "\"current\":%.4f,"
      "\"power\":%.2f,"
      "\"powerFactor\":%.4f,"
      "\"energy\":%lu,"
      "\"frequency\":%.2f,"
      "\"apparentPower\":%.2f,"
      "\"reactivePower\":%.2f,"
      "\"uptime\":%lu,"
      "\"temperature\":%.2f}",
      latestVoltage,
      latestCurrent,
      latestActivePower,
      latestPowerFactor,
      latestEnergyWh,
      latestFrequency,
      latestApparentPower,
      latestReactivePower,
      latestUptimeMs,
      temperatureC
    );
  } else {
    length = snprintf(
      payload,
      sizeof(payload),
      "{\"voltage\":%.2f,"
      "\"current\":%.4f,"
      "\"power\":%.2f,"
      "\"powerFactor\":%.4f,"
      "\"energy\":%lu,"
      "\"frequency\":%.2f,"
      "\"apparentPower\":%.2f,"
      "\"reactivePower\":%.2f,"
      "\"uptime\":%lu,"
      "\"temperature\":null}",
      latestVoltage,
      latestCurrent,
      latestActivePower,
      latestPowerFactor,
      latestEnergyWh,
      latestFrequency,
      latestApparentPower,
      latestReactivePower,
      latestUptimeMs
    );
  }
  if (length <= 0 || length >= (int)sizeof(payload)) {
    Serial.println("# ERROR: MQTT payload buffer too small.");
    return;
  }
  if (mqttClient.publish(MQTT_TOPIC, payload)) {
    Serial.printf(
      "# MQTT published to %s: %s\n",
      MQTT_TOPIC,
      payload
    );
    newMeasurementAvailable = false;
  } else {
    Serial.println(
      "# ERROR: MQTT publish failed; reading will be retried."
    );
  }
}
// ============================================================
// SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println();
  Serial.println("# ========================================");
  Serial.println("# VOLTGUARD SENSOR + MQTT LOGGER");
  Serial.println("# ESP32 + PZEM-004T V4.0 + DS18B20");
  Serial.println("# ========================================");
  // Hardware information
  Serial.printf("# PZEM RX: GPIO %d\n", PZEM_RX_PIN);
  Serial.printf("# PZEM TX: GPIO %d\n", PZEM_TX_PIN);
  Serial.printf("# DS18B20 DATA: GPIO %d\n", DS18B20_PIN);
  Serial.printf(
    "# Reading interval: %lu ms\n",
    READ_INTERVAL_MS
  );
  // USB CSV output
  Serial.println(
    "# USB format: DATA,V,I,P,PF,Wh,Hz,VA,VAr,millis,temperature_C"
  );
  // MQTT configuration
  Serial.printf("# MQTT broker: %s:%u\n",
                MQTT_BROKER_IP.toString().c_str(),
                MQTT_BROKER_PORT);
  Serial.printf("# MQTT client ID: %s\n", MQTT_CLIENT_ID);
  Serial.printf("# MQTT topic: %s\n", MQTT_TOPIC);
  // ==========================================================
  // DS18B20 INITIALIZATION
  // ==========================================================
  temperatureSensor.begin();
  int sensorCount = temperatureSensor.getDeviceCount();
  Serial.printf(
    "# DS18B20 devices detected: %d\n",
    sensorCount
  );
  if (sensorCount == 0) {
    Serial.println(
      "# WARNING: No DS18B20 detected. Check wiring and pull-up."
    );
  }
  // ==========================================================
  // PZEM CALLBACK
  // ==========================================================
  pzem.setCallback(
    [](Mycila::PZEM::EventType eventType,
       const Mycila::PZEM::Data& data) {
      if (eventType == Mycila::PZEM::EventType::EVT_READ) {
        // Store latest measurements
        latestVoltage = data.voltage;
        latestCurrent = data.current;
        latestActivePower = data.activePower;
        latestPowerFactor = data.powerFactor;
        latestEnergyWh =
          (unsigned long)data.activeEnergy;
        latestFrequency = data.frequency;
        latestApparentPower = data.apparentPower;
        latestReactivePower = data.reactivePower;
        latestUptimeMs = millis();
        newMeasurementAvailable = true;
        // Preserve existing Python logger CSV format.
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
          temperatureC
        );
      } else if (
        eventType == Mycila::PZEM::EventType::EVT_READ_TIMEOUT
      ) {
        Serial.println("# ERROR: PZEM response timeout");
      } else if (
        eventType == Mycila::PZEM::EventType::EVT_READ_ERROR
      ) {
        Serial.println("# ERROR: Invalid PZEM response");
      }
    }
  );
  // Start PZEM serial communication
  pzem.begin(Serial2, PZEM_RX_PIN, PZEM_TX_PIN);
  Serial.println("# PZEM initialization attempted.");
  // ==========================================================
  // MQTT CLIENT CONFIGURATION
  // ==========================================================
  mqttClient.setServer(
    MQTT_BROKER_IP,
    MQTT_BROKER_PORT
  );
  mqttClient.setBufferSize(384);
  mqttClient.setKeepAlive(30);
  mqttClient.setSocketTimeout(5);
  // ==========================================================
  // START WI-FI
  // ==========================================================
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  startWiFi();
  Serial.println("# Setup complete.");
  Serial.println("# Waiting for Wi-Fi, MQTT and sensor readings.");
}
// ============================================================
// MAIN LOOP
// ============================================================
void loop() {
  // Maintain Wi-Fi and MQTT connections
  maintainWiFi();
  maintainMqtt();
  // Keep the MQTT connection alive
  if (mqttClient.connected()) {
    mqttClient.loop();
  }
  const unsigned long currentMillis = millis();
  // ==========================================================
  // READ DS18B20
  // ==========================================================
  if (
    currentMillis - lastTemperatureRead
    >= TEMPERATURE_INTERVAL_MS
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
  // ==========================================================
  // READ PZEM
  // ==========================================================
  if (
    currentMillis - lastReadTime
    >= READ_INTERVAL_MS
  ) {
    lastReadTime = currentMillis;
    const bool success = pzem.read();
    if (!success) {
      Serial.println("# ERROR: PZEM read request failed");
    }
  }
  // ==========================================================
  // PUBLISH MQTT TELEMETRY
  // ==========================================================
  publishLatestMeasurement();
  // Short delay to avoid an unnecessarily busy loop
  delay(2);
}````````````````````
