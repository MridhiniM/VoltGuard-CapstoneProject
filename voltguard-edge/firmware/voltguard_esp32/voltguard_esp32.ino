// VoltGuard ESP32 edge node
// PZEM-004T v3 on UART2 (RX2=16, TX2=17), DS18B20 on GPIO4, relay IN on GPIO25.
// Every 5 s: read sensors -> hard rules -> edge ML -> publish to MQTT.
// Hard rules ALWAYS trip locally. Edge ML trips locally only while MQTT is down;
// while online, Node-RED fuses edge ML + laptop ML and sends TRIP back.
//
// Libraries (Arduino Library Manager): PZEM004Tv30 (mandulaj), OneWire,
// DallasTemperature, PubSubClient, ArduinoJson (v7)

#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <PZEM004Tv30.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include "edge_model.h"

// ---------- EDIT THESE ----------
const char* WIFI_SSID   = "YOUR_WIFI";
const char* WIFI_PASS   = "YOUR_PASS";
const char* MQTT_HOST   = "192.168.1.10";   // your laptop's IP (Mosquitto)
const int   MQTT_PORT   = 1883;
const char* TOPIC_DATA  = "voltguard/telemetry";
const char* TOPIC_CMD   = "voltguard/cmd";
#define RELAY_ACTIVE_LOW  true   // most Songle relay modules switch ON when IN is LOW

// Hard protection limits (trip immediately, no network needed)
const float V_MAX = 253.0;   // 230 V +10%
const float V_MIN = 200.0;
const float I_MAX = 0.50;    // A, set for your load (bulb draws ~0.04 A)
const float T_MAX = 60.0;    // degC
const int   HARD_CONFIRM  = 2;   // consecutive readings before hard trip (10 s)
const int   EDGE_CONFIRM  = 6;   // consecutive edge-ML faults before offline trip (30 s)
// --------------------------------

#define PIN_DS18B20 4
#define PIN_RELAY   25
const unsigned long PERIOD_MS = 5000;   // matches the 5 s training data

PZEM004Tv30 pzem(Serial2, 16, 17);
OneWire oneWire(PIN_DS18B20);
DallasTemperature ds(&oneWire);
WiFiClient net;
PubSubClient mqtt(net);

bool tripped = false;
String tripReason = "";
int hardCount = 0, edgeCount = 0;

// ---- rolling buffers for edge features (same as ml/features.py) ----
const int W = 12, TW = 60;
float vBuf[W], iBuf[W], pBuf[W], tBuf[TW + 1];
int nW = 0, nT = 0, headW = 0, headT = 0;
float firstT = NAN;

void resetWindow() { nW = nT = headW = headT = 0; firstT = NAN; }

float stdOf(const float* b, int n) {           // sample std (ddof=1), like pandas
  if (n < 2) return 0;
  float m = 0; for (int k = 0; k < n; k++) m += b[k]; m /= n;
  float s = 0; for (int k = 0; k < n; k++) s += (b[k] - m) * (b[k] - m);
  return sqrtf(s / (n - 1));
}

int edgeInfer(float v, float i, float p, float t) {
  // v_dev uses the PREVIOUS W voltages (before adding the current one)
  float vMean = v; if (nW > 0) { vMean = 0; for (int k = 0; k < nW; k++) vMean += vBuf[k]; vMean /= nW; }
  float vDev = v - vMean;
  vBuf[headW] = v; iBuf[headW] = i; pBuf[headW] = p; headW = (headW + 1) % W; if (nW < W) nW++;

  if (isnan(firstT)) firstT = t;
  tBuf[headT] = t; headT = (headT + 1) % (TW + 1); if (nT < TW + 1) nT++;
  float tOld = (nT == TW + 1) ? tBuf[headT] : firstT;   // oldest = value 60 samples ago

  float f[8] = { v, i, p, t, vDev, stdOf(pBuf, nW), stdOf(iBuf, nW), t - tOld };
  return edgePredict(f);
}

void setRelay(bool on) {
  digitalWrite(PIN_RELAY, (on ^ RELAY_ACTIVE_LOW) ? HIGH : LOW);
}

void trip(const String& why) {
  if (tripped) return;
  tripped = true; tripReason = why; setRelay(false); resetWindow();
  Serial.println("TRIP: " + why);
}

void onCmd(char* topic, byte* payload, unsigned int len) {
  JsonDocument d;
  String cmd;
  if (deserializeJson(d, payload, len) == DeserializationError::Ok && d["cmd"].is<const char*>()) cmd = d["cmd"].as<String>();
  else { for (unsigned k = 0; k < len; k++) cmd += (char)payload[k]; }
  cmd.trim(); cmd.toUpperCase();
  if (cmd == "TRIP") trip(String("node-red: ") + (d["reason"] | "ml"));
  // Remote RESET intentionally disabled. Never remotely re-energize a faulted load.
}

void keepConnected() {                       // non-blocking; protection never waits on Wi-Fi
  static unsigned long last = 0;
  if (millis() - last < 5000) return;
  last = millis();
  if (WiFi.status() != WL_CONNECTED) { WiFi.reconnect(); return; }
  if (!mqtt.connected() && mqtt.connect("voltguard-esp32")) mqtt.subscribe(TOPIC_CMD);
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_RELAY, OUTPUT);
  tripped = true; tripReason = "startup lockout"; setRelay(false); // fail OFF at boot; do not auto-energize
  ds.begin(); ds.setWaitForConversion(true);
  WiFi.mode(WIFI_STA); WiFi.begin(WIFI_SSID, WIFI_PASS);
  mqtt.setServer(MQTT_HOST, MQTT_PORT); mqtt.setCallback(onCmd); mqtt.setBufferSize(512);
}

void loop() {
  mqtt.loop();
  keepConnected();

  static unsigned long lastRead = 0;
  if (millis() - lastRead < PERIOD_MS) return;
  lastRead = millis();

  float v = pzem.voltage(), i = pzem.current(), p = pzem.power(), pf = pzem.pf();
  ds.requestTemperatures(); float t = ds.getTempCByIndex(0);
  bool pzemOk = isfinite(v) && isfinite(i) && isfinite(p) && v >= 0 && i >= 0, tempOk = isfinite(t) && t > -55 && t < 125 && t != 85;
  if (!pzemOk) { v = i = p = 0; trip("sensor fault: PZEM"); }
  if (!tempOk) trip("sensor fault: temperature");

  // ---------- 1) hard rules (always local) ----------
  String hard = "";
  if (pzemOk && v > V_MAX) hard = "overvoltage";
  else if (pzemOk && v > 50 && v < V_MIN) hard = "undervoltage";
  else if (pzemOk && i > I_MAX) hard = "overcurrent";
  else if (tempOk && t > T_MAX) hard = "overtemperature";
  hardCount = hard.length() ? hardCount + 1 : 0;
  if (hardCount >= HARD_CONFIRM) trip("hard: " + hard);

  // ---------- 2) edge ML ----------
  const char* edgeLabel = "n/a";
  bool energized = pzemOk && tempOk && v > 50;
  if (energized && !tripped) {
    edgeLabel = EDGE_LABELS[edgeInfer(v, i, p, t)];
    edgeCount = strcmp(edgeLabel, "normal") ? edgeCount + 1 : 0;
    if (!mqtt.connected() && edgeCount >= EDGE_CONFIRM) trip(String("edge-ml offline: ") + edgeLabel);
  } else if (!energized) { resetWindow(); edgeLabel = "off"; edgeCount = 0; }

  // ---------- 3) publish ----------
  JsonDocument d;
  d["voltage"] = v; d["current"] = i; d["power"] = p; d["pf"] = pf;
  d["temperature"] = tempOk ? t : NAN;
  d["edge_label"] = edgeLabel; d["edge_streak"] = edgeCount;
  d["hard_rule"] = hard; d["relay"] = tripped ? "TRIPPED" : "ON"; d["trip_reason"] = tripReason;
  d["sensor_ok"] = pzemOk && tempOk; d["uptime_s"] = millis() / 1000;
  char buf[384]; size_t n = serializeJson(d, buf);
  if (mqtt.connected()) mqtt.publish(TOPIC_DATA, buf, n);
  Serial.println(buf);
}
