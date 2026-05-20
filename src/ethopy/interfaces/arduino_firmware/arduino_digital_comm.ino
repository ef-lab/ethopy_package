/*
 * Ethopy Arduino Firmware for Raspberry pi setups
 *
 * Simple analog-to-digital conversion for behavioral experiments:
 *  - Reads analog sensors (lick detectors, proximity sensors)
 *  - Outputs digital signals for detected states
 *  - Hardware calibration with EEPROM storage
 *  - Automatic lick calibration based on D4/D5 input
 *
 * Use this version for basic setups that don't need computer communication.
 *
 * Updates:
 * - Center input (A0) uses non-blocking time-window average (no movingAverage() disruption)
 * - Lick threshold calibration runs for LICK_CAL_PERIOD_MS (4sec) and triggered by:
 *     D4 (LOW edge) -> calibrate A2 -> threshold_lick_1
 *     D5 (LOW edge) -> calibrate A1 -> threshold_lick_2
 * - Lick threshold computed from P10/P90 midpoint (single threshold)
 * - Threshold update only if crossings >= LICK_CAL_MIN_CROSSINGS (5sec)
 * - Aborts calibration if multiple D4/D5 inputs are received
 * - Multi-click handler for INTERRUPT_PIN (calibration button):
 *    - 1 press within 1s: calibrate center (A0 -> threshold_center)
 *    - 2 presses within 1s: set threshold_lick_1 from instantaneous A2 read
 *    - 3 presses within 1s: set threshold_lick_2 from instantaneous A1 read
 */

#include <Arduino.h>
#include <EEPROM.h>
#include <math.h>

// Pin assignments
const int LICK_PIN_1 = 8;      // output Port 1
const int LICK_PIN_2 = 9;      // output Port 2
const int CENTER_PIN = 10;     // output Center Port
const int INTERRUPT_PIN = 3;   // interrupt pin (active LOW)

// Reward trigger inputs to start lick calibration (active LOW with pullups)
const int CALIB_LICK_1_PIN = 4; // D4 -> calibrate A2 -> threshold_lick_1
const int CALIB_LICK_2_PIN = 5; // D5 -> calibrate A1 -> threshold_lick_2

// Analog input pins
const int ANALOG_CENTER = A0;  // Analog input for center sensor
const int ANALOG_LICK_1 = A2;  // Analog input for lick detector 1
const int ANALOG_LICK_2 = A1;  // Analog input for lick detector 2

// EEPROM addresses for saving the calibration thresholds
const int EEPROM_LICK_1 = 0;
const int EEPROM_LICK_2 = 2;
const int EEPROM_CENTER = 4;

// Global variables - Sensor thresholds (stored in EEPROM)
int threshold_center;    // Threshold value for center/proximity sensor detection
int threshold_lick_1;    // Threshold value for lick detector 1 activation
int threshold_lick_2;    // Threshold value for lick detector 2 activation

// Current sensor states
bool low_lick_1 = false;
bool low_lick_2 = false;

// Raw analog sensor readings
int CenterInput = 0;         // window-averaged center input
int LickInput1 = 0;
int LickInput2 = 0;

// Center hysteresis variables
float center_buffer = 0.6;   // ratio threshold must drop to turn off
bool _center = false;
bool center = false;

// Non-blocking time-window average for Center (A0)
const unsigned long CENTER_AVG_PERIOD_MS = 50;          // window length
const unsigned long CENTER_AVG_INTERVAL_MS = 2;         // sample interval within window
unsigned long centerAvgStart = 0;
unsigned long centerAvgNext = 0;
uint32_t centerSum = 0;
uint16_t centerCount = 0;

// Lick calibration settings
const unsigned long LICK_CAL_PERIOD_MS = 4000;  // window length to evaluate licking
const unsigned long LICK_CAL_INTERVAL_MS = 10;
const unsigned long LICK_CAL_MIN_CROSSINGS = 5; // minimum number of licks in order to save the threshold
const uint16_t LICK_CAL_MAX_SAMPLES = (LICK_CAL_PERIOD_MS / LICK_CAL_INTERVAL_MS) + 5;

// Calibration guarding logic (global across BOTH pins):
// - Start calibration immediately on the FIRST falling edge of D4 or D5.
// - Open a ARM_WINDOW_MS duration guard window.
// - If ANY additional activation (D4 or D5) occurs during that window,
//   the calibration result will be DISCARDED (thresholds not stored / not applied).
const unsigned long ARM_WINDOW_MS = 2000;  // Time window in which any additional activation will result in aborted calibration
const unsigned long ARM_DEBOUNCE_MS = 50;

// Calibration Interrupt pin activation settings
const unsigned long BTN_WINDOW_MS = 1000; // Calibration button press period
const unsigned long BTN_DEBOUNCE_MS = 50;

// Non-blocking lick threshold calibration state machine
static uint16_t calRaw[LICK_CAL_MAX_SAMPLES];
static uint16_t calSorted[LICK_CAL_MAX_SAMPLES];
enum CalTarget : uint8_t { CAL_NONE = 0, CAL_LICK_1 = 1, CAL_LICK_2 = 2 };

struct CalState {
  bool active = false;
  CalTarget target = CAL_NONE;
  int analogPin = A0;
  int eepromAddr = 0;
  int *thresholdPtr = nullptr;

  // Result staging (so we can decide later whether to store/apply)
  bool resultReady = false;
  int resultThreshold = 0;
  int resultCrossings = 0;
  float resultP10 = NAN;
  float resultP90 = NAN;

  unsigned long startMs = 0;
  unsigned long nextSampleMs = 0;
  uint16_t n = 0;
};

CalState cal;

// Trigger edge tracking (INPUT_PULLUP => idle HIGH, active LOW)
bool prevD4 = HIGH;
bool prevD5 = HIGH;

// Calibration guarding state machine
struct ArmState {
  bool active = false;              // guard window running
  unsigned long startMs = 0;        // window start
  unsigned long lastEdgeMs = 0;     // debounce
  uint8_t edges = 0;               // total edges seen (across BOTH pins)
  bool cancelStore = false;         // true if 2+ edges in window
};

ArmState arm;

void onAnyArmEdge(unsigned long now) {
  // Debounce
  if (now - arm.lastEdgeMs < ARM_DEBOUNCE_MS) return;
  arm.lastEdgeMs = now;

  if (!arm.active) {
    arm.active = true;
    arm.startMs = now;
    arm.edges = 1;
    arm.cancelStore = false;
  } else {
    if (arm.edges < 255) arm.edges++;
    if (arm.edges >= 2) arm.cancelStore = true;
  }
}

void serviceArmWindow() {
  if (!arm.active) return;
  unsigned long now = millis();
  if (now - arm.startMs >= ARM_WINDOW_MS) {
    // Window finished; keep cancelStore as-is for the ongoing calibration.
    arm.active = false;
  }
}

struct BtnClickState {
  bool active = false;
  unsigned long startMs = 0;
  unsigned long lastEdgeMs = 0;
  uint8_t clicks = 0;
};

BtnClickState btn;

bool prevBtn = HIGH;

void onButtonEdge(unsigned long now) {
  if (now - btn.lastEdgeMs < BTN_DEBOUNCE_MS) return;
  btn.lastEdgeMs = now;

  if (!btn.active) {
    btn.active = true;
    btn.startMs = now;
    btn.clicks = 1;
  } else {
    if (btn.clicks < 255) btn.clicks++;
  }
}

void serviceButtonClicks() {
  if (!btn.active) return;
  unsigned long now = millis();
  if (now - btn.startMs < BTN_WINDOW_MS) return;

  // Window expired: perform action based on clicks
  if (btn.clicks == 1) {
    calibrateCenterOnly();
  } else if (btn.clicks == 2) {
    int v = analogRead(ANALOG_LICK_1); // A2 (Port 1)
    threshold_lick_1 = v;
    writeIntIntoEEPROM(EEPROM_LICK_1, threshold_lick_1);
  } else if (btn.clicks >= 3) {
    int v = analogRead(ANALOG_LICK_2); // A1 (Port 2)
    threshold_lick_2 = v;
    writeIntIntoEEPROM(EEPROM_LICK_2, threshold_lick_2);
  }

  btn.active = false;
  btn.clicks = 0;
}


// Sort helper
void sortArray(uint16_t arr[], uint16_t size) {
  for (uint16_t i = 0; i < size; i++) {
    for (uint16_t j = 0; j + 1 < size - i; j++) {
      if (arr[j] > arr[j + 1]) {
        uint16_t t = arr[j];
        arr[j] = arr[j + 1];
        arr[j + 1] = t;
      }
    }
  }
}

float movingAverage(float value) {
  const byte nvalues = 100;             // Moving average window size

  static byte current = 0;            // Index for current value
  static byte cvalues = 0;            // Count of values read (<= nvalues)
  static float sum = 0;               // Rolling sum
  static float values[nvalues];

  sum += value;

  // If the window is full, adjust the sum by deleting the oldest value
  if (cvalues == nvalues)
    sum -= values[current];

  values[current] = value;          // Replace the oldest with the latest

  if (++current >= nvalues)
    current = 0;

  if (cvalues < nvalues)
    cvalues += 1;

  return sum/cvalues;
}

float percentileLinear(const uint16_t arrSorted[], uint16_t n, float pct) {
  if (n == 0) return NAN;
  if (n == 1) return arrSorted[0];

  float idx = (pct / 100.0f) * (n - 1);
  int lo = (int)floor(idx);
  int hi = (int)ceil(idx);
  if (lo == hi) return arrSorted[lo];

  float w = idx - lo;
  return arrSorted[lo] * (1.0f - w) + arrSorted[hi] * w;
}

int countCrossingsSingleThreshold(const uint16_t *raw, uint16_t n, int T) {
  if (n < 2) return 0;

  bool above = (raw[0] > (uint16_t)T);
  int crossings = 0;

  for (uint16_t i = 1; i < n; i++) {
    bool nowAbove = (raw[i] > (uint16_t)T);
    if (nowAbove != above) {
      crossings++;
      above = nowAbove;
    }
  }
  return crossings;
}

void startLickCalibration(CalTarget target) {
  if (cal.active) return; // ignore triggers while already calibrating

  cal.active = true;
  cal.target = target;
  cal.resultReady = false;
  cal.startMs = millis();
  cal.nextSampleMs = cal.startMs;
  cal.n = 0;

  if (target == CAL_LICK_1) {
    cal.analogPin = ANALOG_LICK_1;     // A2
    cal.eepromAddr = EEPROM_LICK_1;
    cal.thresholdPtr = &threshold_lick_1;
  } else if (target == CAL_LICK_2) {
    cal.analogPin = ANALOG_LICK_2;     // A1
    cal.eepromAddr = EEPROM_LICK_2;
    cal.thresholdPtr = &threshold_lick_2;
  }
}

void serviceLickCalibrationNonBlocking() {
  if (!cal.active) return;

  unsigned long now = millis();

  // sample during window
  if (now - cal.startMs < LICK_CAL_PERIOD_MS) {
    if (now >= cal.nextSampleMs) {
      if (cal.n < LICK_CAL_MAX_SAMPLES) {
        calRaw[cal.n++] = (uint16_t)analogRead(cal.analogPin);
      }
      cal.nextSampleMs += LICK_CAL_INTERVAL_MS;
    }
    return;
  }

  // finalize (single-shot computation when window ends)
  for (uint16_t i = 0; i < cal.n; i++) calSorted[i] = calRaw[i];
  sortArray(calSorted, cal.n);

  float p10 = percentileLinear(calSorted, cal.n, 10.0f);
  float p90 = percentileLinear(calSorted, cal.n, 90.0f);

  int T = (int)lround((p10 + p90) / 1.2f);
  if (T < 0) T = 0;
  if (T > 1023) T = 1023;

  int crossings = countCrossingsSingleThreshold(calRaw, cal.n, T);

  // Stage result; storing/applying is decided later (guard-window rule)
  cal.resultReady = true;
  cal.resultThreshold = T;
  cal.resultCrossings = crossings;
  cal.resultP10 = p10;
  cal.resultP90 = p90;

  // done
  cal.active = false;
  cal.target = CAL_NONE;
}

void finalizeCalibrationIfReady() {
  if (!cal.resultReady) return;

  // Store/apply only if:
  // - crossings >= 5
  // - NO second activation occurred within the 2s guard window (global across D4/D5)
  const bool pctSpreadOK = (!isnan(cal.resultP10) && !isnan(cal.resultP90)) ? (cal.resultP90 >= (cal.resultP10 * 1.10f)) : false;
  const bool allowStore = (cal.resultCrossings >= LICK_CAL_MIN_CROSSINGS) && pctSpreadOK && (arm.edges == 1) && (!arm.cancelStore) && (!arm.active);

  if (allowStore && cal.thresholdPtr != nullptr) {
    *cal.thresholdPtr = cal.resultThreshold;
    EEPROM.put(cal.eepromAddr, (uint8_t)(cal.resultThreshold >> 8));
    EEPROM.put(cal.eepromAddr + 1, (uint8_t)(cal.resultThreshold & 0xFF));
  }

  // Reset staged result and arm state for next trigger
  cal.resultReady = false;
  cal.resultP10 = NAN;
  cal.resultP90 = NAN;
  arm.edges = 0;
  arm.cancelStore = false;
}

// ------------------------------------------------------------
// EEPROM helpers (same behavior as your code)
// ------------------------------------------------------------
void writeIntIntoEEPROM(int address, int number) {
  EEPROM.put(address, (uint8_t)(number >> 8));
  EEPROM.put(address + 1, (uint8_t)(number & 0xFF));
}

int readIntFromEEPROM(int address) {
  return (EEPROM.read(address) << 8) + EEPROM.read(address + 1);
}

// Center-only calibration on INTERRUPT_PIN
void calibrateCenterOnly() {
  int calibration_value = analogRead(ANALOG_CENTER);
  writeIntIntoEEPROM(EEPROM_CENTER, calibration_value);
  threshold_center = readIntFromEEPROM(EEPROM_CENTER);
}

// ------------------------------------------------------------
// Setup / Loop
// ------------------------------------------------------------
void setup() {
  threshold_lick_1 = readIntFromEEPROM(EEPROM_LICK_1);
  threshold_lick_2 = readIntFromEEPROM(EEPROM_LICK_2);
  threshold_center = readIntFromEEPROM(EEPROM_CENTER);

  pinMode(INTERRUPT_PIN, INPUT_PULLUP);

  pinMode(CALIB_LICK_1_PIN, INPUT_PULLUP);
  pinMode(CALIB_LICK_2_PIN, INPUT_PULLUP);
  pinMode(LICK_PIN_1, OUTPUT);
  pinMode(LICK_PIN_2, OUTPUT);
  pinMode(CENTER_PIN, OUTPUT);
}

void loop() {
  // service non-blocking calibration (if active)
  serviceLickCalibrationNonBlocking();
  finalizeCalibrationIfReady();

  // handle multi-click calibration button on INTERRUPT_PIN
  bool b = digitalRead(INTERRUPT_PIN);
  unsigned long nowBtn = millis();
  if (prevBtn == HIGH && b == LOW) onButtonEdge(nowBtn);
  prevBtn = b;
  serviceButtonClicks();
  //    Start calibration immediately on FIRST activation of D4 or D5,
  //    but only STORE/APPLY if no second activation occurs on EITHER pin within 2s.
  bool d4 = digitalRead(CALIB_LICK_1_PIN);
  bool d5 = digitalRead(CALIB_LICK_2_PIN);

  unsigned long nowMs = millis();
  if (prevD4 == HIGH && d4 == LOW) {
    onAnyArmEdge(nowMs);
    // First edge starts calibration immediately
    if (arm.edges == 1 && !cal.active) startLickCalibration(CAL_LICK_1);
  }
  if (prevD5 == HIGH && d5 == LOW) {
    onAnyArmEdge(nowMs);
    if (arm.edges == 1 && !cal.active) startLickCalibration(CAL_LICK_2);
  }

  // store previous values
  prevD4 = d4;
  prevD5 = d5;

  // Close the 2s guard window when it expires
  serviceArmWindow();

  // Read analog values with moving average
  CenterInput = movingAverage(analogRead(ANALOG_CENTER));

  // Read lick channels normally (instant reads are cheap)
  LickInput1 = analogRead(ANALOG_LICK_1);
  LickInput2 = analogRead(ANALOG_LICK_2);

  // Compare to thresholds
  low_lick_1 = (LickInput1 > threshold_lick_1);
  low_lick_2 = (LickInput2 > threshold_lick_2);

  // Center hysteresis compare
  if (CenterInput > threshold_center && !center) {
    _center = true;
  } else if (CenterInput < (int)(threshold_center * center_buffer) && center) {
    _center = false;
  } else {
    _center = center;
  }

  // store previous
  center = _center;

  // Output
  digitalWrite(LICK_PIN_1, low_lick_1);
  digitalWrite(LICK_PIN_2, low_lick_2);
  digitalWrite(CENTER_PIN, _center);
}