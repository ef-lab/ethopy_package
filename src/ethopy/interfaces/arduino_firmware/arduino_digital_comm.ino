/*
 * Ethopy Arduino Firmware - Digital Communication Version
 * 
 * Simple analog-to-digital conversion for behavioral experiments:
 * - Reads analog sensors (lick detectors, proximity sensors)
 * - Outputs digital signals for detected states
 * - Hardware calibration with EEPROM storage
 * 
 * Use this version for basic setups that don't need computer communication.
 */

#include <EEPROM.h>

// Pin assignments
const int LICK_PIN_1 = 8;      // output Port 1
const int LICK_PIN_2 = 9;      // output Port 2
const int CENTER_PIN = 10;     // output Center Port
const int INTERRUPT_PIN = 3;   // interrupt pin

// Analog input pins
const int ANALOG_CENTER = A0;  // Analog input for center sensor
const int ANALOG_LICK_1 = A2;  // Analog input for lick detector 1
const int ANALOG_LICK_2 = A1;  // Analog input for lick detector 2

// EEPROM addresses for saving the the calibration thresholds
const int EEPROM_LICK_1 = 0;
const int EEPROM_LICK_2 = 2;
const int EEPROM_CENTER = 4;

// Global variables - Sensor thresholds (stored in EEPROM)
int threshold_center;    // Threshold value for center/proximity sensor detection
int threshold_lick_1;    // Threshold value for lick detector 1 activation
int threshold_lick_2;    // Threshold value for lick detector 2 activation

// Current sensor states (true when sensor is activated)
bool low_center = false; // Current state of center/proximity sensor
bool low_lick_1 = false; // Current state of lick detector 1
bool low_lick_2 = false; // Current state of lick detector 2

// Raw analog sensor readings
int CenterInput;         // Raw analog reading from center/proximity sensor
int LickInput1;          // Raw analog reading from lick detector 1
int LickInput2;          // Raw analog reading from lick detector 2

// Temporary variable for calibration process
int calibration_value;   // Holds sensor readings during calibration

void setup() {
  // Get stored thresholds
  threshold_lick_1 = readIntFromEEPROM(EEPROM_LICK_1);
  threshold_lick_2 = readIntFromEEPROM(EEPROM_LICK_2);
  threshold_center = readIntFromEEPROM(EEPROM_CENTER);

  // Setup input/output pins
  pinMode(INTERRUPT_PIN, INPUT_PULLUP);  // sets interrupt pin
  pinMode(LICK_PIN_1, OUTPUT);  // sets the pin as output Port 1
  pinMode(LICK_PIN_2, OUTPUT);  // sets the pin as output Port 2
  pinMode(CENTER_PIN, OUTPUT);  // sets the pin as output Port Center
}

void loop() {
  // if interrupt, calibrate
  if (digitalRead(INTERRUPT_PIN) == LOW) {
    calibrate();
  }

  // Read analog values
  CenterInput = analogRead(ANALOG_CENTER);
  LickInput1 = analogRead(ANALOG_LICK_1);
  LickInput2 = analogRead(ANALOG_LICK_2);

  // Compare to stored thresholds
  low_lick_1 = LickInput1 > threshold_lick_1;
  low_lick_2 = LickInput2 > threshold_lick_2;
  low_center = CenterInput > threshold_center;

  // Write digital outputs
  digitalWrite(LICK_PIN_1, low_lick_1);
  digitalWrite(LICK_PIN_2, low_lick_2);
  digitalWrite(CENTER_PIN, low_center);
}

void calibrate() {
  // read max values to calibrate
  calibration_value = analogRead(ANALOG_LICK_1);
  writeIntIntoEEPROM(EEPROM_LICK_1, calibration_value);
  calibration_value = analogRead(ANALOG_LICK_2);
  writeIntIntoEEPROM(EEPROM_LICK_2, calibration_value);
  calibration_value = analogRead(ANALOG_CENTER);
  writeIntIntoEEPROM(EEPROM_CENTER, calibration_value);

  // Get stored thresholds
  threshold_lick_1 = readIntFromEEPROM(EEPROM_LICK_1);
  threshold_lick_2 = readIntFromEEPROM(EEPROM_LICK_2);
  threshold_center = readIntFromEEPROM(EEPROM_CENTER);
}

void writeIntIntoEEPROM(int address, int number) { 
  EEPROM.put(address, number >> 8);
  EEPROM.put(address + 1, number & 0xFF);
}

int readIntFromEEPROM(int address) {
  return (EEPROM.read(address) << 8) + EEPROM.read(address + 1);
}