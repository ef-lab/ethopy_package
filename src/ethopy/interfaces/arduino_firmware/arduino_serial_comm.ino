/*
 * Ethopy Arduino Firmware - Serial Communication Version
 * 
 * Full-featured firmware for behavioral experiments supporting:
 * - Analog sensor reading (lick detectors, proximity sensors)
 * - Digital output control (reward delivery)
 * - Bidirectional JSON communication with host computer
 * - Hardware calibration with EEPROM storage
 * 
 * Use this version for computer-controlled experiments with real-time communication.
 * Requires ArduinoJson library.
 */

#include <EEPROM.h>
#include <ArduinoJson.h>

// Pin assignments
const int LICK_PIN_1 = 8;      // output Port 1
const int LICK_PIN_2 = 9;      // output Port 2
const int CENTER_PIN = 10;     // output Center Port
const int LIQUID_PIN_1 = 4;    // liquid dispenser 1
const int LIQUID_PIN_2 = 5;    // liquid dispenser 2
const int INTERRUPT_PIN = 3;   // interrupt pin

// Analog input pins
const int ANALOG_CENTER = A0;  // Analog input for center sensor
const int ANALOG_LICK_1 = A2;  // Analog input for lick detector 1
const int ANALOG_LICK_2 = A1;  // Analog input for lick detector 2

// EEPROM addresses
const int EEPROM_LICK_1 = 0;
const int EEPROM_LICK_2 = 2;
const int EEPROM_CENTER = 4;

// Global variables - Sensor thresholds (stored in EEPROM)
int threshold_center;    // Threshold value for center/proximity sensor detection
int threshold_lick_1;    // Threshold value for lick detector 1 activation
int threshold_lick_2;    // Threshold value for lick detector 2 activation

// Previous sensor states (for detecting state changes)
bool state_center = false; // Previous state of center/proximity sensor
bool state_lick_1 = false; // Previous state of lick detector 1  
bool state_lick_2 = false; // Previous state of lick detector 2

// Raw analog sensor readings (0-1023)
int CenterInput;         // Raw analog reading from center/proximity sensor
int LickInput1;          // Raw analog reading from lick detector 1
int LickInput2;          // Raw analog reading from lick detector 2

// Temporary variable for calibration process
int calibration_value;   // Holds sensor readings during calibration

// Communication variables for JSON messaging
StaticJsonDocument<900> outgoing; // JSON document for sending messages to host
StaticJsonDocument<900> incoming; // JSON document for receiving messages from host

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
  pinMode(LIQUID_PIN_1, OUTPUT);  // sets the pin as output Port 1
  pinMode(LIQUID_PIN_2, OUTPUT);  // sets the pin as output Port 2
  
  Serial.begin(115200);
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
  bool current_lick_1 = LickInput1 > threshold_lick_1;
  bool current_lick_2 = LickInput2 > threshold_lick_2;
  bool current_center = CenterInput > threshold_center;

  // Write digital outputs
  digitalWrite(LICK_PIN_1, current_lick_1);
  digitalWrite(LICK_PIN_2, current_lick_2);
  digitalWrite(CENTER_PIN, current_center);

  // Serial communication
  handleSerialCommunication(current_lick_1, current_lick_2, current_center);

  state_center = current_center;
  state_lick_1 = current_lick_1;
  state_lick_2 = current_lick_2;
}

void handleSerialCommunication(bool current_lick_1, bool current_lick_2, bool current_center) {
  // Process incoming commands
  if (Serial.available() > 0) {
    DeserializationError error = deserializeJson(incoming, Serial);
    if (error) {
      outgoing["msg"] = "Arduino Error: " + String(error.c_str());
      serializeJson(outgoing, Serial);
      Serial.print('\n');
      return;
    }

    // Handle different command types
    if (incoming["type"] == "pulse") {
      int port = incoming["port"];
      int duration = incoming["duration"];
      
      if (port == 1 && duration > 0) {
        pulse(LIQUID_PIN_1, duration);
      } else if (port == 2 && duration > 0) {
        pulse(LIQUID_PIN_2, duration);
      } else {
        outgoing["msg"] = "Invalid pulse command";
        serializeJson(outgoing, Serial);
        Serial.print('\n');
      }
    } else if (incoming["type"] == "proximity") {
      // Send current proximity state
      outgoing["port"] = 3;
      outgoing["type"] = "proximity";
      outgoing["state"] = current_center;
      serializeJson(outgoing, Serial);
      Serial.print('\n');
    } else if (incoming["type"] == "status") {
      // Send all current states
      outgoing["lick_1"] = current_lick_1;
      outgoing["lick_2"] = current_lick_2;
      outgoing["center"] = current_center;
      outgoing["thresholds"]["lick_1"] = threshold_lick_1;
      outgoing["thresholds"]["lick_2"] = threshold_lick_2;
      outgoing["thresholds"]["center"] = threshold_center;
      serializeJson(outgoing, Serial);
      Serial.print('\n');
    }
  }

  if (current_lick_1 && !state_lick_1) {
    outgoing["port"] = 1;
    outgoing["type"] = "lick";
    outgoing["state"] = true;
    serializeJson(outgoing, Serial);
    Serial.print('\n');
  }
  
  if (current_lick_2 && !state_lick_2) {
    outgoing["port"] = 2;
    outgoing["type"] = "lick";
    outgoing["state"] = true;
    serializeJson(outgoing, Serial);
    Serial.print('\n');
  }
  
  if (current_center != state_center) {
    outgoing["port"] = 3;
    outgoing["type"] = "proximity";
    outgoing["state"] = current_center;
    serializeJson(outgoing, Serial);
    Serial.print('\n');
  }
}

void pulse(int pin, int duration) {
  digitalWrite(pin, HIGH);
  delay(duration);
  digitalWrite(pin, LOW);
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

  // Notify host of calibration completion
  outgoing["type"] = "calibration";
  outgoing["status"] = "complete";
  outgoing["thresholds"]["lick_1"] = threshold_lick_1;
  outgoing["thresholds"]["lick_2"] = threshold_lick_2;
  outgoing["thresholds"]["center"] = threshold_center;
  serializeJson(outgoing, Serial);
  Serial.print('\n');
}

void writeIntIntoEEPROM(int address, int number) {
  EEPROM.put(address, number >> 8);
  EEPROM.put(address + 1, number & 0xFF);
}

int readIntFromEEPROM(int address) {
  return (EEPROM.read(address) << 8) + EEPROM.read(address + 1);
}