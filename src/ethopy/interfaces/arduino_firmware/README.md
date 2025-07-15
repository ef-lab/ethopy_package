# Arduino Firmware for Ethopy

This directory contains two Arduino firmware options for interfacing with behavioral experiment hardware.

## Choose Your Firmware

### Option 1: arduino_digital_comm.ino
**For basic setups with digital pin outputs**

**What it does:** 
- Reads analog sensors and outputs digital HIGH/LOW signals on pins 8, 9, 10
- Perfect for connecting to Raspberry Pi GPIO or other digital input devices
- No serial communication (doesn't send data to computers via USB)
- Hardware calibration with button press
- Minimal memory usage and simple operation

### Option 2: arduino_serial_comm.ino  
**For computer-controlled experiments with serial communication**

**What it does:** 
- Everything from Option 1 PLUS full computer communication
- Sends real-time JSON messages to computer about sensor events
- Receives commands from computer to control liquid dispensers
- Digital pin outputs (same as Option 1) for immediate hardware responses
- Event detection and logging

**Use this if:**
- You want to record all animal behavior data on your computer
- You need to control liquid dispensers
- You want real-time monitoring and data logging

**Technical details:**
- Bidirectional JSON communication at 115200 baud
- Real-time event notifications (lick detection, proximity changes)
- Remote liquid dispensing control via serial commands
- State change detection to avoid message spam

## Hardware Requirements

- Arduino or compatible microcontroller
- Analog sensors (capacitive/optical lick detectors, proximity sensor)
- Liquid dispensers (solenoid valves or peristaltic pumps)
- Pushbutton for calibration
- Appropriate power supply for dispensers

## Pin Connections

```
Analog Inputs:
- A0: Center/proximity sensor
- A1: Lick detector 2  
- A2: Lick detector 1

Digital Outputs:
- Pin 8:  Lick detector 1  (on/off)
- Pin 9:  Lick detector 2 state (on/off)
- Pin 10: Center sensor state (on/off)
- Pin 4:  Liquid valve 1 control (only in in serial_comm)
- Pin 5:  Liquid valve 2 control (only in in serial_comm)

Digital Inputs:
- Pin 3:  Calibration button
```

## Installation

### For arduino_digital_comm.ino (Simple Setup)

1. **Install Arduino IDE** (version 1.8.0 or newer)
2. **Open firmware file**:
   - Open `arduino_digital_comm.ino` in Arduino IDE
3. **Upload to Arduino**:
   - Select your Arduino board type
   - Select the correct COM port
   - Click Upload

### For arduino_serial_comm.ino (Advanced Setup)

1. **Install Arduino IDE** (version 1.8.0 or newer)
2. **Install ArduinoJson library**:
   - Open Arduino IDE
   - Go to Sketch → Include Library → Manage Libraries
   - Search for "ArduinoJson"
   - Install version 6.x
3. **Open firmware file**:
   - Open `arduino_serial_comm.ino` in Arduino IDE
4. **Upload to Arduino**:
   - Select your Arduino board type
   - Select the correct COM port
   - Click Upload

## Which Firmware Should You Choose?

**Choose arduino_digital_comm.ino if:**
- You want simple analog-to-digital conversion
- You don't need computer serial communication

**Choose arduino_serial_comm.ino if:**
- You need real-time communication with a computer

## Calibration

1. **Hardware calibration**:
   - Press and hold the calibration button (Pin 3)
   - Current sensor readings become new thresholds
   - Thresholds are stored in EEPROM
   - Release button to complete calibration

2. **When to calibrate**:
   - First time setup
   - After hardware changes
   - If sensor sensitivity changes
   - Environmental condition changes


## Integration with Ethopy

### For arduino_serial_comm.ino (Serial Communication)
Works with Ethopy's `Arduino.py` interface:
- Automatic serial port detection and connection
- JSON message parsing and validation
- Event logging with precise timestamps
- Integration with experiment control workflows
- Baudrate: 115200 (configurable in code)

### For arduino_digital_comm.ino (Digital Pins)
Works with Ethopy's `RPPorts.py` interface for Raspberry Pi:
- Direct GPIO pin reading (pins 8, 9, 10)
- No serial communication overhead
- Perfect for real-time applications
- Raspberry Pi hardware SPI/I2C compatibility

## Sensor Types & Wiring

### Supported Sensor Types
- **Capacitive lick detectors**: Connect to analog inputs A1, A2
- **Optical break-beam sensors**: Use with voltage divider circuits
- **Proximity sensors**: IR or capacitive types on A0

## Testing & Troubleshooting

### Testing Without Full Setup
1. **Upload firmware** to Arduino
2. **Open Serial Monitor** (115200 baud for serial_comm version)
3. **Touch analog pins** A0, A1, A2 with finger to simulate sensors
4. **Press Pin 3 to GND** to test calibration
5. **Watch digital pins 8, 9, 10** with LEDs or multimeter

### Common Issues & Solutions
| Problem | Solution |
|---------|----------|
| Upload fails | Check board type and COM port selection |
| Sensors not responding | Verify power and ground connections |
| Thresholds incorrect | Recalibrate using button (Pin 3) |
| JSON errors | Install ArduinoJson library v6.x |
| No serial communication | Check baud rate (115200) and cable |
| False triggers | Adjust sensor distance or add noise filtering |

### Customization Options
- **Change pin assignments**: Modify `const int` declarations at top of file
- **Adjust baud rate**: Change `Serial.begin(115200)` to desired rate
- **Modify thresholds**: Use calibration button or edit EEPROM values
- **Add sensors**: Extend code following existing analog input pattern

## Safety Notes

- Always disconnect power when wiring
- Use appropriate current limiting for LEDs (220Ω resistors recommended)
- Ensure proper isolation for liquid dispensers (use relays/optocouplers)
- Test all connections before animal experiments
- Keep water and liquids away from electronics
- Use shielded cables for analog sensors to reduce noise