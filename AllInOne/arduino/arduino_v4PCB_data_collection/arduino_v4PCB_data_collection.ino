// Author: Christian Shamo
// Date: 06/21/2024
// Description: This code interfaces with the Rodney V4 PCB, using a PCA9546A multiplexer to switch between 
// four ADS1115 ADCs connected via I2C (SCL/SDA pins). Each ADC measures the differential voltage (A0-A1) 
// from a wheatstone half bridge. The code reads these differentials as fast as possible in continuous mode 
// and sends them to the serial port for a Python script to process. A button toggles data collection: first 
// press starts it (sends '#'), second press stops it (sends "test ended"). If all readings are zero, it 
// reinitializes the I2C and ADCs. Runs standalone after flashing; only the Python script needs to be open.

#include <Wire.h>              // Library for I2C communication
#include <Adafruit_ADS1X15.h>  // Library for ADS1115 ADCs

#define PCA9546A_ADDR 0x70     // I2C address of the PCA9546A multiplexer

#define START_SIGNAL '#'       // Signal to start data collection in Python
#define RESET_SIGNAL '$'       // Signal to reset ADCs if all readings are zero

// Pin and variable definitions
const int BUTTON_PIN = 2;      // Pin for the start/stop button
bool data_collection_active = false;  // Tracks if Python should store data
unsigned long time_init;       // Start time for data collection (microseconds)
int16_t diff0, diff1, diff2, diff3;   // Differential readings from ADCs (bus 0-3)

// ADC instances for each multiplexer bus
Adafruit_ADS1115 ads1, ads2, ads3, ads4;

// Selects a bus (0-3) on the PCA9546A multiplexer
void pca9546a_select(uint8_t bus) {
  if (bus > 3) return;          // Validate bus number
  Wire.beginTransmission(PCA9546A_ADDR);
  Wire.write(1 << bus);         // Set the corresponding bus bit
  Wire.endTransmission();
}

// Initializes all ADS1115 ADCs on their respective buses
void initializeADS() {
  // Bus 0: ADC 1
  pca9546a_select(0);
  if (!ads1.begin()) {
    Serial.println("Failed to initialize ADS1115 on bus 0");
    while (1);  // Halt if initialization fails
  }
  ads1.setDataRate(RATE_ADS1115_860SPS);      // Max speed: 860 samples per second
  ads1.setGain(GAIN_SIXTEEN);                 // Max sensitivity (adjust per signal range)
  ads1.startADCReading(ADS1X15_REG_CONFIG_MUX_DIFF_0_1, true);  // Continuous A0-A1 differential

  // Bus 1: ADC 2
  pca9546a_select(1);
  if (!ads2.begin()) {
    Serial.println("Failed to initialize ADS1115 on bus 1");
    while (1);
  }
  ads2.setDataRate(RATE_ADS1115_860SPS);
  ads2.setGain(GAIN_SIXTEEN);
  ads2.startADCReading(ADS1X15_REG_CONFIG_MUX_DIFF_0_1, true);

  // Bus 2: ADC 3
  pca9546a_select(2);
  if (!ads3.begin()) {
    Serial.println("Failed to initialize ADS1115 on bus 2");
    while (1);
  }
  ads3.setDataRate(RATE_ADS1115_860SPS);
  ads3.setGain(GAIN_SIXTEEN);
  ads3.startADCReading(ADS1X15_REG_CONFIG_MUX_DIFF_0_1, true);

  // Bus 3: ADC 4
  pca9546a_select(3);
  if (!ads4.begin()) {
    Serial.println("Failed to initialize ADS1115 on bus 3");
    while (1);
  }
  ads4.setDataRate(RATE_ADS1115_860SPS);
  ads4.setGain(GAIN_SIXTEEN);
  ads4.startADCReading(ADS1X15_REG_CONFIG_MUX_DIFF_0_1, true);
}

void setup() {
  Serial.begin(115200);         // Start serial at 115200 baud
  Wire.setClock(400000);        // Set I2C to 400 kHz (ADS1115 max)
  Wire.begin();                 // Initialize I2C

  pinMode(BUTTON_PIN, INPUT_PULLUP);  // Button with internal pull-up
  initializeADS();              // Set up all ADCs
}

void loop() {
  // Read latest differential from each ADC
  pca9546a_select(0);
  diff0 = ads1.getLastConversionResults();
  
  pca9546a_select(1);
  diff1 = ads2.getLastConversionResults();
  
  pca9546a_select(2);
  diff2 = ads3.getLastConversionResults();
  
  pca9546a_select(3);
  diff3 = ads4.getLastConversionResults();

  // Start data collection on first button press
  if (digitalRead(BUTTON_PIN) == LOW && !data_collection_active) {
    Serial.println(START_SIGNAL);    // Signal Python to start
    data_collection_active = true;
    delay(50);                       // Debounce (reduced from 750ms)
    time_init = micros();            // Record start time
  }

  // Stop data collection on second button press
  if (digitalRead(BUTTON_PIN) == LOW && data_collection_active) {
    Serial.println("test ended");    // Signal Python to stop
    data_collection_active = false;
    delay(50);                       // Debounce
  }

  // Reinitialize if all readings are zero (error condition)
  if (diff0 == 0 && diff1 == 0 && diff2 == 0 && diff3 == 0) {
    Wire.end();
    delay(10);
    Wire.begin();
    initializeADS();
    Serial.print(RESET_SIGNAL);
    Serial.print(",");
    Serial.println(micros() - time_init);
    return;  // Exit loop to recover
  }

  // Output data: time (micros), diff0, diff2, diff1, diff3
  Serial.print(micros() - time_init);
  Serial.print(",");
  Serial.print(diff0);
  Serial.print(",");
  Serial.print(diff3);
  Serial.print(",");
  Serial.print(diff0);
  Serial.print(",");
  Serial.println(diff3);
}