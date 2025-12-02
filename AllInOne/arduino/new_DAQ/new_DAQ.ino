#include <Wire.h>
#include <Adafruit_ADS1X15.h>

Adafruit_ADS1115 ads1;                  // Single ADS1115 instance

unsigned long time_init;               // Will be set in setup()
int16_t diff1, diff2, diff3, diff4;    // diff1 = A0-A1, diff2 = A2-A3 (diff3/diff4 kept for future use)

void setup() {
  Serial.begin(115200);
  Wire.setClock(400000);               // Fast I2C (400 kHz) – ADS1115 supports it
  Wire.begin();

  // Initialize the single ADS1115 (default address 0x48, change if your ADDR pin is tied elsewhere)
  if (!ads1.begin()) {
    Serial.println("Failed to initialize ADS1115! Check wiring/address.");
    while (1) delay(10);
  }

  ads1.setGain(GAIN_SIXTEEN);           // ±0.256 V range → 7.8125 µV/LSB (highest sensitivity)
  ads1.setDataRate(RATE_ADS1115_860SPS); // Maximum sampling rate

  // Start continuous conversion on channel A0–A1 first
  // ads1.startADCReading(ADS1X15_REG_CONFIG_MUX_DIFF_0_1, /*continuous=*/true);

  time_init = micros();                // Reference timestamp – set after everything is ready
}

void loop() {
  static bool readChannel01 = true;    // Alternates between the two differential pairs

  if (readChannel01) {
    // We are currently converting A0–A1
      diff1 = ads1.readADC_Differential_0_1();   // Read A0–A1 result

      // Immediately switch to A2–A3 for the next conversion
      // ads1.startADCReading(ADS1X15_REG_CONFIG_MUX_DIFF_2_3, /*continuous=*/true);
      readChannel01 = false;
  } 
  else {
    // We are currently converting A2–A3
      diff2 = ads1.readADC_Differential_2_3();   // Read A2–A3 result

      // Switch back to A0–A1 for the next conversion
      // ads1.startADCReading(ADS1X15_REG_CONFIG_MUX_DIFF_0_1, /*continuous=*/true);
      readChannel01 = true;

      // ────── Output both values with timestamp (only when we have a fresh pair) ──────
      Serial.print(micros() - time_init);
      Serial.print(",");
      Serial.print(diff1);
      Serial.print(",");
      Serial.print(diff2);
      Serial.print(",");
      Serial.print(diff1);   // placeholders – currently 0
      Serial.print(",");
      Serial.println(diff2); // placeholders – currently 0
  }
}