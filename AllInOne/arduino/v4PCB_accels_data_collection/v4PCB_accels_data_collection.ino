#include <Wire.h>
#include <Adafruit_ADS1X15.h>

#define PCA9546A_ADDR 0x70
#define START_SIGNAL '#'
#define RESET_SIGNAL '$'

const int BUTTON_PIN = 2;
const int MPU1 = 0x68;
const int MPU2 = 0x69;

bool data_collection_active = false;
unsigned long time_init;
int16_t diff0, diff1, diff2, diff3;
int16_t AcX1, AcY1, AcZ1, AcX2, AcY2, AcZ2;

Adafruit_ADS1115 ads1, ads2, ads3, ads4;

void pca9546a_select(uint8_t bus) {
  if (bus > 3) return;
  Wire.beginTransmission(PCA9546A_ADDR);
  Wire.write(1 << bus);
  Wire.endTransmission();
}

void initializeADS() {
  pca9546a_select(0);
  if (!ads1.begin()) {
    Serial.println("Failed to initialize ADS1115 on bus 0");
    while (1);
  }
  ads1.setDataRate(RATE_ADS1115_860SPS);
  ads1.setGain(GAIN_SIXTEEN);
  ads1.startADCReading(ADS1X15_REG_CONFIG_MUX_DIFF_0_1, true);

  pca9546a_select(1);
  if (!ads2.begin()) {
    Serial.println("Failed to initialize ADS1115 on bus 1");
    while (1);
  }
  ads2.setDataRate(RATE_ADS1115_860SPS);
  ads2.setGain(GAIN_SIXTEEN);
  ads2.startADCReading(ADS1X15_REG_CONFIG_MUX_DIFF_0_1, true);

  pca9546a_select(2);
  if (!ads3.begin()) {
    Serial.println("Failed to initialize ADS1115 on bus 2");
    while (1);
  }
  ads3.setDataRate(RATE_ADS1115_860SPS);
  ads3.setGain(GAIN_SIXTEEN);
  ads3.startADCReading(ADS1X15_REG_CONFIG_MUX_DIFF_0_1, true);

  pca9546a_select(3);
  if (!ads4.begin()) {
    Serial.println("Failed to initialize ADS1115 on bus 3");
    while (1);
  }
  ads4.setDataRate(RATE_ADS1115_860SPS);
  ads4.setGain(GAIN_SIXTEEN);
  ads4.startADCReading(ADS1X15_REG_CONFIG_MUX_DIFF_0_1, true);
}

void readAccel(int mpu, int16_t &x, int16_t &y, int16_t &z) {
  Wire.beginTransmission(mpu);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(mpu, 6, true);
  x = Wire.read() << 8 | Wire.read();
  y = Wire.read() << 8 | Wire.read();
  z = Wire.read() << 8 | Wire.read();
}

void setup() {
  Serial.begin(115200);
  Wire.setClock(400000);
  Wire.begin();
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  initializeADS();

  Wire.beginTransmission(MPU1);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission(true);

  Wire.beginTransmission(MPU2);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission(true);
}

void loop() {
  pca9546a_select(0);
  diff0 = ads1.getLastConversionResults();
  // pca9546a_select(1);
  // diff1 = ads2.getLastConversionResults();
  diff1 = 0;
  // pca9546a_select(2);
  // diff2 = ads3.getLastConversionResults();
  diff2 = 0;
  pca9546a_select(3);
  diff3 = ads4.getLastConversionResults();

  // readAccel(MPU1, AcX1, AcY1, AcZ1);
  // readAccel(MPU2, AcX2, AcY2, AcZ2);

  if (digitalRead(BUTTON_PIN) == LOW && !data_collection_active) {
    Serial.println(START_SIGNAL);
    data_collection_active = true;
    delay(50);
    time_init = micros();
  }
  if (digitalRead(BUTTON_PIN) == LOW && data_collection_active) {
    Serial.println("test ended");
    data_collection_active = false;
    delay(50);
  }

  if (diff0 == 0 && diff1 == 0 && diff2 == 0 && diff3 == 0) {
    Wire.end();
    delay(10);
    Wire.begin();
    initializeADS();
    Serial.print(RESET_SIGNAL);
    Serial.print(",");
    Serial.println(micros() - time_init);
    return;
  }

  Serial.print(micros() - time_init); Serial.print(",");
  Serial.print(diff0); Serial.print(",");
  // Serial.print(diff1); Serial.print(",");
  // Serial.print(diff2); Serial.print(",");
  Serial.println(diff3); //Serial.print(",");
  //Serial.print(AcX1); Serial.print(",");
  //Serial.print(AcY1); Serial.print(",");
  //Serial.println(AcZ1); //Serial.print(",");
  // Serial.print(AcX2); Serial.print(",");
  // Serial.print(AcY2);  Serial.print(",");
  // Serial.println(AcZ2);
}