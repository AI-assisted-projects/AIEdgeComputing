// gesture_capture.ino -- record your own gesture data with the Arduino Nano 33 BLE Sense.
//
// Every time you move the board, it records one window (1 second, 50 readings
// of ax, ay, az [g] and gx, gy, gz [deg/s]) and prints it as CSV on the serial
// port (115200 baud), followed by an empty line. Lines starting with '#' are
// comments.
//
// Two ways to save the data:
//   * easiest: notebook 01, section "Record your own gestures", which reads the
//     serial port and writes data/recorded/<gesture>.csv for you;
//   * by hand: open the Serial Monitor, perform e.g. 30 punches, copy all the
//     text into data/recorded/punch.csv. Repeat for every gesture.
//
// Hold the board the way you will hold it when using the classifier!

// 1 = "Nano 33 BLE Sense" (LSM9DS1), 2 = "Nano 33 BLE Sense Rev2" (BMI270).
#define BOARD_REVISION 1

#if BOARD_REVISION == 2
#include <Arduino_BMI270_BMM150.h>
#else
#include <Arduino_LSM9DS1.h>
#endif

#include "gesture_recorder.h"

GestureRecorder recorder;
unsigned long n_recorded = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial) {}                    // wait until the Serial Monitor / notebook connects
  pinMode(LED_BUILTIN, OUTPUT);

  if (!IMU.begin()) {
    Serial.println("# ERROR: IMU not found. Is BOARD_REVISION set correctly?");
    while (true) {}
  }
  Serial.println("# gesture_capture ready -- move the board to record a gesture");
  Serial.println("ax,ay,az,gx,gy,gz");
}

void loop() {
  digitalWrite(LED_BUILTIN, recorder.isRecording() ? HIGH : LOW);   // LED on while recording
  if (!recorder.update()) return;

  const float* w = recorder.window();
  for (int t = 0; t < MODEL_WINDOW_SAMPLES; ++t) {
    for (int c = 0; c < MODEL_N_CHANNELS; ++c) {
      Serial.print(w[t * MODEL_N_CHANNELS + c], c < 3 ? 4 : 2);
      Serial.print(c + 1 < MODEL_N_CHANNELS ? ',' : '\n');
    }
  }
  Serial.println();
  ++n_recorded;
  Serial.print("# recorded gesture ");
  Serial.println(n_recorded);
}
