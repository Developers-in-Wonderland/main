/*
 * Robot Arm Controller (PCA9685, Safe CSV Parser, Per-Axis Limits)
 * - Input: "d0,d1,d2,d3,d4,d5,delay\n"
 *   d0..d5 = tick increments per axis [-50..+50]
 *   delay  = step delay in ms [10..5000]
 * - Improvements:
 *   • Safe parser (strtok_r + trimming + CRLF/space tolerant)
 *   • Overflow flush if line > buffer
 *   • Serial timeout shortened for responsiveness
 *   • Per-axis degree limits + (optional) per-axis pulse range scaffold
 *   • Saturation: warn & clamp per-axis (no global abort)
 *   • State sync (degree↔pulse) consistent
 *   • begin() return-type agnostic (Adafruit lib often returns void)
 *   • test() applies clamped PWM once at limit before exit
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <string.h>   // strtok_r
#include <stdlib.h>   // strtol
#include <ctype.h>    // isspace

// ---------------------- CONFIG ----------------------
#define AXIS_COUNT 6

// Default pulse range (tick) ~ 500–2500us at 50Hz; adjust per axis after calibration
static const int SERVOMIN_TICK[AXIS_COUNT] = {102,102,102,102,102,102};
static const int SERVOMAX_TICK[AXIS_COUNT] = {512,512,512,512,512,512};

// Per-axis degree limits (mechanical range)
static const int AXIS_MIN_DEG[AXIS_COUNT] = {  0,  0, 30,  0, 30, 20};
static const int AXIS_MAX_DEG[AXIS_COUNT] = {180, 90,150,120,150,160};

// Global mapping range (logical)
#define DEGREE_MIN 0
#define DEGREE_MAX 180

// Input buffer (longer than before to reduce wrap risk)
#define INPUT_BUFFER_SIZE 128
char inputBuffer[INPUT_BUFFER_SIZE];

// Step delay constraints (avoid 0ms to reduce heat/noise bursts)
#define MIN_DELAY_MS 10
#define MAX_DELAY_MS 5000

// Tick increment constraints per command
#define MIN_TICK_STEP -50
#define MAX_TICK_STEP  50

// ---------------------- STATE ----------------------
int motor_vals[AXIS_COUNT + 1] = {0};           // d0..d5 + delay
int curr_degree[AXIS_COUNT]    = {100, 55, 100, 100, 100, 150};
int curr_pulse[AXIS_COUNT]     = {0, 0, 0, 0, 0, 0};

Adafruit_PWMServoDriver pca9685 = Adafruit_PWMServoDriver(0x40);

// ---------------------- UTILITIES ----------------------
static inline int clampInt(int v, int lo, int hi) { return (v < lo) ? lo : (v > hi ? hi : v); }

static inline void rtrim_inplace(char* s) {
  if (!s) return;
  int len = (int)strlen(s);
  while (len > 0 && (isspace((unsigned char)s[len-1]) || s[len-1] == '\r')) {
    s[--len] = '\0';
  }
}
static inline char* ltrim_ptr(char* s) {
  if (!s) return s;
  while (*s && isspace((unsigned char)*s)) ++s;
  return s;
}

// degree↔tick mapping with per-axis pulse tables
int degToPulse(int axis, int degree) {
  degree = clampInt(degree, DEGREE_MIN, DEGREE_MAX);
  int minT = SERVOMIN_TICK[axis];
  int maxT = SERVOMAX_TICK[axis];
  long pulse = map(degree, DEGREE_MIN, DEGREE_MAX, minT, maxT);
  return clampInt((int)pulse, minT, maxT);
}
int pulseToDeg(int axis, int pulse) {
  int minT = SERVOMIN_TICK[axis];
  int maxT = SERVOMAX_TICK[axis];
  pulse = clampInt(pulse, minT, maxT);
  long deg = map(pulse, minT, maxT, DEGREE_MIN, DEGREE_MAX);
  return clampInt((int)deg, DEGREE_MIN, DEGREE_MAX);
}

// per-axis clamping helpers (return true if value was inside before clamping)
bool clampDegreeAxis(int axis, int &deg) {
  int old = deg;
  deg = clampInt(deg, AXIS_MIN_DEG[axis], AXIS_MAX_DEG[axis]);
  return (deg == old);
}
bool clampPulseAxis(int axis, int &pulse) {
  int old = pulse;
  int minT = SERVOMIN_TICK[axis], maxT = SERVOMAX_TICK[axis];
  pulse = clampInt(pulse, minT, maxT);
  return (pulse == old);
}

// Sync degrees → pulses safely using per-axis tables and axis limits
void syncDegreeAndPulse() {
  for (int i = 0; i < AXIS_COUNT; ++i) {
    clampDegreeAxis(i, curr_degree[i]);
    curr_pulse[i] = degToPulse(i, curr_degree[i]);
  }
}

// Apply degree to servo (also sync pulse state to keep consistent)
void setServoAngle(int axis) {
  int pulse = degToPulse(axis, curr_degree[axis]);
  curr_pulse[axis] = pulse;
  pca9685.setPWM(axis, 0, pulse);
}

// Initial positioning
void firstSet() {
  for (int i = 0; i < AXIS_COUNT; ++i) setServoAngle(i);
  syncDegreeAndPulse();
}

// Flush residual bytes until newline (used when line > buffer)
void flushUntilNewline() {
  unsigned long start = millis();
  while (millis() - start < 10) { // short burst flush
    if (!Serial.available()) break;
    int c = Serial.read();
    if (c == '\n') break;
  }
}

// ---------------------- CORE LOGIC ----------------------
// Parse one CSV line into motor_vals[], return true if valid
bool parseSerialInput() {
  if (!Serial.available()) return false;

  // Read one line
  int bytesRead = Serial.readBytesUntil('\n', inputBuffer, INPUT_BUFFER_SIZE - 1);
  inputBuffer[bytesRead] = '\0';

  if (bytesRead == 0) return false;

  // If buffer filled and no newline, flush remainder of the line
  if (bytesRead == (INPUT_BUFFER_SIZE - 1) && inputBuffer[bytesRead - 1] != '\n') {
    flushUntilNewline();
  }

  // Reset outputs
  for (int i = 0; i < AXIS_COUNT + 1; ++i) motor_vals[i] = 0;

  // Tokenize CSV
  char *saveptr = nullptr;
  char *tok = strtok_r(inputBuffer, ",", &saveptr);
  uint8_t idx = 0;

  while (tok && idx < AXIS_COUNT + 1) {
    // Trim token (leading/trailing spaces, trailing \r)
    tok = ltrim_ptr(tok);
    rtrim_inplace(tok);

    // Strict number parse
    char *endptr = nullptr;
    long v = strtol(tok, &endptr, 10);
    if (endptr && *endptr != '\0') {
      Serial.print("Error: Invalid number at index "); Serial.print(idx);
      Serial.print(" ["); Serial.print(tok); Serial.println("]");
      return false;
    }

    if (idx < AXIS_COUNT) {
      if (v < MIN_TICK_STEP || v > MAX_TICK_STEP) {
        Serial.print("Error: Tick step out of range [-50,50]: "); Serial.println(v);
        return false;
      }
    } else {
      if (v < MIN_DELAY_MS || v > MAX_DELAY_MS) {
        Serial.print("Warning: Delay clamped from "); Serial.print(v);
        Serial.print(" to ");
        v = clampInt((int)v, MIN_DELAY_MS, MAX_DELAY_MS);
        Serial.println(v);
      }
    }

    motor_vals[idx++] = (int)v;
    tok = strtok_r(nullptr, ",", &saveptr);
  }

  if (idx != AXIS_COUNT + 1) {
    Serial.print("Error: Expected 7 values, got "); Serial.println(idx);
    return false;
  }

  return true;
}

// Move one step based on motor_vals[]; returns true if any axis moved
bool moving() {
  bool anyMoved = false;

  for (int i = 0; i < AXIS_COUNT; ++i) {
    int npulse = curr_pulse[i] + motor_vals[i];

    if (!clampPulseAxis(i, npulse)) {
      Serial.print("Warn: Pulse clamped (axis "); Serial.print(i);
      Serial.print(") -> "); Serial.println(npulse);
    }

    int ndeg = pulseToDeg(i, npulse);

    if (!clampDegreeAxis(i, ndeg)) {
      Serial.print("Warn: Degree clamped (axis "); Serial.print(i);
      Serial.print(") -> "); Serial.println(ndeg);
      // Recompute pulse to match clamped degree
      npulse = degToPulse(i, ndeg);
    }

    // Apply if changed
    if (npulse != curr_pulse[i]) {
      curr_pulse[i]  = npulse;
      curr_degree[i] = ndeg;
      pca9685.setPWM(i, 0, npulse);
      anyMoved = true;
    }
  }

  int delayTime = clampInt(motor_vals[AXIS_COUNT], MIN_DELAY_MS, MAX_DELAY_MS);
  delay(delayTime);
  return anyMoved;
}

// 입력할 때, 0,0,0,0,0,0,100 이런식으로 입력하면 100ms delay로 아무 움직임 없이 대기
// 입력할 떄, 0,+3,0,0,0,0,100 이런식으로 입력하면 2축이 +3틱 움직이고 100ms delay
void moveArm() { 
  if (!parseSerialInput()) return;

  // Optional echo (can be muted after debugging)
  Serial.print("Command: ");
  for (int i = 0; i < AXIS_COUNT + 1; ++i) {
    Serial.print(motor_vals[i]);
    if (i < AXIS_COUNT) Serial.print(",");
  }
  Serial.println();

  bool ok = moving();

  if (ok) Serial.println("Movement applied");
  else    Serial.println("No movement (already at limits)");

  // Status
  Serial.print("Current degrees: ");
  for (int i = 0; i < AXIS_COUNT; ++i) {
    Serial.print(curr_degree[i]);
    if (i < AXIS_COUNT - 1) Serial.print(",");
  }
  Serial.println();
}

// Safe test on axis 0: move to max, then to min, then back to start
void test() {
  Serial.println("Starting safe test for axis 0");
  int axis = 0;

  int startPulse = curr_pulse[axis];
  int maxPulse = degToPulse(axis, AXIS_MAX_DEG[axis]);
  int minPulse = degToPulse(axis, AXIS_MIN_DEG[axis]);

  Serial.print("Moving from "); Serial.print(curr_degree[axis]);
  Serial.print(" to "); Serial.println(AXIS_MAX_DEG[axis]);

  while (curr_pulse[axis] < maxPulse) {
    curr_pulse[axis]++;
    clampPulseAxis(axis, curr_pulse[axis]);
    curr_degree[axis] = pulseToDeg(axis, curr_pulse[axis]);
    pca9685.setPWM(axis, 0, curr_pulse[axis]);

    Serial.print("Degree: "); Serial.print(curr_degree[axis]);
    Serial.print(", Pulse: "); Serial.println(curr_pulse[axis]);
    delay(20);
    if (curr_pulse[axis] >= maxPulse) break;
  }

  delay(500);

  Serial.print("Moving from "); Serial.print(curr_degree[axis]);
  Serial.print(" to "); Serial.println(AXIS_MIN_DEG[axis]);

  while (curr_pulse[axis] > minPulse) {
    curr_pulse[axis]--;
    clampPulseAxis(axis, curr_pulse[axis]);
    curr_degree[axis] = pulseToDeg(axis, curr_pulse[axis]);
    pca9685.setPWM(axis, 0, curr_pulse[axis]);

    Serial.print("Degree: "); Serial.print(curr_degree[axis]);
    Serial.print(", Pulse: "); Serial.println(curr_pulse[axis]);
    delay(20);
    if (curr_pulse[axis] <= minPulse) break;
  }

  delay(500);

  Serial.println("Returning to start position");
  curr_pulse[axis]  = startPulse;
  curr_degree[axis] = pulseToDeg(axis, curr_pulse[axis]);
  pca9685.setPWM(axis, 0, curr_pulse[axis]);
  Serial.println("Test completed");
}

// ---------------------- SETUP/LOOP ----------------------
void setup() {
  Serial.begin(115200);
  // Shorter timeout for responsiveness (default is 1000ms)
  Serial.setTimeout(30);

  while (!Serial) { ; }

  Serial.println("Initializing Robot Arm Controller...");

  // Adafruit PWM Servo Driver init
  // Many library versions: begin() returns void; avoid boolean check
  pca9685.begin();
  pca9685.setPWMFreq(50);  // 50~60Hz typical for hobby servos
  Serial.println("PCA9685 initialized");

  // Initial position
  firstSet();
  syncDegreeAndPulse();

  Serial.println("Robot arm ready!");
  Serial.println("Commands: d0,d1,d2,d3,d4,d5,delay (tick incs, delay ms)");
  Serial.println("Tick step range: [-50..+50], Delay: [10..5000] ms");
  Serial.print("Current position (deg): ");
  for (int i = 0; i < AXIS_COUNT; ++i) {
    Serial.print(curr_degree[i]);
    if (i < AXIS_COUNT - 1) Serial.print(",");
  }
  Serial.println();

  delay(300);
}

void loop() {
  moveArm();
  // You can call test() manually from setup() for validation
  // test(); # 각도 1틱씩 움직이면서 축 한계, 방향성 테스트
}
