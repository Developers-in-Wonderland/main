#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

int motor_vals[7] = { 0 };
int curr_degree[6] = {105, 45, 90, 70, 110, 110};
int curr_pulse[6] = {0, 0, 0, 0, 0, 0};

// ----- 서보 관련 상수 정의 -----
#define SERVOMIN 102
#define SERVOMAX 512
#define DEGREE_MIN 0
#define DEGREE_MAX 180

// ----- PCA9685 객체 정의 (외부에서 extern으로 참조됨) -----
Adafruit_PWMServoDriver pca9685 = Adafruit_PWMServoDriver(0x40);


void firstSet() {
    for (int i = 0 ; i < 6; i++) {
        setServoAngle(i);
    }

    for (int i = 0; i < 6; i++) {
        curr_pulse[i] = setPulse(curr_degree[i]);
    }
}

void checklimit() {
    curr_degree[0] = constrain(curr_degree[0], 0, 180);
    curr_degree[1] = constrain(curr_degree[1], 0, 90);
    curr_degree[2] = constrain(curr_degree[2], 30, 150);
    curr_degree[3] = constrain(curr_degree[3], 0, 120);
    curr_degree[4] = constrain(curr_degree[4], 30, 150);
    curr_degree[5] = constrain(curr_degree[5], 20, 160);
}

void setServoAngle(int pinNum) {
    int pulse = setPulse(curr_degree[pinNum]);
    pca9685.setPWM(pinNum, 0, pulse);
}

int setPulse(int degree) {
    int pulse = map(degree, DEGREE_MIN, DEGREE_MAX, SERVOMIN, SERVOMAX);
    return pulse;
}

int setDegree(int pulse) {
    int degree = map(pulse, SERVOMIN, SERVOMAX, DEGREE_MIN, DEGREE_MAX);
    return degree;
}

void moving() { // 해당 방향으로 1도 움직임

    for (int i = 0; i < 6; i++) {
        curr_pulse[i] += motor_vals[i];  
        checklimit();
        curr_degree[i] = setDegree(curr_pulse[i]);
        setServoAngle(i);
    }
    delay(motor_vals[6]);
}

// 시리얼 입력 기반 동작 테스트 함수
void moveArm() {
    if (Serial.available()) {
        String input = Serial.readStringUntil('\n');
        input.trim();

        uint8_t idx = 0;
        char* token = strtok((char*)input.c_str(), ",");

        while (token && idx < 7) {
            motor_vals[idx++] = atoi(token);
            token = strtok(NULL, ",");
        }

        moving();
    }
}

void test() {
    for (int i = 0; i < 300; i++) {
        curr_pulse[0]++;
        pca9685.setPWM(0, 0, curr_pulse[0]);
        curr_degree[0] = setDegree(curr_pulse[0]);
        Serial.println(curr_degree[0]);
        delay(20);
    }

    for (int i = 0; i < 300; i++) {
        curr_pulse[0]--;
        pca9685.setPWM(0, 0, curr_pulse[0]);
        curr_degree[0] = setDegree(curr_pulse[0]);
        Serial.println(curr_degree[0]);
        delay(20);
    }
}


void setup() {
    Serial.begin(115200);

    pca9685.begin();
    pca9685.setPWMFreq(50);

    firstSet();
    delay(50);
}

void loop() {
   moveArm();
}
