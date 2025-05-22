#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

int motor_vals[7] = { 0 };
int curr_degree[6] = {90, 40, 95, 87, 90, 62};

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
}

void checklimit() {
    curr_degree[0] = constrain(curr_degree[0], 0, 180);
    curr_degree[1] = constrain(curr_degree[1], 0, 90);
    curr_degree[2] = constrain(curr_degree[2], 30, 180);
    curr_degree[3] = constrain(curr_degree[3], 30, 150);
    curr_degree[4] = constrain(curr_degree[4], 30, 180);
    curr_degree[5] = constrain(curr_degree[5], 0, 120);
}

void setServoAngle(int pinNum) {
    int pulse = map(curr_degree[pinNum], DEGREE_MIN, DEGREE_MAX, SERVOMIN, SERVOMAX);
    pca9685.setPWM(pinNum, 0, pulse);
}

void moving() { // 해당 방향으로 1도 움직임

    for (int i = 0; i < 6; i++) {
        curr_degree[i] += motor_vals[i];
        curr_degree[i] = constrain(curr_degree[i], DEGREE_MIN, DEGREE_MAX);
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

}

void setup() {
    Serial.begin(115200);

    Wire.begin();
    Wire.setClock(400000);

    pca9685.begin();
    pca9685.setPWMFreq(50);

    firstSet();
    delay(50);

    // test();
}

void loop() {
    moveArm();
}
