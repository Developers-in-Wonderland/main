#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// PCA9685 객체 생성 (기본 주소 0x40)
Adafruit_PWMServoDriver pca9685 = Adafruit_PWMServoDriver(0x40);

// 서보모터 동작 범위 (펄스 최소/최대 값 설정)
#define SERVOMIN 1000   // 최소 펄스 길이
#define SERVOMAX 2000  // 최대 펄스 길이

// 서보모터 클래스를 정의
class ServoMotor {
private:
    uint8_t servoPin;
    int angle;  // 현재 각도

public:
    // 생성자 (서보모터 핀 번호 설정)
    ServoMotor(uint8_t pin) {
        servoPin = pin;
        angle = 90;  // 초기 각도 90도
    }

    // 서보모터 이동 함수
    void setAngle(int value) {
        angle = constrain(angle + value, 0, 180);  // 안전한 범위 유지
        int pulse = map(angle, 0, 180, SERVOMIN, SERVOMAX); // 각도를 서보모터 내 사용 변수로 변환
        pca9685.setPWM(servoPin, 0, pulse);
    }

    // 현재 각도 반환 함수
    int getAngle() { return angle; }
};

ServoMotor baseServo(0);
ServoMotor joint1Servo1(1);
ServoMotor joint1Servo2(2);
ServoMotor joint2Servo1(3);
ServoMotor joint2Servo2(4);
ServoMotor wristServo(5);

// 베이스 회전 함수 (좌우)
void setBaseAngle(int value) { baseServo.setAngle(value); }

// 관절 1 회전 함수 (앞뒤)
void setJoint1Angle(int value) {
    joint1Servo1.setAngle(value);
    joint1Servo2.setAngle(-value);
}

// 관절 2 회전 함수 (위아래)
void setJoint2Angle(int value) {
    joint2Servo1.setAngle(value);
    joint2Servo2.setAngle(-value);
}

// 손목 회전 함수
void setWristangle(int value) { wristServo.setAngle(value); }

// 서보모터 준비 함수
void setServoReady() {
    // PCA9685 초기화
    pca9685.begin();
    pca9685.setPWMFreq(50); // 서보모터는 50Hz PWM 신호 사용
}
