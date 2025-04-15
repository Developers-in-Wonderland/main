#include <FastLED.h>

#define NUM_LEDS 30      // 컨트롤할 LED 개수
#define DATA_PIN 3      // 아두이노-네오 픽셀 연결 핀핀

CRGBW leds[NUM_LEDS];   // SK6812 RGBW 전용 배열

void setup() {
    FastLED.addLeds<SK6812, DATA_PIN, RGBW>(leds, NUM_LEDS);
    Serial.begin(9600);
}

void loop() {
    if (Serial.available() > 0) { // 라즈베리파이와의의 통신
        char receivedChar = Serial.read(); 

        int brightness = 0;
        if (receivedChar >= '1' && receivedChar <= '5') {
            brightness = (255 / 5) * (receivedChar - '0'); // '1'이 오면 밝기가 255/5*1
        }

        // 하얀색으로 밝기 조절
        for (int i = 0; i < NUM_LEDS; i++) {
            leds[i] = CRGBW(0, 0, 0, brightness);  // W 채널만 사용하여 하얀색
        }

        FastLED.show();
        delay(500);
    }
}