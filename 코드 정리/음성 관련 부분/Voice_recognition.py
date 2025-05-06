import os
import sys
import time
import collections

import pyaudio
import webrtcvad
from google.cloud import speech
from collections import defaultdict

# 1) ALSA/PortAudio 내부 로그 무시
sys.stderr = open(os.devnull, 'w')

# 2) GCP 인증 정보
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = \
    "/home/pi/usb_4_mic_array/data/plucky-sound-433806-d9-f99c357c998e.json"

# 3) 오디오 설정
RATE = 16000
FRAME_DURATION_MS = 30                  # 프레임 길이 (ms)
FRAME_SIZE = int(RATE * FRAME_DURATION_MS / 1000)  # 샘플 수
MAX_SILENCE_MS = 500                    # 종료 판정할 침묵 길이 (ms)
MAX_SILENCE_FRAMES = MAX_SILENCE_MS // FRAME_DURATION_MS

# 4) STT 클라이언트 초기화
client = speech.SpeechClient()

# 5) 명령어별 가중치 설정
phrase_boosts = {
    "5초 뒤에 촬영 시작": 40.0,
    "촬영 시작":         12.0,
    "촬영 종료":         12.0,
    "1단계":              8.0,
    "2단계":              8.0,
    "3단계":              8.0,
    "4단계":              8.0,
    "5단계":              8.0,
}

# 6) COMMANDS / ACTION_MAP 정의
COMMANDS = {
    "5초 뒤에 촬영 시작": "delayed_recording",
    "5초 뒤 촬영 시작": "delayed_recording",
    "촬영 시작":         "start_recording",
    "촬영 종료":         "stop_recording",
    "1단계":              "set_brightness_1",
    "2단계":              "set_brightness_2",
    "3단계":              "set_brightness_3",
    "4단계":              "set_brightness_4",
    "5단계":              "set_brightness_5",
}

def delayed_recording():
    print("[ACTION] 5초 뒤 촬영 시작")

def start_recording():
    print("[ACTION] 촬영 시작")

def stop_recording():
    print("[ACTION] 촬영 종료")

def set_brightness(n):
    print(f"[ACTION] 밝기 단계 {n} 설정")

ACTION_MAP = {
    "delayed_recording": delayed_recording,
    "start_recording":   start_recording,
    "stop_recording":    stop_recording,
    "set_brightness_1":  lambda: set_brightness(1),
    "set_brightness_2":  lambda: set_brightness(2),
    "set_brightness_3":  lambda: set_brightness(3),
    "set_brightness_4":  lambda: set_brightness(4),
    "set_brightness_5":  lambda: set_brightness(5),
}

# 7) phrase_boosts → speech_contexts 생성
contexts = defaultdict(list)
for phrase, boost in phrase_boosts.items():
    contexts[boost].append(phrase)

speech_contexts = [
    speech.SpeechContext(phrases=phrases, boost=boost)
    for boost, phrases in contexts.items()
]

# 8) RecognitionConfig 생성
recognition_config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=RATE,
    language_code="ko-KR",
    speech_contexts=speech_contexts
)

# 9) VAD 초기화
vad = webrtcvad.Vad(2)  # 민감도 0~3

# 10) PyAudio 스트림 (한 번만)
pa = pyaudio.PyAudio()
stream = pa.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=FRAME_SIZE
)

print("음성 인식 대기 중… Ctrl+C로 종료")

def record_utterance():
    """사람의 음성이 시작될 때부터 끝날 때까지 오디오를 수집하여 반환."""
    frames = []
    silence_frames = 0

    # 음성 시작 대기
    while True:
        chunk = stream.read(FRAME_SIZE, exception_on_overflow=False)
        if vad.is_speech(chunk, RATE):
            frames.append(chunk)
            break

    # 음성 중, 그리고 침묵 감지 시 종료 대기
    while True:
        chunk = stream.read(FRAME_SIZE, exception_on_overflow=False)
        frames.append(chunk)
        if not vad.is_speech(chunk, RATE):
            silence_frames += 1
            if silence_frames > MAX_SILENCE_FRAMES:
                break
        else:
            silence_frames = 0

    return b"".join(frames)

try:
    while True:
        # 11) VAD 기반 녹음
        audio_bytes = record_utterance()
        audio = speech.RecognitionAudio(content=audio_bytes)

        # 12) 동기 Recognize 호출
        response = client.recognize(config=recognition_config, audio=audio)
        if not response.results:
            continue

        transcript = response.results[0].alternatives[0].transcript.strip()
        print("인식된 텍스트:", transcript)

        # 13) 매칭된 명령어 처리
        for phrase, key in COMMANDS.items():
            if phrase in transcript:
                print(f"→ 명령어 감지: {phrase} (boost={phrase_boosts[phrase]})")
                ACTION_MAP[key]()
                break

except KeyboardInterrupt:
    print("\n프로그램 종료 중…")

finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
