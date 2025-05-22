import os
import sys
import pyaudio
import webrtcvad
from google.cloud import speech
from collections import defaultdict

from camera import CameraRecorder

# 로그 무시, GCP 인증
sys.stderr = open(os.devnull,'w')
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=(
  "C:\\Dev\\Capstone\\plucky-sound-433806-d9-f99c357c998e.json"
)

# 음성/STT 설정
RATE=16000; FRAME_MS=30
FRAME_SIZE=RATE*FRAME_MS//1000
MAX_SILENCE_MS=500
MAX_SILENCE_FRAMES=MAX_SILENCE_MS//FRAME_MS

client = speech.SpeechClient()
phrase_boosts = {
  "5초 뒤에 촬영 시작":20.0,
  "촬영 시작":12.0,
  "촬영 종료":12.0,
  "연속 촬영":10.0,
}
COMMANDS = {
  "5초 뒤에 촬영 시작":"delayed_recording",
  "촬영 시작":"start_recording",
  "촬영 종료":"stop_recording",
  "연속 촬영":"continuous_photo",
}

cam = CameraRecorder(device_index=1)

def delayed_recording(): cam.delayed_recording(5)
def start_recording():   cam.start_recording()
def stop_recording():    cam.stop_recording()
def continuous_photo():  cam.continuous_photo(count=3,interval=5)

ACTION_MAP = {
  "delayed_recording":delayed_recording,
  "start_recording":start_recording,
  "stop_recording":stop_recording,
  "continuous_photo":continuous_photo
}

# STT 컨텍스트 준비
contexts=defaultdict(list)
for p,b in phrase_boosts.items(): contexts[b].append(p)
speech_contexts=[
  speech.SpeechContext(phrases=ph,boost=b)
  for b,ph in contexts.items()
]
recognition_config=speech.RecognitionConfig(
  encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
  sample_rate_hertz=RATE,
  language_code="ko-KR",
  speech_contexts=speech_contexts
)

# VAD & PyAudio
vad=webrtcvad.Vad(2)
pa = pyaudio.PyAudio()
stream=pa.open(format=pyaudio.paInt16,channels=1,
               rate=RATE,input=True,frames_per_buffer=FRAME_SIZE)

def record_utterance():
  frames=[]; silence=0
  while True:
    c=stream.read(FRAME_SIZE,exception_on_overflow=False)
    if vad.is_speech(c,RATE):
      frames.append(c); break
  while True:
    c=stream.read(FRAME_SIZE,exception_on_overflow=False)
    frames.append(c)
    if not vad.is_speech(c,RATE):
      silence+=1
      if silence>MAX_SILENCE_FRAMES: break
    else: silence=0
  return b"".join(frames)

print("음성 대기… Ctrl+C 종료")
try:
  while True:
    audio_bytes=record_utterance()
    audio=speech.RecognitionAudio(content=audio_bytes)
    resp = client.recognize(config=recognition_config,audio=audio)
    if not resp.results: continue
    text=resp.results[0].alternatives[0].transcript.strip()
    print("인식된 텍스트:",text)
    for phr,cmd in COMMANDS.items():
      if phr in text:
        print("→ 명령어:",phr)
        ACTION_MAP[cmd]()
        break

except KeyboardInterrupt:
  print("\n종료 중…")

finally:
  stream.stop_stream()
  stream.close()
  pa.terminate()
  cam.close()
