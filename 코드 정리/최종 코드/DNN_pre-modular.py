# ─────────────────────────────────────────────────
# Merged Code: 1학기 최종본 + (2학기) DNN 얼굴인식 / 1~9 연속촬영(3초) / 오버레이
# ─────────────────────────────────────────────────
import cv2  # openCV 얼굴인식, 영상처리. 프레임 캡쳐 등
import serial  # 아두이노 시리얼 통신을 위한 모듈
import time  # 시간지연, 아두이노 초기화 대기에 사용
import numpy as np  # 수학 계산, clip()/각도 제한 처리용
import threading  # 파이썬 비동기 처리, 시리얼 통신을 영상과 동시에 돌림
import queue  # 두 스레드 간 데이터 주고 받을 떄 사용용
import os  # 경로 조작에 사용됨
import re  # 정규표현식 사용을 위함, 파일 이름에서 숫자를 추출할 때 사용
import sys  # For silencing PyAudio errors (optional)

# --- Voice Control Imports ---
ENABLE_VOICE = True
from collections import defaultdict
try:
    import pyaudio
    import webrtcvad
    from google.cloud import speech
except Exception as e:
    print(f"[Voice Disabled] 음성 의존성 누락: {e}")
    ENABLE_VOICE = False


# ─────────────────────────────────────────────────
# PATHS and GLOBAL CONFIGURATIONS
# ─────────────────────────────────────────────────
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
os.makedirs(desktop_path, exist_ok=True)

GOOGLE_CREDENTIALS_PATH = "C:\\Dev\\Capstone\\plucky-sound-433806-d9-f99c357c998e.json"  # 사용자 환경에 맞게 수정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIALS_PATH

try:
    sys.stderr = open(os.devnull, 'w')
except Exception:
    pass

# --- Serial Communication Setup ---
SERIAL_PORT = 'COM5'  # 실제 사용하는 COM 포트로 변경
SERIAL_BAUD = 115200
motor_cmd_queue = queue.Queue(maxsize=10)
move_ready = threading.Event()
move_ready.set()

# --- Camera Setup ---
CAMERA_INDEX = 0
CAMERA_BACKEND = cv2.CAP_MSMF 
# -- CAMERA_BACKEND = cv2.CAP_DSHOW
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
VIDEO_FPS = 20.0

# --- Face Tracking & Motor Control Parameters ---
DESIRED_FACE_AREA = 35000
smoothing_alpha = 0.7
prev_cx, prev_cy = None, None

motor_freeze_time = {"x": 0, "y": 0, "z": 0}
FREEZE_DURATION_S = 0.6

# --- Prediction & Scan Logic Parameters ---
last_known_angles = None
last_face_detected_time = 0
PREDICTION_COOLDOWN_S = 0.8
PREDICTION_ACTIVE_DURATION_S = 2.0
prediction_start_time = 0
is_predicting_movement = False
DECAY_RATE = 0.65
prediction_decay_count = 0
MAX_PREDICTION_REPEAT = 3

is_scanning = False
SCAN_WAIT_DURATION_S = 4.0
scan_start_time_tracker = 0
scan_direction = 1
SCAN_STEP_MOTOR1 = 1
SCAN_MAX_STEPS_ONE_DIRECTION = 20
scan_current_step_count = 0
SCAN_DELAY_MS = 120
RETURN_TO_DEFAULT_DELAY_MS = 250

# --- Recording & Photo State Variables ---
recording = False
out = None
fourcc = cv2.VideoWriter_fourcc(*'XVID')

photo_shooting = False
photo_count_target = 0
photo_taken_count = 0
photo_interval_config = 3  # ← 숫자키 기반 연속촬영 기본 간격 3초(2학기 사양)
countdown_start_time = 0

# --- Voice Command Request Flags ---
request_start_recording = False
request_stop_recording = False
request_continuous_photo = False
voice_photo_count_target = 3
voice_photo_interval_config = 5  # 음성 명령 기본 간격(1학기 스타일 유지)

# --- Thread Control ---
stop_all_threads = threading.Event()

# ─────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────
def get_new_filename(base_name="output", ext="avi"):
    existing_files = os.listdir(desktop_path)
    pattern = re.compile(rf"{re.escape(base_name)}_(\d+)\.{re.escape(ext)}")
    numbers = [int(match.group(1)) for file in existing_files if (match := pattern.match(file))]
    next_number = max(numbers, default=0) + 1
    return os.path.join(desktop_path, f"{base_name}_{next_number}.{ext}")

def get_new_picture_filename(base_name="picture", ext="jpg"):
    existing_files = os.listdir(desktop_path)
    pattern = re.compile(rf"{re.escape(base_name)}_(\d+)\.{re.escape(ext)}")
    numbers = [int(match.group(1)) for file in existing_files if (match := pattern.match(file))]
    next_number = max(numbers, default=0) + 1
    return os.path.join(desktop_path, f"{base_name}_{next_number}.{ext}")

def draw_text(img, text, org, font_scale=1.0, thickness=2):
    # 하얀 외곽선 + 검정 본문(가독성 Up)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

def compute_delay(dx, dy, min_delay=20, max_delay=60):
    distance = np.sqrt(dx ** 2 + dy ** 2)
    normalized = min(distance / 320, 1.0)
    delay = int(max_delay - (max_delay - min_delay) * normalized)
    return delay

def compute_motor_angles(center_x, center_y, area, frame_shape, desired_area=DESIRED_FACE_AREA, deadzone_xy=20, deadzone_area=12000):
    frame_h, frame_w = frame_shape[:2]
    dx = center_x - (frame_w // 2)
    dy = center_y - (frame_h // 2)
    dz = desired_area - area

    ddx = 0 if abs(dx) <= deadzone_xy else (-1 if dx > 0 else 1)
    ddy = 0 if abs(dy) <= deadzone_xy else (-1 if dy > 0 else 1)
    ddz = 0 if abs(dz) <= deadzone_area else (1 if dz > 0 else -1)

    delay = compute_delay(dx, dy)

    return {
        "motor_1": ddx,
        "motor_2": 0,
        "motor_3": ddy,
        "motor_4": 3 * ddz,
        "motor_5": -2 * ddz,
        "motor_6": ddz,
        "motor_7": delay
    }

def clip_motor_angles(motor_cmds, limits=(-80, 80)):
    clipped = {}
    for k, v_float in motor_cmds.items():
        v = int(v_float)
        if k == "motor_7":
            clipped[k] = int(np.clip(v, 10, 500))
        else:
            clipped[k] = int(np.clip(v, limits[0], limits[1]))
    return clipped

def should_freeze(axis, now):
    return now - motor_freeze_time[axis] < FREEZE_DURATION_S

def update_freeze_timer(ddx, ddy, ddz, now):
    if ddx == 0:
        motor_freeze_time["x"] = now
    if ddy == 0:
        motor_freeze_time["y"] = now
    if ddz == 0:
        motor_freeze_time["z"] = now

def draw_face_info(frame, x, y, w, h, center_x, center_y, area):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
    cv2.putText(frame, f"Center:({center_x},{center_y})", (x, y - 25 if y - 25 > 10 else y + h + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(frame, f"Width:{w} Area:{area}", (x, y - 10 if y - 10 > 0 else y + h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# ─────────────────────────────────────────────────
# (2학기) DNN 얼굴 검출 블록 추가
# ─────────────────────────────────────────────────
prototxt_path = r"C:\face_models\deploy.prototxt"
model_path = r"C:\face_models\res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

def detect_faces_dnn(frame, conf_thresh=0.5):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_thresh:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes

# ─────────────────────────────────────────────────
# SERIAL WORKER THREAD
# ─────────────────────────────────────────────────
def serial_worker_thread_func(q, port, baud):
    global move_ready, stop_all_threads
    ser = None
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        print(f"시리얼 연결 완료 ({port}@{baud})")
    except serial.SerialException as e:
        print(f"시리얼 연결 실패 ({port}): {e}")
        return

    while not stop_all_threads.is_set():
        try:
            motor_cmds = q.get(timeout=0.1)
            if motor_cmds is None:
                print("[Serial] 종료 신호 수신")
                break

            # 최신 명령만 남기고 비움(지연 줄이기)
            temp_cmds = motor_cmds
            while not q.empty():
                latest = q.get_nowait()
                if latest is not None:
                    temp_cmds = latest
                else:
                    stop_all_threads.set()
                    print("[Serial] 큐 정리 중 종료 신호")
                    break
            motor_cmds = temp_cmds
            if stop_all_threads.is_set() and motor_cmds is None:
                break
            if motor_cmds is None:
                print("[Serial] 최종 명령 None으로 종료")
                break

            values = [motor_cmds[f"motor_{i}"] for i in range(1, 8)]
            message = ','.join(map(str, values)) + '\n'
            ser.write(message.encode('utf-8'))

            delay_ms = motor_cmds["motor_7"]
            move_ready.clear()
            time.sleep(delay_ms / 1000.0)
            move_ready.set()
        except queue.Empty:
            continue
        except serial.SerialException as e:
            print(f"[Serial Error] 시리얼 통신 오류: {e}")
            break
        except Exception as e:
            print(f"[Serial Worker Error] 예기치 않은 오류: {e}")
            break

    if ser and ser.is_open:
        ser.close()
    print("시리얼 스레드 종료됨.")

# ─────────────────────────────────────────────────
# VOICE CONTROL THREAD and Functions (1학기 로직 유지)
# ─────────────────────────────────────────────────
VAD_RATE = 16000
VAD_FRAME_MS = 30
VAD_FRAME_SIZE = VAD_RATE * VAD_FRAME_MS // 1000
VAD_MAX_SILENCE_MS = 1500
VAD_MAX_SILENCE_FRAMES = VAD_MAX_SILENCE_MS // VAD_FRAME_MS
speech_client = None
try:
    speech_client = speech.SpeechClient()
except Exception as e:
    print(f"Google Speech Client 초기화 실패: {e}.")

# ---- Voice config (guarded) ----
if ENABLE_VOICE:
    PHRASE_BOOSTS = {
        "5초 뒤에 촬영 시작": 20.0, "촬영 시작": 20.0,
        "촬영 종료": 20.0, "연속 촬영": 20.0,
    }
    VOICE_COMMANDS_MAP = {
        "5초 뒤에 촬영 시작": "delayed_recording", "촬영 시작": "start_recording",
        "촬영 종료": "stop_recording", "연속 촬영": "continuous_photo",
    }

    # defaultdict는 전역에서: from collections import defaultdict (이미 추가했어야 함)
    speech_contexts_list = defaultdict(list)
    for p, b in PHRASE_BOOSTS.items():
        speech_contexts_list[b].append(p)

    speech_contexts_config = [
        speech.SpeechContext(phrases=pl, boost=bv)
        for bv, pl in speech_contexts_list.items()
    ]

    # STT 설정
    VAD_RATE = 16000  # 상단에 이미 있다면 중복 정의 제거해도 됨
    recognition_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=VAD_RATE,
        language_code="ko-KR",
        speech_contexts=speech_contexts_config,
        enable_automatic_punctuation=True,
        model="command_and_search",
    )

    # VAD & Speech Client
    vad = webrtcvad.Vad(2)
    try:
        speech_client = speech.SpeechClient()
    except Exception as e:
        print(f"Google Speech Client 초기화 실패: {e}.")
        ENABLE_VOICE = False
        speech_client = None
else:
    # 음성 비활성화 시 참조 에러 방지
    speech_client = None


def vc_act_delay_rec():
    global request_start_recording
    print(f"[Voice] 지연 녹화 시작 요청")
    request_start_recording = True

def trig_vc_delay_rec(d=5):
    print(f"[Voice] {d}초 후 녹화 명령")
    t = threading.Timer(d, vc_act_delay_rec)
    t.daemon = True
    t.start()

def trig_vc_start_rec():
    global request_start_recording
    print(f"[Voice] 즉시 녹화 명령")
    request_start_recording = True

def trig_vc_stop_rec():
    global request_stop_recording
    print(f"[Voice] 녹화 종료 명령")
    request_stop_recording = True

def trig_vc_cont_photo(c=3, i=5):
    global request_continuous_photo, voice_photo_count_target, voice_photo_interval_config
    print(f"[Voice] 연속 촬영 명령 ({c}장, {i}초)")
    voice_photo_count_target = c
    voice_photo_interval_config = i
    request_continuous_photo = True

VOICE_ACTION_HANDLER_MAP = {
    "delayed_recording": trig_vc_delay_rec, "start_recording": trig_vc_start_rec,
    "stop_recording": trig_vc_stop_rec, "continuous_photo": trig_vc_cont_photo,
}

def record_utterance(p_stream):
    frms = []
    sil_frms = 0
    print("음성 감지 대기 중...")
    while not stop_all_threads.is_set():
        try:
            aud_chk = p_stream.read(VAD_FRAME_SIZE, exception_on_overflow=False)
            if vad.is_speech(aud_chk, VAD_RATE):
                print("음성 감지 시작!")
                frms.append(aud_chk)
                break
        except IOError as e:
            print(f"[PyAudio VAD Err]: {e}")
            return None
        if stop_all_threads.is_set():
            return None

    if not frms:
        return None

    while not stop_all_threads.is_set():
        try:
            aud_chk = p_stream.read(VAD_FRAME_SIZE, exception_on_overflow=False)
            frms.append(aud_chk)
            if not vad.is_speech(aud_chk, VAD_RATE):
                sil_frms += 1
                if sil_frms > VAD_MAX_SILENCE_FRAMES:
                    print("음성 종료 (침묵).")
                    break
            else:
                sil_frms = 0
        except IOError as e:
            print(f"[PyAudio Rec Err]: {e}")
            return b"".join(frms) if frms else None
        if stop_all_threads.is_set():
            return None

    return b"".join(frms) if frms else None

def voice_listener_thread_func():
    global stop_all_threads, speech_client
    if speech_client is None:
        print("GSC 미초기화. 음성인식 불가.")
        return

    pa = pyaudio.PyAudio()
    stream = None
    try:
        stream = pa.open(format=pyaudio.paInt16, channels=1,
                         rate=VAD_RATE, input=True,
                         frames_per_buffer=VAD_FRAME_SIZE)
    except Exception as e:
        print(f"PyAudio 스트림 열기 실패: {e}.")
        if pa:
            pa.terminate()
        return

    print("음성 인식 스레드 시작.")
    while not stop_all_threads.is_set():
        try:
            if not stream or not stream.is_active():
                print("PyAudio 스트림 재시도...")
                time.sleep(1)
                if stream:
                    try:
                        stream.close()
                    except Exception as e_close:
                        print(f"기존 스트림 닫기 중 오류: {e_close}")
                try:
                    stream = pa.open(format=pyaudio.paInt16, channels=1,
                                     rate=VAD_RATE, input=True,
                                     frames_per_buffer=VAD_FRAME_SIZE)
                    print("PyAudio 스트림 재연결 성공.")
                except Exception as er:
                    print(f"PyAudio 스트림 재연결 실패: {er}.")
                    time.sleep(1)
                    continue
                continue  # 스트림 재연결 후 루프 처음으로

            aud_b = record_utterance(stream)
            if aud_b is None:
                if stop_all_threads.is_set():
                    break
                continue  # 음성 입력이 없거나 중단된 경우 다음 시도

            aud_in = speech.RecognitionAudio(content=aud_b)
            resp = speech_client.recognize(config=recognition_config, audio=aud_in)

            if not resp.results:
                print("Google STT: 텍스트 없음.")
                continue

            rec_txt = resp.results[0].alternatives[0].transcript.strip()
            print(f"인식된 텍스트(Raw): \"{rec_txt}\"")

            cmd_key_exec = None
            act_trig = False

            # 1. 정확한 전체 구문 일치 확인
            for phr, cmd_k in VOICE_COMMANDS_MAP.items():
                if phr == rec_txt:
                    cmd_key_exec = cmd_k
                    print(f"→ 명령어 정확히 일치: \"{phr}\" (키: {cmd_key_exec})")
                    break

            # 2. 정의된 구문이 부분 문자열로 포함되는지 확인
            if cmd_key_exec is None:
                for phr, cmd_k in VOICE_COMMANDS_MAP.items():
                    if phr in rec_txt:
                        cmd_key_exec = cmd_k
                        print(f"→ 명령어 부분 일치: \"{phr}\" in \"{rec_txt}\" (키: {cmd_key_exec})")
                        break

            # 3. 단독 키워드 포함 확인
            if cmd_key_exec is None:
                if "시작" in rec_txt:
                    print(f"→ 키워드 '시작' 포함. '촬영 시작' 간주.")
                    cmd_key_exec = "start_recording"
                elif "종료" in rec_txt:
                    print(f"→ 키워드 '종료' 포함. '촬영 종료' 간주.")
                    cmd_key_exec = "stop_recording"
                elif "연속" in rec_txt:
                    print(f"→ 키워드 '연속' 포함. '연속 촬영' 간주.")
                    cmd_key_exec = "continuous_photo"

            if cmd_key_exec and cmd_key_exec in VOICE_ACTION_HANDLER_MAP:
                VOICE_ACTION_HANDLER_MAP[cmd_key_exec]()
                act_trig = True

            if not act_trig and rec_txt:
                print(f"→ \"{rec_txt}\" 처리 가능한 명령어/키워드 없음.")

        except KeyboardInterrupt:
            stop_all_threads.set()
            break
        except Exception as e:
            print(f"[Voice Thread Err] {e}")
            if any(s in str(e) for s in ["HttpError 401", "Default Credentials"]):
                stop_all_threads.set()
            elif any(s in str(e) for s in ["Deadline Exceeded", "RESOURCE_EXHAUSTED"]):
                time.sleep(5)
            elif any(s in str(e) for s in ["Stream closed", "Unanticipated host error", "Invalid input device"]):
                if stream:
                    try:
                        stream.stop_stream()
                        stream.close()
                    except:
                        pass
                stream = None
                time.sleep(1)
            else:
                time.sleep(1)

    print("음성 인식 스레드 종료 중...")
    if stream:
        try:
            if stream.is_active():
                stream.stop_stream()
            stream.close()
        except:
            pass
    if pa:
        pa.terminate()
        print("PyAudio 리소스 해제 완료.")

# ─────────────────────────────────────────────────
# MAIN APPLICATION LOGIC
# ─────────────────────────────────────────────────
def main():
    global recording, out, photo_shooting, photo_count_target, photo_taken_count, photo_interval_config
    global countdown_start_time, stop_all_threads, motor_cmd_queue, move_ready, prev_cx, prev_cy
    global request_start_recording, request_stop_recording, request_continuous_photo
    global voice_photo_count_target, voice_photo_interval_config
    global last_known_angles, last_face_detected_time, prediction_start_time
    global is_predicting_movement, prediction_decay_count
    global is_scanning, scan_start_time_tracker, scan_direction, scan_current_step_count

    serial_thread = threading.Thread(target=serial_worker_thread_func, args=(motor_cmd_queue, SERIAL_PORT, SERIAL_BAUD), daemon=True)
    serial_thread.start()
    voice_thread = threading.Thread(target=voice_listener_thread_func, daemon=True)
    voice_thread.start()

    cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
    if not cap.isOpened():
        print(f"카메라 인덱스 {CAMERA_INDEX} ({CAMERA_BACKEND}) 열기 실패")
        stop_all_threads.set()
        if serial_thread.is_alive():
            motor_cmd_queue.put(None)
            serial_thread.join(timeout=2)
        if voice_thread.is_alive():
            voice_thread.join(timeout=2)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    print("Main loop 시작. 's'=녹화, 'e'=녹화종료, 숫자(1~9)=사진촬영(3초 간격), 'q'=종료. 음성 명령 가능.")

    try:
        while not stop_all_threads.is_set():
            ret, frame = cap.read()
            if not ret:
                print("프레임 읽기 실패, 카메라 재시도...")
                if cap.isOpened():
                    cap.release()
                time.sleep(0.5)
                cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
                if not cap.isOpened():
                    print("카메라 재연결 실패. 종료.")
                    stop_all_threads.set()
                    break
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, FPS)
                print("카메라 재연결 성공.")
                continue

            current_time = time.time()
            key = cv2.waitKey(1) & 0xFF

            # ── 음성/키보드 명령 처리 ─────────────────────────
            if request_start_recording:
                if not recording and not photo_shooting:
                    output_path = get_new_filename()
                    print(f"녹화 시작 (음성)! 파일명: {os.path.basename(output_path)}")
                    h_f, w_f = frame.shape[:2]
                    out = cv2.VideoWriter(output_path, fourcc, VIDEO_FPS, (w_f, h_f))
                    if out.isOpened():
                        recording = True
                    else:
                        print("VideoWriter 열기 실패 (음성)")
                request_start_recording = False

            if request_stop_recording:
                if recording:
                    print("녹화 종료 (음성)!")
                    recording = False
                    if out:
                        out.release()
                        out = None
                request_stop_recording = False

            if request_continuous_photo:
                if not photo_shooting and not recording:
                    photo_count_target = voice_photo_count_target
                    photo_interval_config = voice_photo_interval_config  # 음성은 1학기 기본값 유지(5초 등)
                    photo_taken_count = 0
                    countdown_start_time = current_time
                    photo_shooting = True
                    print(f"{photo_count_target}장 사진 촬영 시작 (음성, {photo_interval_config}초 간격)!")
                request_continuous_photo = False

            if key == ord('q'):
                stop_all_threads.set()
                break

            # 녹화 시작/종료 (키보드)
            if key == ord('s') and not recording and not photo_shooting:
                output_path = get_new_filename()
                print(f"녹화 시작 (키보드)! 파일명: {os.path.basename(output_path)}")
                h_f, w_f = frame.shape[:2]
                out = cv2.VideoWriter(output_path, fourcc, VIDEO_FPS, (w_f, h_f))
                if out.isOpened():
                    recording = True
                else:
                    print("VideoWriter 열기 실패 (키보드)")
            elif key == ord('e') and recording:
                print("녹화 종료 (키보드)!")
                recording = False
                if out:
                    out.release()
                    out = None
            # 1~9 연속촬영 (키보드, 2학기 사양: 3초 간격 고정)
            elif (ord('1') <= key <= ord('9')) and not photo_shooting and not recording:
                photo_count_target = key - ord('0')
                photo_interval_config = 3  # 숫자키는 3초 고정
                photo_taken_count = 0
                countdown_start_time = current_time
                photo_shooting = True
                print(f"{photo_count_target}장 사진 촬영 시작 (키보드, {photo_interval_config}초 간격)!")

            # ── 녹화 상태 오버레이 ─────────────────────────
            if recording:
                if out:
                    out.write(frame)
                draw_text(frame, "REC ●", (10, frame.shape[0] - 15), 0.8, 2)

            # ── 연속촬영 로직(2학기 스타일 오버레이) ────────────
            if photo_shooting:
                # 다음 촬영까지 남은 시간 계산
                next_shot_at = countdown_start_time + photo_interval_config
                remain = max(0.0, next_shot_at - current_time)
                remain_ceil = int(np.ceil(remain))

                # 좌상단 카운트다운: 3 → 2 → 1 cheese~!
                if remain_ceil >= 3:
                    cd_str = "3"
                elif remain_ceil == 2:
                    cd_str = "2"
                elif remain_ceil == 1:
                    cd_str = "1 cheese~!"
                else:
                    cd_str = "cheese~!"
                draw_text(frame, cd_str, (20, 40), 1.2, 2)

                # 우상단 잔여 장수
                shots_left = max(0, photo_count_target - photo_taken_count)
                right_txt = f"Shots left: {shots_left}"
                (tw, th), _ = cv2.getTextSize(right_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                draw_text(frame, right_txt, (frame.shape[1] - tw - 20, 40), 0.8, 2)

                # 촬영 시점 도달
                if current_time >= next_shot_at:
                    pic_filename = get_new_picture_filename()
                    cv2.imwrite(pic_filename, frame)  # 오버레이 없는 원본 필요하면 frame.copy()를 사용
                    photo_taken_count += 1
                    print(f"{photo_taken_count}/{photo_count_target} 사진 저장: {os.path.basename(pic_filename)}")
                    countdown_start_time = current_time  # 다음 장을 위한 기준 재설정

                    if photo_taken_count >= photo_count_target:
                        photo_shooting = False
                        print("연속 사진 촬영 완료!")

            # ── 얼굴 검출/추적 (DNN 사용, 나머지 예측/스캔 로직은 1학기 그대로) ──
            current_status_text = ""
            status_color = (0, 0, 0)

            if not photo_shooting:  # 촬영 중엔 오버레이가 겹치지 않게 기존 설계 유지
                # DNN 기반 얼굴 검출
                all_faces = detect_faces_dnn(frame, conf_thresh=0.5)

                if all_faces:
                    if is_predicting_movement or is_scanning:
                        print("[상태] 얼굴 재감지. 예측/스캔 중단.")
                    is_predicting_movement = False
                    is_scanning = False
                    prediction_decay_count = 0
                    scan_current_step_count = 0
                    if not motor_cmd_queue.empty():
                        print(f"[Queue] 얼굴 감지, 이전 명령 {motor_cmd_queue.qsize()}개 큐 비움.")
                        with motor_cmd_queue.mutex:
                            motor_cmd_queue.queue.clear()

                    all_faces.sort(key=lambda r: r[2] * r[3], reverse=True)
                    x, y, w, h = all_faces[0]
                    current_cx = x + w // 2
                    current_cy = y + h // 2
                    if prev_cx is None:
                        smoothed_cx, smoothed_cy = current_cx, current_cy
                    else:
                        smoothed_cx = int(smoothing_alpha * prev_cx + (1 - smoothing_alpha) * current_cx)
                        smoothed_cy = int(smoothing_alpha * prev_cy + (1 - smoothing_alpha) * current_cy)
                    prev_cx, prev_cy = smoothed_cx, smoothed_cy
                    area = w * h
                    last_face_detected_time = current_time

                    if not recording:
                        draw_face_info(frame, x, y, w, h, smoothed_cx, smoothed_cy, area)

                    if move_ready.is_set() and not recording:
                        angles = compute_motor_angles(smoothed_cx, smoothed_cy, area, frame.shape,
                                                      deadzone_xy=20, deadzone_area=12000)

                        dx_val = smoothed_cx - (frame.shape[1] // 2)
                        dy_val = smoothed_cy - (frame.shape[0] // 2)
                        dz_val = DESIRED_FACE_AREA - area
                        ddx_fr = 0 if abs(dx_val) <= 20 else (-1 if dx_val > 0 else 1)
                        ddy_fr = 0 if abs(dy_val) <= 20 else (-1 if dy_val > 0 else 1)
                        ddz_fr = 0 if abs(dz_val) <= 12000 else (1 if dz_val > 0 else -1)
                        update_freeze_timer(ddx_fr, ddy_fr, ddz_fr, current_time)

                        if should_freeze("x", current_time):
                            angles["motor_1"] = 0
                        if should_freeze("y", current_time):
                            angles["motor_3"] = 0
                        if should_freeze("z", current_time):
                            angles["motor_4"], angles["motor_5"], angles["motor_6"] = 0, 0, 0

                        clipped_angles = clip_motor_angles(angles)
                        last_known_angles = clipped_angles.copy()
                        if not motor_cmd_queue.full():
                            motor_cmd_queue.put(clipped_angles)
                else:
                    prev_cx, prev_cy = None, None
                    current_status_text = "얼굴 없음"
                    status_color = (0, 0, 255)
                    if is_scanning:
                        current_status_text = "스캔 중..."
                        status_color = (255, 100, 0)
                        if move_ready.is_set():
                            scan_current_step_count += 1
                            if scan_current_step_count > SCAN_MAX_STEPS_ONE_DIRECTION:
                                scan_direction *= -1
                                scan_current_step_count = 0
                            scan_cmd = {f"motor_{i}": 0 for i in range(1, 7)}
                            scan_cmd["motor_1"] = SCAN_STEP_MOTOR1 * scan_direction
                            scan_cmd["motor_2"] = 0
                            scan_cmd["motor_7"] = SCAN_DELAY_MS
                            if not motor_cmd_queue.full():
                                motor_cmd_queue.put(scan_cmd)
                    elif is_predicting_movement:
                        current_status_text = "예측 중..."
                        status_color = (0, 165, 255)
                        elapsed_pred_time = current_time - prediction_start_time
                        if elapsed_pred_time > PREDICTION_ACTIVE_DURATION_S:
                            print(f"[예측 종료] 시간 초과. 스캔 대기 시작.")
                            is_predicting_movement = False
                            last_known_angles = None
                            scan_start_time_tracker = current_time
                            stop_cmd = {f"motor_{i}": 0 for i in range(1, 7)}
                            stop_cmd["motor_2"] = 0
                            stop_cmd["motor_7"] = RETURN_TO_DEFAULT_DELAY_MS
                            if not motor_cmd_queue.full():
                                motor_cmd_queue.put(stop_cmd)
                        elif move_ready.is_set() and last_known_angles:
                            if prediction_decay_count < MAX_PREDICTION_REPEAT:
                                decayed_cmd = last_known_angles.copy()
                                for i in [1, 3, 4, 5, 6]:
                                    decayed_cmd[f"motor_{i}"] = int(decayed_cmd[f"motor_{i}"] * (DECAY_RATE ** prediction_decay_count))
                                decayed_cmd["motor_7"] = 60
                                if not motor_cmd_queue.full():
                                    motor_cmd_queue.put(decayed_cmd)
                                prediction_decay_count += 1
                            else:
                                print(f"[예측 종료] 최대 반복. 스캔 대기 시작.")
                                is_predicting_movement = False
                                last_known_angles = None
                                scan_start_time_tracker = current_time
                                stop_cmd = {f"motor_{i}": 0 for i in range(1, 7)}
                                stop_cmd["motor_2"] = 0
                                stop_cmd["motor_7"] = RETURN_TO_DEFAULT_DELAY_MS
                                if not motor_cmd_queue.full():
                                    motor_cmd_queue.put(stop_cmd)
                    elif last_known_angles and move_ready.is_set():
                        if (current_time - last_face_detected_time) > PREDICTION_COOLDOWN_S:
                            print(f"[예측 시작] {PREDICTION_COOLDOWN_S:.1f}초간 얼굴 못찾음.")
                            is_predicting_movement = True
                            prediction_start_time = current_time
                            prediction_decay_count = 0
                            init_pred_cmd = last_known_angles.copy()
                            init_pred_cmd["motor_7"] = 70
                            if not motor_cmd_queue.full():
                                motor_cmd_queue.put(init_pred_cmd)
                    elif last_known_angles is None:
                        current_status_text = "스캔 대기 중..."
                        status_color = (200, 200, 0)
                        if scan_start_time_tracker == 0:
                            scan_start_time_tracker = current_time
                        if (current_time - scan_start_time_tracker) > SCAN_WAIT_DURATION_S:
                            print(f"[스캔 준비] {SCAN_WAIT_DURATION_S:.1f}초 대기 후 스캔 시작.")
                            is_scanning = True
                            scan_current_step_count = 0
                            scan_direction = 1
                            default_pos_cmd = {f"motor_{i}": 0 for i in range(1, 7)}
                            default_pos_cmd["motor_2"] = 0
                            default_pos_cmd["motor_7"] = RETURN_TO_DEFAULT_DELAY_MS
                            if not motor_cmd_queue.empty():
                                with motor_cmd_queue.mutex:
                                    motor_cmd_queue.queue.clear()
                            if not motor_cmd_queue.full():
                                motor_cmd_queue.put(default_pos_cmd)
                                print(f"[Robo] 기본 위치로 이동 (스캔 시작 전)")
                    if not recording and current_status_text:
                        cv2.putText(frame, current_status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            cv2.imshow('Live Camera Feed with Enhanced Tracking', frame)
    except KeyboardInterrupt:
        print("KeyboardInterrupt. 프로그램 종료 중...")
    finally:
        print("메인 루프 종료. 리소스 정리 시작...")
        stop_all_threads.set()
        if cap.isOpened():
            cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        if serial_thread.is_alive():
            print("시리얼 스레드 종료 대기...")
            motor_cmd_queue.put(None)
            serial_thread.join(timeout=3)
            if not serial_thread.is_alive():
                print("시리얼 스레드 정상 종료됨.")
            else:
                print("시리얼 스레드가 시간 내에 종료되지 않았습니다.")
        else:
            print("시리얼 스레드가 이미 종료되었거나 시작되지 않았습니다.")
        if voice_thread.is_alive():
            print("음성 인식 스레드 종료 대기...")
            voice_thread.join(timeout=2)
        print("모든 리소스 정리 완료. 프로그램 종료.")

if __name__ == '__main__':
    main()
