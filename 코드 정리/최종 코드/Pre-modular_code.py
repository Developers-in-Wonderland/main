import cv2
import numpy as np
import threading
import queue
import serial
import serial.tools.list_ports  # ⭐ 포트 확인용
import os
import re
import time
import sys
from collections import deque

cv2.setUseOptimized(True)
try:
    cv2.ocl.setUseOpenCL(True)
except Exception:
    pass

# ====== 한글 텍스트 유지를 위한 PIL 사용 ======
from PIL import ImageFont, ImageDraw, Image

FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"
def draw_text_kr(img, text, org, font_size=26, thickness=2):
    if not text:
        return img
    img_pil = Image.fromarray(img)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, max(0.5, font_size/26.0),
                    (255,255,255), thickness+2, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, max(0.5, font_size/26.0),
                    (0,0,0), thickness, cv2.LINE_AA)
        return img
    draw = ImageDraw.Draw(img_pil)
    x, y = org
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        draw.text((x+dx, y+dy), text, font=font, fill=(255,255,255))
    draw.text((x, y), text, font=font, fill=(0,0,0))
    return np.array(img_pil)

# ============================================================
# 설정값
# ============================================================
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
CAP_WIDTH, CAP_HEIGHT, CAP_FPS = 1920, 1080, 40
#CAP_WIDTH, CAP_HEIGHT, CAP_FPS = 640, 480, 60
RECORD_USE_STAB = True

# ⭐ 디버깅 설정
DEBUG_MODE = True  # False로 변경하면 디버깅 끄기
DEBUG_DETAIL = False  # True로 변경하면 상세 디버깅 (매 프레임)
DEBUG_SERIAL_TEST = False  # True로 변경하면 시리얼 테스트 모드

# 시리얼 포트 설정 (사용자 환경에 맞게 수정)
SERIAL_PORT = 'COM5'  # ⭐ 여기를 수정하세요!
SERIAL_BAUD = 115200

# 시리얼 통신 진단
serial_health = {
    "last_success_time": 0,
    "consecutive_errors": 0,
    "total_sent": 0,
    "total_errors": 0,
    "connection_lost": False
}

# 오버레이 3프레임마다 그리기
OVERLAY_EVERY = 1   # 3프레임마다만 draw_text_kr 실행

# 검출/추적
DETECT_EVERY = 2 # 1프레임 단위로 검출
LEAD_FACE_SEC = 0.12
CM_PER_PIXEL = 0.050

# 제어(로봇팔) - 방법 A
DESIRED_FACE_AREA = 35000
DEADZONE_XY = 40 # 화질에 따라 조정 필요
DEADZONE_AREA = 12000
move_ready = threading.Event()
move_ready.set()
motor_freeze_time = {"x": 0, "y": 0, "z": 0}
FREEZE_DURATION_S = 0.2 # 모터 프리즈 지속시간

# 중앙고정 & 줌
RATIO_TRANSLATE = 0.3 # 최대 이동 비율, 디지털 짐벌

# 정량지표
reacquire_t0 = None
metric1_times = [] # 재획득 시간 목록
metric1_speeds_px = []
metric1_speeds_cm = []

DT_THRESH_PX = 10.0 # 안정 판정 임계 이동량
STAB_WIN_SEC = 3.0
stab_buf = deque()
metric2_ratios = []

STOP_SPEED_THR = 10.0 #정지 판단 (px/s)
STOP_HOLD_START = 0.5 #정지 시작 후 유예
STOP_HOLD_SEC = 3.0 # 집계시간
icr3_phase = "idle"
icr3_center = None
icr3_t0 = 0.0
icr3_inside = 0
icr3_total = 0
ICR_RATIO = 0.03 # 지표3의 원 반경 화면 대각선 * 3%
CIRCLE_RADIUS_RATIO = 0.02  # 원의 반지름 비율 (화면 가로 기준)
ICR_RADIUS = 0.0
metric3_ratios = []
matric3_text = ""

_prev_cx, _prev_cy = None, None # 이전 프레임 좌표 (지표용)
_prev_t = None

# 디버깅 카운터
debug_counters = {
    "frame_count": 0,
    "face_detected": 0,
    "face_lost": 0,
    "serial_sent": 0,
    "serial_error": 0,
    "motor_frozen": 0
}

# ⭐ 추적 테스트 변수 (평가지표 3용)
test_mode_active = False
test_phase = "idle"
test_start_time = 0
test_stop_start_time = 0
test_coordinates = []
test_reference_point = None

# ⭐ 평가지표 1 변수 (dh1_code.py 방식)
from collections import deque
FACE_PRESENCE_WINDOW_SEC = 0.5
FACE_PRESENCE_Q = deque()

tracking_test_mode = False
tracking_enabled = False
test_duration = 2.0
DETECTION_TIME = 2.0

def update_face_presence(now, present):
    FACE_PRESENCE_Q.append((now, 1 if present else 0))
    while FACE_PRESENCE_Q and (now - FACE_PRESENCE_Q[0][0]) > FACE_PRESENCE_WINDOW_SEC:
        FACE_PRESENCE_Q.popleft()

def recent_face_ratio():
    if not FACE_PRESENCE_Q:
        return 0.0
    s = sum(v for _, v in FACE_PRESENCE_Q)
    return s * 100.0 / len(FACE_PRESENCE_Q)

def reset_test_mode(duration=2.0):
    global tracking_test_mode, tracking_enabled, test_duration
    tracking_test_mode = True
    tracking_enabled = False
    test_duration = duration

    return {
        "test_start_time": time.time(),
        "countdown_printed": {
            "wait2": False, "wait1": False, "start": False,
            "3sec": False, "2sec": False, "1sec": False,
            "move": False
        },
        "movement_start_time": None,
        "face_detection_checked": False,
        "last_move_second": -1,
        "last_printed_time": -1.0
    }

# ============================================================
# 디버깅 함수
# ============================================================
def debug_log(message, level="INFO", force=False):
    """
    디버깅 메시지 출력
    level: INFO, WARN, ERROR, DETAIL
    force: True이면 DEBUG_MODE 무시하고 항상 출력
    """
    if not DEBUG_MODE and not force:
        return
    
    if level == "DETAIL" and not DEBUG_DETAIL:
        return
    
    timestamp = time.strftime("%H:%M:%S")
    
    if level == "ERROR":
        prefix = "❌ [ERROR]"
    elif level == "WARN":
        prefix = "⚠️  [WARN]"
    elif level == "DETAIL":
        prefix = "🔍 [DETAIL]"
    else:
        prefix = "ℹ️  [INFO]"
    
    print(f"{timestamp} {prefix} {message}")

# ============================================================
# 도우미 함수
# ============================================================
# 영상, 사진 저장
def get_new_filename(base_name="output", ext="avi"):
    existing = os.listdir(desktop_path)
    pat = re.compile(rf"{re.escape(base_name)}_(\d+)\.{re.escape(ext)}")
    nums = [int(m.group(1)) for f in existing if (m := pat.match(f))]
    n = max(nums, default=0) + 1
    filename = os.path.join(desktop_path, f"{base_name}_{n}.{ext}")
    debug_log(f"새 파일명 생성: {os.path.basename(filename)}", "DETAIL")
    return filename

def get_new_image_filename(base_name="shot", ext="jpg"):
    return get_new_filename(base_name, ext)

def est_speed_px_per_s(cx, cy, prev_cx, prev_cy, dt):
    if prev_cx is None or dt <= 0:
        return 0.0
    dx = float(cx - prev_cx)
    if dx < 5:
        dx = 0
    dy = float(cy - prev_cy)
    if dy < 5:
        dy = 0
    speed = (dx*dx + dy*dy) ** 0.5 / max(dt, 1e-6)
    if speed > 100 and DEBUG_DETAIL:
        debug_log(f"높은 속도 감지: {speed:.1f} px/s", "DETAIL")
    return speed

def should_freeze(axis, now):
    frozen = now - motor_freeze_time[axis] < FREEZE_DURATION_S
    if frozen:
        debug_counters["motor_frozen"] += 1
        # if DEBUG_DETAIL:
            # debug_log(f"Freeze 활성: {axis}축", "DETAIL")
    return frozen

def update_freeze_timer(ddx, ddy, ddz, now):
    if ddx == 0:
        motor_freeze_time["x"] = now
        # debug_log(f"Freeze 타이머 시작: X축", "DETAIL")
    if ddy == 0:
        motor_freeze_time["y"] = now
        # debug_log(f"Freeze 타이머 시작: Y축", "DETAIL")
    if ddz == 0:
        motor_freeze_time["z"] = now
        # debug_log(f"Freeze 타이머 시작: Z축", "DETAIL")

# ============================================================
# 방법 A: 안전하고 빠른 Step 제어
# ============================================================
def compute_motor_angles_safe(center_x, center_y, area, frame_shape):
    """
    거리별 차등 스텝 크기로 빠른 추적 + 안정성 확보
    """
    frame_h, frame_w = frame_shape[:2]
    dx = center_x - (frame_w // 2)
    dy = center_y - (frame_h // 2)
    dz = DESIRED_FACE_AREA - area
    
    distance = np.sqrt(dx**2 + dy**2)
    
    # 거리별 스텝 크기 및 delay 조정 (떨림 개선: step 감소, delay 최소화)
    if distance > 150:
        step = 3  # 5 → 3 (기계적 진동 감소)
        delay = 20  # 80 → 20 (갱신률 향상: 12.5Hz → 50Hz)
        dist_category = "아주 멀리"
    elif distance > 80:
        step = 2  # 3 → 2
        delay = 15  # 60 → 15 (갱신률 향상: 16.7Hz → 66Hz)
        dist_category = "중간"
    else:
        step = 1
        delay = 15  # 40 → 15 (갱신률 향상: 25Hz → 66Hz)
        dist_category = "가까이"
    
    # 데드존 처리
    ddx = 0 if abs(dx) <= DEADZONE_XY else (-step if dx > 0 else step)
    ddy = 0 if abs(dy) <= DEADZONE_XY else (-step if dy > 0 else step)
    ddz = 0 if abs(dz) <= DEADZONE_AREA else (1 if dz > 0 else -1)
    
    debug_log(f"모터 계산: 거리={distance:.0f}px ({dist_category}), "
              f"오차=({dx:+.0f},{dy:+.0f}), 스텝=({ddx:+d},{ddy:+d})", "DETAIL")
    
    return {
        "motor_1": -ddx,
        "motor_2": 0,
        "motor_3": -ddy,
        "motor_4": 0,
        "motor_5": 0,
        "motor_6": 0,
        "motor_7": delay
    }

def clip_motor_angles(motor_cmds, limits=(-80, 80)):
    clipped = {}
    clipped_flag = False
    for k, v_float in motor_cmds.items():
        v = int(v_float)
        if k == "motor_7":
            clipped[k] = int(np.clip(v, 10, 500))
        else:
            original = v
            v = int(np.clip(v, limits[0], limits[1]))
            if v != original:
                clipped_flag = True
                debug_log(f"{k} 클리핑: {original} → {v}", "WARN")
            clipped[k] = v
    
    if clipped_flag:
        debug_log(f"각도 제한 적용됨", "WARN")
    
    return clipped

# ============================================================
# One Euro 필터
# ============================================================
class OneEuro:
    def __init__(self, min_cutoff=0.8, beta=0.04, dcutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None
        debug_log(f"OneEuro 필터 초기화: cutoff={min_cutoff}, beta={beta}", "DETAIL")
    
    @staticmethod
    def alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
    
    def filter(self, x, t):
        if self.t_prev is None:
            self.t_prev, self.x_prev = t, float(x)
            return float(x)
        dt = max(1e-3, t - self.t_prev)
        dx = (x - self.x_prev) / dt
        a_d = OneEuro.alpha(self.dcutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = OneEuro.alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.t_prev, self.x_prev, self.dx_prev = t, x_hat, dx_hat
        return float(x_hat)

# ============================================================
# 캡처 스레드
# ============================================================
class CaptureThread:
    def __init__(self, cam_index=1, backend=cv2.CAP_DSHOW):
        debug_log(f"카메라 초기화 시작: index={cam_index}", "INFO", force=True)
        self.cap = cv2.VideoCapture(cam_index, backend)

        # 포맷을 먼저 못박아 두는 게 협상 지연을 줄이는 데 도움됨
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS,          CAP_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.0)   # 일부 장치는 0.25=auto, 0.0=manual (반대인 경우도 있어 둘 다 시도)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)         # 장치마다 -5~-8 범위 테스트


        if not self.cap.isOpened():
            debug_log("카메라 열기 실패!", "ERROR", force=True)
            raise RuntimeError("카메라 열기 실패")

        # 워밍업: 캡이 실제 스트리밍을 시작하도록 첫 0.5~1초간 프레임 버림
        t0 = time.time()
        # 워밍업: 초기 자동노출/포커스 안정화용
        for _ in range(20):  # 약 20프레임 버리기 (0.3~0.5초)
            self.cap.grab()

        actual_w  = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h  = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps= self.cap.get(cv2.CAP_PROP_FPS)
        debug_log(f"카메라 설정: {actual_w}x{actual_h} @ {actual_fps}fps", "INFO", force=True)

        self.lock = threading.Lock()
        self.latest = None
        self.running = True
        self.frame_count = 0
        self.th = threading.Thread(target=self.loop, daemon=True)
        self.th.start()
        debug_log("캡처 스레드 시작됨", "INFO", force=True)

    def loop(self):
        # 오래된 프레임을 덜어내고, 디코딩 비용을 줄이기 위한 루프
        DROP_OLD_FRAMES = True     # 필요 없으면 False

        while self.running:
            if DROP_OLD_FRAMES:
                 for _ in range(3):  # 프레임 3장 버리기
                    self.cap.grab()

            # grab으로 캡처 → retrieve로 디코딩 (read()보다 유연)
            ret = self.cap.grab()
            if not ret:
                debug_log("프레임 grab 실패", "WARN")
                continue

            ret, f = self.cap.retrieve()
            if ret:
                with self.lock:
                    self.latest = f
                    self.frame_count += 1
            else:
                debug_log("프레임 retrieve 실패", "WARN")

    def read(self):
        with self.lock:
            if self.latest is None:
                return False, None
            return True, self.latest.copy()
    
    def release(self):
        debug_log(f"캡처 스레드 종료 (총 {self.frame_count} 프레임)", "INFO", force=True)
        self.running = False
        self.th.join(timeout=0.5)
        self.cap.release()

# ============================================================
# 시리얼 워커 스레드 (통신 진단 강화)
# ============================================================
def serial_worker(q, port, baud):
    global move_ready, debug_counters, serial_health
    
    debug_log(f"시리얼 연결 시도: {port} @ {baud}bps", "INFO", force=True)
    
    # 연결 시도
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        debug_log(f"시리얼 연결 완료: {port}", "INFO", force=True)
        
        # ⭐ 연결 테스트
        test_msg = "0,0,0,0,0,0,100\n"
        ser.write(test_msg.encode('utf-8'))
        debug_log(f"초기 테스트 신호 전송: {test_msg.strip()}", "INFO", force=True)
        time.sleep(0.2)
        
        # 아두이노 응답 확인 (옵션)
        if ser.in_waiting > 0:
            response = ser.readline().decode('utf-8', errors='ignore').strip()
            debug_log(f"아두이노 응답: {response}", "INFO", force=True)
        else:
            debug_log(f"아두이노 응답 없음 (정상일 수 있음)", "WARN")
            
    except serial.SerialException as e:
        debug_log(f"시리얼 연결 실패: {e}", "ERROR", force=True)
        debug_log(f"", "ERROR", force=True)
        debug_log(f"🔧 문제 해결 방법:", "ERROR", force=True)
        debug_log(f"  1. 장치 관리자에서 COM 포트 확인", "ERROR", force=True)
        debug_log(f"  2. 아두이노 USB 재연결", "ERROR", force=True)
        debug_log(f"  3. 아두이노 IDE 시리얼 모니터 닫기", "ERROR", force=True)
        debug_log(f"  4. SERIAL_PORT 설정 확인 (현재: {port})", "ERROR", force=True)
        serial_health["connection_lost"] = True
        return
    except Exception as e:
        debug_log(f"알 수 없는 오류: {e}", "ERROR", force=True)
        serial_health["connection_lost"] = True
        return
    
    # ⭐ 시리얼 테스트 모드
    if DEBUG_SERIAL_TEST:
        debug_log("", "INFO", force=True)
        debug_log("=" * 70, "INFO", force=True)
        debug_log("🧪 시리얼 테스트 모드 시작", "INFO", force=True)
        debug_log("5초마다 테스트 신호를 전송합니다.", "INFO", force=True)
        debug_log("아두이노 시리얼 모니터를 열어 데이터를 확인하세요!", "INFO", force=True)
        debug_log("=" * 70, "INFO", force=True)
        debug_log("", "INFO", force=True)
        
        test_count = 0
        try:
            while True:
                test_count += 1
                test_values = [test_count % 10, 0, test_count % 10, 0, 0, 0, 1000]
                test_msg = ','.join(map(str, test_values)) + '\n'
                
                debug_log(f"", "INFO", force=True)
                debug_log(f"[테스트 #{test_count}] 전송: {test_msg.strip()}", "INFO", force=True)
                
                try:
                    ser.write(test_msg.encode('utf-8'))
                    debug_log(f"  ✅ 전송 성공", "INFO", force=True)
                    
                    # 아두이노 응답 확인
                    time.sleep(0.1)
                    if ser.in_waiting > 0:
                        response = ser.readline().decode('utf-8', errors='ignore').strip()
                        debug_log(f"  📩 아두이노 응답: {response}", "INFO", force=True)
                    else:
                        debug_log(f"  📭 아두이노 응답 없음", "WARN", force=True)
                        
                except Exception as e:
                    debug_log(f"  ❌ 전송 실패: {e}", "ERROR", force=True)
                
                time.sleep(5)
                
        except KeyboardInterrupt:
            debug_log("", "INFO", force=True)
            debug_log("테스트 모드 종료", "INFO", force=True)
            ser.close()
            return
    
    # 정상 동작 모드
    serial_health["last_success_time"] = time.time()
    
    try:
        while True:
            motor_cmds = q.get()
            if motor_cmds is None:
                debug_log("시리얼 종료 신호 수신", "INFO", force=True)
                break
            
            # 최신 명령만 사용 (큐 비우기)
            skip_count = 0
            while not q.empty():
                latest = q.get_nowait()
                if latest is not None:
                    motor_cmds = latest
                    skip_count += 1
                else:
                    break
            
            if skip_count > 0:
                debug_log(f"큐에서 {skip_count}개 명령 건너뜀", "DETAIL")
            
            if motor_cmds is None:
                break
            
            try:
                # motor_1 ~ motor_7 값 전송
                vals = [motor_cmds.get(f"motor_{i}", 0) for i in range(1, 8)]
                message = ','.join(map(str, vals)) + '\n'
                
                # ⭐ 전송 시도
                ser.write(message.encode('utf-8'))
                serial_health["total_sent"] += 1
                serial_health["last_success_time"] = time.time()
                serial_health["consecutive_errors"] = 0
                debug_counters["serial_sent"] += 1
                
                debug_log(f"시리얼 전송 #{debug_counters['serial_sent']}: {message.strip()}", "DETAIL")
                
                # ⭐ 주기적인 통신 상태 체크 (100번마다)
                if debug_counters["serial_sent"] % 100 == 0:
                    elapsed = time.time() - serial_health["last_success_time"]
                    error_rate = (serial_health["total_errors"] / serial_health["total_sent"] * 100) if serial_health["total_sent"] > 0 else 0
                    
                    if error_rate > 5:
                        debug_log(f"⚠️  시리얼 오류율 높음: {error_rate:.1f}% ({serial_health['total_errors']}/{serial_health['total_sent']})", "WARN")
                    else:
                        debug_log(f"✅ 시리얼 통신 양호: 오류율 {error_rate:.1f}%", "INFO")
                
                # delay 대기 (move_ready 플래그)
                delay_ms = motor_cmds.get("motor_7", 50)
                move_ready.clear()
                time.sleep(delay_ms / 1000.0)
                move_ready.set()
                
            except serial.SerialException as e:
                serial_health["total_errors"] += 1
                serial_health["consecutive_errors"] += 1
                debug_counters["serial_error"] += 1
                
                debug_log(f"시리얼 쓰기 오류 #{debug_counters['serial_error']}: {e}", "ERROR", force=True)
                
                # ⭐ 연속 오류 감지
                if serial_health["consecutive_errors"] >= 5:
                    debug_log(f"", "ERROR", force=True)
                    debug_log(f"❌ 심각: 연속 {serial_health['consecutive_errors']}회 오류!", "ERROR", force=True)
                    debug_log(f"", "ERROR", force=True)
                    debug_log(f"🔧 가능한 원인:", "ERROR", force=True)
                    debug_log(f"  1. USB 케이블 불량 또는 연결 불안정", "ERROR", force=True)
                    debug_log(f"  2. 아두이노 전원 부족", "ERROR", force=True)
                    debug_log(f"  3. 아두이노 처리 속도 느림 (버퍼 오버플로우)", "ERROR", force=True)
                    debug_log(f"  4. 아두이노 코드에서 Serial.read() 안 함", "ERROR", force=True)
                    debug_log(f"", "ERROR", force=True)
                    debug_log(f"💡 해결 시도:", "ERROR", force=True)
                    debug_log(f"  - USB 재연결", "ERROR", force=True)
                    debug_log(f"  - 아두이노 리셋", "ERROR", force=True)
                    debug_log(f"  - delay 값 증가 (45 → 100)", "ERROR", force=True)
                    debug_log(f"", "ERROR", force=True)
                    
                    serial_health["connection_lost"] = True
                    break
                    
            except Exception as e:
                serial_health["total_errors"] += 1
                debug_counters["serial_error"] += 1
                debug_log(f"예상치 못한 오류 #{debug_counters['serial_error']}: {e}", "ERROR", force=True)
                
    finally:
        ser.close()
        debug_log(f"시리얼 종료 (전송: {serial_health['total_sent']}회, "
                  f"오류: {serial_health['total_errors']}회)", "INFO", force=True)
        
        # 최종 진단
        if serial_health["total_sent"] > 0:
            error_rate = (serial_health["total_errors"] / serial_health["total_sent"] * 100)
            if error_rate > 10:
                debug_log(f"", "WARN", force=True)
                debug_log(f"⚠️  시리얼 통신 품질 나쁨: 오류율 {error_rate:.1f}%", "WARN", force=True)
                debug_log(f"   하드웨어 연결 상태를 확인하세요!", "WARN", force=True)
            elif error_rate > 0:
                debug_log(f"✅ 시리얼 통신 정상 종료: 오류율 {error_rate:.1f}%", "INFO", force=True)
            else:
                debug_log(f"✅ 시리얼 통신 완벽: 오류 없음!", "INFO", force=True)

# ============================================================
# 얼굴 DNN
# ============================================================
prototxt_path = r"C:\face_models\deploy.prototxt"
model_path = r"C:\face_models\res10_300x300_ssd_iter_140000.caffemodel"

debug_log("DNN 모델 로드 시작", "INFO", force=True)
debug_log(f"  prototxt: {prototxt_path}", "DETAIL")
debug_log(f"  model: {model_path}", "DETAIL")

try:
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    debug_log("DNN 모델 로드 성공", "INFO", force=True)
except Exception as e:
    debug_log(f"DNN 모델 로드 실패: {e}", "ERROR", force=True)
    raise

def detect_faces_dnn(frame, conf_thresh=0.5):
    frame_h, frame_w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104.0,177.0,123.0))
    net.setInput(blob)
    det = net.forward()
    boxes = []
    for i in range(det.shape[2]):
        conf = float(det[0,0,i,2])
        if conf > conf_thresh:
            x1,y1,x2,y2 = (det[0,0,i,3:7] * np.array([frame_w,frame_h,frame_w,frame_h])).astype(int)
            x1,y1,x2,y2 = max(0,x1),max(0,y1),min(frame_w-1,x2),min(frame_h-1,y2)
            boxes.append((x1,y1,x2-x1,y2-y1))
            debug_log(f"얼굴 검출: conf={conf:.2f}, bbox=({x1},{y1},{x2-x1},{y2-y1})", "DETAIL")
    
    if boxes:
        debug_counters["face_detected"] += 1
    
    return boxes

# ============================================================
# 칼만 필터
# ============================================================
def init_kalman():
    debug_log("칼만 필터 초기화", "INFO")
    kf = cv2.KalmanFilter(4,2)
    kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kf.processNoiseCov = np.diag([1e-2,1e-2,1e-1,1e-1]).astype(np.float32)
    kf.measurementNoiseCov = np.diag([2.0,2.0]).astype(np.float32)
    kf.errorCovPost = np.diag([10,10,10,10]).astype(np.float32)
    return kf

def kalman_predict(kf, dt):
    kf.transitionMatrix = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], np.float32)
    pred = kf.predict()
    px, py = int(pred[0,0]), int(pred[1,0])
    debug_log(f"칼만 예측: ({px}, {py})", "DETAIL")
    return px, py

def kalman_correct(kf, x, y):
    kf.correct(np.array([[np.float32(x)],[np.float32(y)]], np.float32))
    cx, cy = int(kf.statePost[0,0]), int(kf.statePost[1,0])
    debug_log(f"칼만 보정: 측정=({x},{y}) → 추정=({cx},{cy})", "DETAIL")

# ============================================================
# 메인
# ============================================================
def main():
    global icr3_phase, icr3_center, icr3_t0, icr3_inside, icr3_total
    global _prev_cx, _prev_cy, _prev_t, reacquire_t0, ICR_RADIUS, matric3_text
    global debug_counters
    global test_mode_active, test_phase, test_start_time, test_stop_start_time
    global test_coordinates, test_reference_point
    global tracking_test_mode, tracking_enabled, test_duration

    print("\n" + "=" * 70)
    print("🎥 얼굴 추적 로봇팔 제어 시스템 (방법 A)")
    print("=" * 70)
    print(f"디버깅 모드: {'🟢 ON' if DEBUG_MODE else '🔴 OFF'}")
    if DEBUG_MODE:
        print(f"상세 디버깅: {'🟢 ON' if DEBUG_DETAIL else '🔴 OFF'}")
    if DEBUG_SERIAL_TEST:
        print(f"시리얼 테스트 모드: 🟢 ON")
        print("  → 5초마다 테스트 신호를 전송합니다.")
        print("  → 아두이노 시리얼 모니터를 열어 확인하세요!")
    print("=" * 70)
    print("키 조작:")
    print("  i     : 평가지표 1 테스트 (얼굴 검출 비율)")
    print("  o     : 평가지표 3 테스트 (원 내부 비율)")
    print("  s     : 녹화 시작")
    print("  e     : 녹화 종료")
    print("  1~9   : 연속 촬영")
    print("  q     : 종료")
    print("=" * 70)
    print()
    
    # ⭐ 시리얼 테스트 모드일 때는 카메라 없이 실행
    if DEBUG_SERIAL_TEST:
        debug_log("시리얼 테스트 전용 모드 시작", "INFO", force=True)
        q = queue.Queue()
        serial_thread = threading.Thread(target=serial_worker, args=(q, SERIAL_PORT, SERIAL_BAUD), daemon=True)
        serial_thread.start()
        
        try:
            serial_thread.join()  # 시리얼 스레드가 끝날 때까지 대기
        except KeyboardInterrupt:
            debug_log("KeyboardInterrupt - 종료", "INFO", force=True)
        finally:
            q.put(None)
        return

    # 스레드 준비
    q = queue.Queue()
    threading.Thread(target=serial_worker, args=(q, SERIAL_PORT, SERIAL_BAUD), daemon=True).start()
    cap_thread = CaptureThread()

    # ⭐ 평가지표 1 테스트 변수
    test1_vars = {
        "test_start_time": None,
        "countdown_printed": {},
        "movement_start_time": None,
        "face_detection_checked": False,
        "last_move_second": -1,
        "last_printed_time": -1.0
    }

    # ⭐ 평가지표 3 테스트 변수
    test3_countdown_printed = {}

    print("\n" + "=" * 70)
    print("🧪 추적 성능 테스트")
    print("=" * 70)
    print("📌 'i' 키: 평가지표 1 (얼굴 검출 비율 테스트)")
    print("📌 'o' 키: 평가지표 3 (원 내부 비율 테스트)")
    print("=" * 70)
    print()

    # 비디오 저장
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    recording, out = False, None

    # 상태
    kf = init_kalman()
    kalman_inited = False
    last_kf_ts = time.time()

    # 표시 스무딩
    cx_oe = OneEuro(0.9, 0.04, 1.2)
    cy_oe = OneEuro(0.9, 0.05, 1.2)

    # 모터 제어 스무딩 (더 강한 필터)
    motor_cx_oe = OneEuro(0.4, 0.02, 1.0)
    motor_cy_oe = OneEuro(0.4, 0.02, 1.0)

    ever_locked = False
    LOG_INTERVAL, last_log = 0.3, 0.0

    # 연속촬영
    photo_interval = 3.0
    photo_shooting = False
    photo_count = 0
    photo_taken = 0
    next_shot_at = 0.0

    MSG_DUR = 1.2
    msg_lt_text, msg_lt_until = "", 0.0
    msg_lt_display = False
    msg_rt_text, msg_rt_until = "", 0.0
    msg_rt_display = False

    face_boxes_preFrame = []

    debug_log("메인 루프 시작", "INFO", force=True)
    print()

    try:
        frame_idx = 0
        frame_per_sec = 0
        frame_idx_per_sec = 0
        sum_time_per_sec = 0

        box_l=box_t=box_w=box_h=box_cx=box_cy=0
        area = 0
        pre_frame_time = 0

        ##-----------------------------------------------------------------------------------------
        ## 251025_MJ_떨림 보정을 위한 이전과 현재 Frame의 Data
        ##-----------------------------------------------------------------------------------------
        
        pre_gray = None     # 이전 Frame Image (알고리즘 속도를 위해 Color가 아닌 Gray 영상으로 저장)
        cur_gray = None     # 현재 Frame Image (알고리즘 속도를 위해 Color가 아닌 Gray 영상으로 저장)
        pre_pts = None      # 이전 Frame의 Feature Point 위치
        cur_pts = None      # 현재 Frame의 Feature Point 위치
        comp_frame_cx = 0   # 떨림 보정을 위한 Frame Center X
        comp_frame_cy = 0   # 떨림 보정을 위한 Frame Center Y
        ##-----------------------------------------------------------------------------------------

        while True:
            ok, frame = cap_thread.read()
            if not ok:
                # debug_log("프레임 읽기 실패", "WARN")
                continue
            frame_resize = cv2.resize(frame, (100,100))
            cur_gray = cv2.cvtColor(frame_resize,cv2.COLOR_BGR2GRAY)
            now = time.time()
            debug_counters["frame_count"] += 1

            # ⭐⭐⭐ 평가지표 1 카운트다운 로직 (dh1_code.py 방식) ⭐⭐⭐
            if tracking_test_mode and test1_vars["test_start_time"] is not None:
                elapsed = now - test1_vars["test_start_time"]
                countdown_printed = test1_vars["countdown_printed"]
                movement_start_time = test1_vars["movement_start_time"]
                face_detection_checked = test1_vars["face_detection_checked"]
                last_move_second = test1_vars["last_move_second"]
                last_printed_time = test1_vars["last_printed_time"]

                if not countdown_printed.get("wait2") and elapsed >= 0:
                    print("⏳ 2초 대기 중...")
                    countdown_printed["wait2"] = True

                if not countdown_printed.get("wait1") and elapsed >= 1:
                    print("⏳ 1초 대기 중...")
                    countdown_printed["wait1"] = True

                if not countdown_printed.get("start") and elapsed >= 2:
                    print("🔔 카운터 시작")
                    countdown_printed["start"] = True

                if not countdown_printed.get("3sec") and elapsed >= 3:
                    print("⏱️  3초")
                    countdown_printed["3sec"] = True

                if not countdown_printed.get("2sec") and elapsed >= 4:
                    print("⏱️  2초")
                    countdown_printed["2sec"] = True

                if not countdown_printed.get("1sec") and elapsed >= 5:
                    print("⏱️  1초")
                    countdown_printed["1sec"] = True

                if not countdown_printed.get("move") and elapsed >= 6:
                    print(f"\n🚀 움직임 시작! 지금 좌우로 움직이세요! (테스트 시간: {test_duration}초)\n")
                    test1_vars["movement_start_time"] = now
                    movement_start_time = now
                    tracking_enabled = True
                    test1_vars["last_move_second"] = -1
                    test1_vars["last_printed_time"] = -1.0
                    debug_log("로봇팔 추적 활성화됨", "INFO", force=True)
                    countdown_printed["move"] = True

                # 움직임 경과 시간 출력
                if movement_start_time and not face_detection_checked:
                    elapsed_move = now - movement_start_time

                    current_second = int(elapsed_move)
                    if current_second > test1_vars["last_move_second"] and current_second < int(test_duration):
                        print(f"⏱️  움직임 경과: {current_second}초")
                        test1_vars["last_move_second"] = current_second

                    if elapsed_move >= test_duration and test1_vars["last_printed_time"] < test_duration:
                        print(f"⏱️  움직임 경과: {test_duration}초")
                        test1_vars["last_printed_time"] = test_duration

                # 얼굴 검출 체크 (항상 2초 후)
                if movement_start_time and not face_detection_checked and (now - movement_start_time) >= DETECTION_TIME:
                    if (now - movement_start_time) >= (DETECTION_TIME + 0.2):
                        test1_vars["face_detection_checked"] = True
                        print(f"⏱️  {DETECTION_TIME + 0.2:.1f}초 경과 - 추적 결과 확인 중...\n")

                        ratio = recent_face_ratio()
                        face_detected = (ratio >= 60.0)

                        print("=" * 70)
                        print("📊 평가지표 1 - 추적 테스트 결과")
                        print("=" * 70)
                        print(f"⏱️  움직임 시간: {test_duration}초")
                        print(f"⏱️  검출 체크 시간: {DETECTION_TIME}초")
                        print(f"🎯 얼굴 검출 비율: {ratio:.1f}%")

                        if face_detected:
                            print("✅ 성공: 로봇팔이 사용자를 성공적으로 추적했습니다!")
                            print(f"   → {DETECTION_TIME}초 후에도 얼굴이 카메라 영역 내에 유지되었습니다.")
                        else:
                            print("❌ 실패: 로봇팔이 사용자를 놓쳤습니다!")
                            print(f"   → {DETECTION_TIME}초 후 얼굴이 카메라 영역을 벗어났습니다.")
                        print("=" * 70)
                        print("🔄 정상 추적 모드로 전환합니다...")
                        print("💡 'i' 키를 눌러 다시 테스트할 수 있습니다.\n")
                        tracking_test_mode = False

            # ⭐⭐⭐ 평가지표 3 카운트다운 로직 (원 내부 비율) ⭐⭐⭐
            if test_mode_active:
                elapsed_test = now - test_start_time

                if test_phase == "waiting":
                    if not test3_countdown_printed.get("3sec") and elapsed_test >= 1:
                        print("⏱️  3초")
                        test3_countdown_printed["3sec"] = True

                    if not test3_countdown_printed.get("2sec") and elapsed_test >= 2:
                        print("⏱️  2초")
                        test3_countdown_printed["2sec"] = True

                    if not test3_countdown_printed.get("1sec") and elapsed_test >= 3:
                        print("⏱️  1초")
                        test3_countdown_printed["1sec"] = True

                    if not test3_countdown_printed.get("move_start") and elapsed_test >= 4:
                        print("\n🚀 사용자 움직임 시작! 지금 좌우 또는 상하로 움직이세요! (3초)\n")
                        test_phase = "moving"
                        test3_countdown_printed["move_start"] = True

                if test_phase == "moving":
                    if not test3_countdown_printed.get("moving_2sec") and elapsed_test >= 5:
                        print("⏱️  2초")
                        test3_countdown_printed["moving_2sec"] = True

                    if not test3_countdown_printed.get("moving_1sec") and elapsed_test >= 6:
                        print("⏱️  1초")
                        test3_countdown_printed["moving_1sec"] = True

                    if not test3_countdown_printed.get("stop_start") and elapsed_test >= 7:
                        print("\n⏸️  움직임 멈춤! 3초간 정지하세요!\n")
                        test_phase = "stopping"
                        test_stop_start_time = now
                        test_coordinates = []
                        test_reference_point = None
                        test3_countdown_printed["stop_start"] = True

                if test_phase == "stopping":
                    stop_elapsed = now - test_stop_start_time

                    if not test3_countdown_printed.get("stop_2sec") and stop_elapsed >= 1:
                        print("⏱️  2초")
                        test3_countdown_printed["stop_2sec"] = True

                    if not test3_countdown_printed.get("stop_1sec") and stop_elapsed >= 2:
                        print("⏱️  1초")
                        test3_countdown_printed["stop_1sec"] = True

                    if stop_elapsed < 3.0:  # 3초간 수집
                        if len(face_boxes_preFrame) > 0:
                            # ✅ 수정: 화면에 표시되는 크롭 후 좌표 사용
                            test_coordinates.append((disp_addapt_size_ori_cx, disp_addapt_size_ori_cy))

                            if test_reference_point is None:
                                # ✅ 수정: 크롭 후 프레임 중심 사용
                                test_reference_point = (display_w // 2, display_h // 2)
                                debug_log(f"기준점 설정 (화면 중심): {test_reference_point}", "INFO", force=True)

                    elif stop_elapsed >= 3.0:
                        test_mode_active = False
                        test_phase = "done"

                        if test_reference_point and len(test_coordinates) > 0:
                            # ✅ 수정: 크롭 후 프레임 가로 기준으로 반지름 설정
                            TEST_CIRCLE_RADIUS = int(display_w * CIRCLE_RADIUS_RATIO)
                            print(f"📏 원의 반지름: {TEST_CIRCLE_RADIUS}px (화면 가로의 {CIRCLE_RADIUS_RATIO*100}%, 지름 {TEST_CIRCLE_RADIUS*2}px)")

                            inside_count = 0
                            total_count = len(test_coordinates)

                            for (cx, cy) in test_coordinates:
                                distance = np.sqrt((cx - test_reference_point[0])**2 +
                                                (cy - test_reference_point[1])**2)
                                if distance <= TEST_CIRCLE_RADIUS:
                                    inside_count += 1

                            ratio = (inside_count / total_count * 100) if total_count > 0 else 0

                            print("=" * 70)
                            print("📊 평가지표 3 - 추적 안정성 테스트 결과")
                            print("=" * 70)
                            print(f"🎯 기준점: {test_reference_point}")
                            print(f"📏 원의 반지름: {TEST_CIRCLE_RADIUS}px (화면 가로의 {CIRCLE_RADIUS_RATIO*100}%, 지름 {TEST_CIRCLE_RADIUS*2}px)")
                            print(f"📍 수집된 좌표 개수: {total_count}개")
                            print(f"✅ 원 내부 좌표: {inside_count}개")
                            print(f"❌ 원 외부 좌표: {total_count - inside_count}개")
                            print(f"📈 원 내부 비율: {ratio:.2f}%")
                            print("=" * 70)

                            # ⭐ 목표 80% 달성 여부
                            if ratio >= 80:
                                print("✅ 목표 달성! 매우 안정적인 추적!")
                            elif ratio >= 70:
                                print("🟢 양호: 목표에 근접한 추적 (70% 이상)")
                            elif ratio >= 60:
                                print("🟡 보통: 추적 성능 개선 필요 (60~70%)")
                            else:
                                print("🔴 불량: 추적 안정성이 낮음 (60% 미만)")

                            print("\n정상 모드로 전환합니다...")
                            print("💡 'o' 키를 눌러 다시 테스트할 수 있습니다.\n")
                        else:
                            print("⚠️  테스트 실패: 좌표를 수집하지 못했습니다.")
                            print("   얼굴이 검출되지 않았거나 추적이 실패했습니다.\n")

            sum_time_per_sec += (now-pre_frame_time)
            frame_idx_per_sec += 1
            if sum_time_per_sec > 1.0:
                frame_per_sec = frame_idx_per_sec
                debug_log(f"FPS: {frame_per_sec} | "
                         f"얼굴검출: {debug_counters['face_detected']} | "
                         f"시리얼: {debug_counters['serial_sent']}/{debug_counters['serial_error']}",
                         "INFO")
                sum_time_per_sec = 0
                frame_idx_per_sec = 0

            frame = cv2.flip(frame,1)
            frame_h, frame_w = frame.shape[:2]
            #frame_cx = frame_h//2
            #frame_cy = frame_w//2

            if ICR_RADIUS <= 0:
                ICR_RADIUS = int(frame_w * CIRCLE_RADIUS_RATIO)
                debug_log(f"ICR 반경 설정: {ICR_RADIUS}px (화면 가로의 {CIRCLE_RADIUS_RATIO*100}%)", "INFO")

            frame_idx += 1
            do_detect = (frame_idx % DETECT_EVERY == 0)

            dt_kf = max(1e-3, now - last_kf_ts)
            last_kf_ts = now

            # 얼굴 검출
            if do_detect:
                face_boxes = detect_faces_dnn(frame)
                face_boxes_preFrame = face_boxes
            else:
                face_boxes = face_boxes_preFrame

            face_found = len(face_boxes) > 0

            # ⭐ 평가지표 1을 위한 얼굴 검출 기록
            update_face_presence(now, face_found)

            # 얼굴 상태 변화 로깅
            if face_found and reacquire_t0 is not None:
                reacq = now - reacquire_t0
                metric1_times.append(reacq)
                # debug_log(f"얼굴 재인식 완료: {reacq:.3f}초", "INFO")
                reacquire_t0 = None
            elif not face_found and reacquire_t0 is None:
                reacquire_t0 = now
                debug_counters["face_lost"] += 1
                # debug_log(f"얼굴 손실 (#{debug_counters['face_lost']})", "WARN")

            if face_found:
                face_boxes.sort(key=lambda b: b[2]*b[3], reverse=True)
                box_l, box_t, box_w, box_h = face_boxes[0]
                box_cx, box_cy = box_l + round(box_w/2), round(box_t + box_h/2)
                area = box_w*box_h
                
                if not ever_locked:
                    ever_locked = True
                    debug_log(f"첫 얼굴 락온! 위치=({box_cx},{box_cy}), 크기={box_w}x{box_h}", "INFO")

                if not kalman_inited:
                    kf.statePost = np.array([[box_cx],[box_cy],[0],[0]], np.float32)
                    kalman_inited = True
                    debug_log(f"칼만 필터 초기화 완료", "INFO")
                
                kpx, kpy = kalman_predict(kf, dt_kf)
                kalman_correct(kf, box_cx, box_cy)
                kpx, kpy = int(kf.statePost[0,0]), int(kf.statePost[1,0])
            else:
                if kalman_inited:
                    kpx, kpy = kalman_predict(kf, dt_kf)
                else:
                    kpx, kpy = (frame_w//2, frame_h//2)
            
            use_cx, use_cy = kpx, kpy
            if kalman_inited:
                use_cx += int(kf.statePost[2,0] * LEAD_FACE_SEC)
                use_cy += int(kf.statePost[3,0] * LEAD_FACE_SEC)
            
            ##-----------------------------------------------------------------
            ## ⚡ 로봇팔 제어 (평가지표 1에서는 tracking_enabled 조건 추가)
            ##-----------------------------------------------------------------
            # 평가지표 1 모드에서는 tracking_enabled가 True일 때만 추적
            # 평가지표 3 모드나 일반 모드에서는 항상 추적
            should_track = (not tracking_test_mode) or tracking_enabled

            if face_found and move_ready.is_set() and should_track:
                debug_log(f"로봇팔 제어 시작", "DETAIL")

                # 모터 제어용 필터 적용 (DNN 박스 떨림 제거)
                filtered_motor_cx = int(motor_cx_oe.filter(box_cx, now))
                filtered_motor_cy = int(motor_cy_oe.filter(box_cy, now))

                angles = compute_motor_angles_safe(filtered_motor_cx, filtered_motor_cy, area, frame.shape)
                
                # Freeze 타이머 업데이트
                dx_val = box_cx - (frame_w // 2)
                dy_val = box_cy - (frame_h // 2)
                dz_val = DESIRED_FACE_AREA - area
                ddx = 0 if abs(dx_val) <= DEADZONE_XY else (-1 if dx_val > 0 else 1)
                ddy = 0 if abs(dy_val) <= DEADZONE_XY else (-1 if dy_val > 0 else 1)
                ddz = 0 if abs(dz_val) <= DEADZONE_AREA else (1 if dz_val > 0 else -1)
                update_freeze_timer(ddx, ddy, ddz, now)
                
                # Freeze 체크
                freeze_applied = False
                if should_freeze("x", now):
                    angles["motor_1"] = 0
                    freeze_applied = True
                if should_freeze("y", now):
                    angles["motor_3"] = 0
                    freeze_applied = True
                if should_freeze("z", now):
                    angles["motor_4"] = 0
                    angles["motor_5"] = 0
                    angles["motor_6"] = 0
                    freeze_applied = True
                
                if freeze_applied:
                    debug_log(f"Freeze 적용됨", "DETAIL")
                
                clipped_angles = clip_motor_angles(angles)
                
                if not q.full():
                    q.put(clipped_angles)
                    debug_log(f"모터 명령 큐 추가 (큐 크기: {q.qsize()})", "DETAIL")
                else:
                    debug_log(f"모터 명령 큐 가득 참!", "WARN")
                    
            elif not face_found and ever_locked:
                debug_log(f"얼굴 없음 - 정지 명령", "DETAIL")
                stop_cmd = {f"motor_{i}": 0 for i in range(1, 7)}
                stop_cmd["motor_7"] = 50
                if not q.full():
                    q.put(stop_cmd)
            elif not move_ready.is_set():
                debug_log(f"move_ready 대기 중...", "DETAIL")
            ##-----------------------------------------------------------------

            # 화면 표시용 스무딩
            #disp_kf_cx = int(cx_oe.filter(use_cx, now)) # kalman
            #disp_kf_cy = int(cy_oe.filter(use_cy, now)) # kalmal
            #disp_kf_cx = frame_cx# original
            #disp_kf_cy = frame_cy# original

            ##-----------------------------------------------------------------
            ## 251025_MJ_Image의 떨림을 분석해서 Frame 처리 나눔
            ##-----------------------------------------------------------------
                
            """
            떨림의 판단은 이전 Frame, 현재 Frame간의 Feature 거리로 판단 (임의로 50px로 고정해놨음)
                현재 문제가 Image Center 판단을 DNN box Center로 하는데 그게 DNN에서 얼굴 Box Size가 변함에 따라서도 Center가 흔들리는 현상이 있음.
                그 흔들리는 Center위치로 Image Center를 강제로 고정하여 움직이니까 멈춰 있는 Image 에서도 떨리는 현상 발생
                그래서 One Euro로 떨림을 완화 시켰는데 이걸 적용하면 문제가 모터가 떨려서 Image가 흔들릴때 그걸 보정하지 못함 (보정은 하는데 부드럽게 떨리니까 흔들리는게 보정 안됨)
                그래서 Frame간에 얼굴을 제외한 배경의 떨림을 분석해서 box Center와 One euro 보정을 나누어서 적용함.

            1. 떨림이 심할 경우 (모터 작동 중이거나 모터가 흔들릴때. 이경우에는 Image 전체가 떨린다)
                - 판단 기준 : 배경의 이전/지금 Frame간 Feature의 이동 거리 전체 평균이 DEADZONE_XY//2 이상일때
                - 알고리즘  : Box Center로 강제 지정
                  (이렇게한 이유는 Image 전체가 움직일 경우에는 얼굴 center인 Box Center로 하는게 그나마 덜 떨려보임)

            2. 떨림이 없을 경우 (가만히 있는데 얼굴의 Box만 떨릴때, 이 경우에는 배경은 안떨리고 멀쩡하기 때문에 최대한 영상처리로 인해 슬쩍 센터 맞춘다) 
                - 판단 기준 : 배경의 이전/지금 Frame간 Feature의 이동 거리 전체 평균이 DEADZONE_XY//2 이하일때
                - 알고리즘 : 두가지의 경우로 나뉘어 조치한다.
                1) box center가 Image Center에서 가까이 있을 경우 
                    - One Euro로 보정된 Center로 지정
            """

            if(comp_frame_cx is 0):                 # 보정 Frame Cetner Data가 없을 경우
                comp_frame_cx = frame_w // 2        # 실제 Frame Center

            if(comp_frame_cy is 0):                 # 보정 Frame Cetner Data가 없을 경우
                comp_frame_cy = frame_h // 2        # 실제 Frame Center

            average_dist = 0                        # 이전 Frame과 현재 Frame 배경 이동의 거리 평균
            average_count = 0                       # average_dist를 구하기 위한 Featur Count
            DEF_MAX_SHAKE_DISTANCE = DEADZONE_XY//2 # Image 떨림을 판단하는 기준 (DEADZONE_XY//2 로 한 이유는 이동이 DEADZONE_XY 이상으로 하면 애초에 모터가 움직이기 때문에 무조건 떨리게 되어있음)
            DEF_MIN_FRAME_CENTER_DISTANCE = 5       # 실제 Frame의 Center와 Box의 거리 (이 거리 이하일 경우에는 Image 보정하지 않고 그대로 둬야 안떨림. 이거 이상일 경우에는 원유로로 스무스하게 보정)            
            fix_image_center = True                 # True일 경우 강제로 Image Center 지정

            if( pre_gray is None ):                 # 이전 Frame이 없으면 그냥 현재 Frame 줌 (버그날까봐 처리해둔거)
                pre_gray = cur_gray

            pre_pts = cv2.goodFeaturesToTrack(pre_gray, maxCorners=1000, qualityLevel=0.01, minDistance=7) # 이전 Frame의 Feature 좌표 추출
            cur_pts, status, err = cv2.calcOpticalFlowPyrLK( # 이전 Frame과 현재 Frame을 비교하여 현재 Frame의 Feature 좌표 추출
                pre_gray, cur_gray, pre_pts, None,
                winSize=(21,21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )
            good_prev = pre_pts[status.ravel()==1] # 유효한 좌표만 추출
            good_next = cur_pts[status.ravel()==1]

            err_good = err[status.ravel()==1].ravel()
            mask = err_good < np.percentile(err_good, 90)  # 상위 10% 큰 오차 제거
            good_prev = good_prev[mask]
            good_next = good_next[mask]

            # --- 🔧 shape 보정 추가 ---
            pts_old = np.squeeze(good_prev)
            pts_new = np.squeeze(good_next)

            # 이전/현재 Frame 간의 배경 Feature 이동 거리 평균 계산
            for (x0, y0), (x1, y1) in zip(pts_old, pts_new):
                if x1 >= box_l and x1 <= (box_l+box_w) and y1 >= box_t and y1 <= (box_t+box_h): # box 안(얼굴) Data는 무시한다.
                    continue

                old_new_dx, old_new_dy = (x1 - x0), (y1 - y0) # 배경의 거리 계산
                average_dist = average_dist + np.sqrt(old_new_dx**2 + old_new_dy**2) # 이동 거리의 유클리디안 거리 계산
                average_count= average_count+1 # 평균 계산을 위해 Count

            if average_count > 0: 
                average_dist = average_dist / average_count # 평균 계산
            else:
                fix_image_center = True # 이동한게 전혀 없으면 box center로 고정한다

            if((average_dist * 5) < DEF_MAX_SHAKE_DISTANCE): # Image 떨림이 심할 경우 (모터 이동중, 모터 떨림)
                fix_image_center = False
            else: # Image 떨림이 없을 경우 (모터 고정하여 가만히 있는 경우)
                fix_image_center = True  

            # image가 전체 떨릴때
            if fix_image_center is True:
                # disp_kf_cx = int(cx_oe.filter(box_cx, now)) # 원유로로 Center로 은근슬쩍 가도록 만든다
                # disp_kf_cy = int(cy_oe.filter(box_cy, now)) 
                disp_kf_cx = box_cx # 센터 고정
                disp_kf_cy = box_cy # 센터 고정
                
            # image가 안떨릴 때
            else:

                # Frame Center와 Box Center의 거리를 구한다
                diff_box_cx_val = box_cx - comp_frame_cx
                diff_box_cy_val = box_cy - comp_frame_cy
                diff_box_dist_val = np.sqrt(diff_box_cx_val**2+diff_box_cy_val**2)
                
                # Frame Center와 Box Center가 가까이 있을 때
                if( diff_box_dist_val < DEF_MIN_FRAME_CENTER_DISTANCE ):
                    disp_kf_cx = comp_frame_cx # 영상이 움직이지 않도록 Frame Center을 준다
                    disp_kf_cy = comp_frame_cy
                else : # Frame Center와 Box Center가 멀리 있을때
                    disp_kf_cx = int(cx_oe.filter(use_cx, now)) # 원유로로 Center로 은근슬쩍 가도록 만든다
                    disp_kf_cy = int(cy_oe.filter(use_cy, now))
                if( diff_box_dist_val < DEF_MIN_FRAME_CENTER_DISTANCE * 3):
                    comp_frame_cx = disp_kf_cx # 그다음 Frame부터는 움직임을 최소화 하기 위해 Frame Center를 보정해준다
                    comp_frame_cy = disp_kf_cy

            pre_gray = cur_gray # 위의 작업이 끝났으면 현재 Frame을 이전 Frame으로 넘겨준다
            
            ##-----------------------------------------------------------------
                
            disp_ori_cx = box_cx
            disp_ori_cy = box_cy
                

            # 중앙 평행이동 + 크롭
            display_w = int(frame_w * (1-RATIO_TRANSLATE))
            display_h = int(frame_h * (1-RATIO_TRANSLATE))
            crop_t = int(disp_kf_cy-(display_h/2))
            crop_b = int(disp_kf_cy+(display_h/2))
            crop_l = int(disp_kf_cx-(display_w/2))
            crop_r = int(disp_kf_cx+(display_w/2))
            
            if crop_t < 0:
                crop_t = 0
                crop_b = crop_t + display_h
            elif crop_b >= frame_h-1:
                crop_b = frame_h-1
                crop_t = crop_b-display_h
            
            if crop_l < 0:
                crop_l = 0
                crop_r = crop_l + display_w
            elif crop_r >= frame_w-1:
                crop_r = frame_w-1
                crop_l = crop_r-display_w
            
            shifted = frame[int(crop_t):int(crop_b), int(crop_l):int(crop_r)]
            
            disp_addapt_size_kf_cx = disp_kf_cx - crop_l
            disp_addapt_size_kf_cy = disp_kf_cy - crop_t
            disp_addapt_size_ori_cx = disp_ori_cx - crop_l
            disp_addapt_size_ori_cy = disp_ori_cy - crop_t
            
            out_frame = shifted
            display = out_frame.copy()

            # 가이드 박스
            guide_w = box_w
            guide_h = box_h
            gx1 = int(disp_addapt_size_ori_cx - (guide_w/2))
            gx2 = int(gx1+guide_w)
            gy1 = int(disp_addapt_size_ori_cy - (guide_h/2))
            gy2 = int(gy1+guide_h)
            gcx = disp_addapt_size_ori_cx
            gcy = disp_addapt_size_ori_cy
            gx1=max(3,gx1)
            gy1=max(3,gy1)
            gx2=min(display.shape[1]-3,gx2)
            gy2=min(display.shape[0]-3,gy2)

            cv2.circle(display, (display_w//2, display_h//2), ICR_RADIUS, (255,0,0), 2)

            if face_found:
                cv2.rectangle(display, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (0,200,0), 2)
                cv2.circle(display, (int(gcx), int(gcy)), 3, (0, 0, 255), -1)

            # ⭐⭐⭐ 테스트 모드 중일 때 원 표시
            if test_phase == "stopping" and test_reference_point:
                # 화면 가로 기준 비율로 원 그리기
                TEST_CIRCLE_RADIUS = int(frame_w * CIRCLE_RADIUS_RATIO)
                cv2.circle(display, (display_w//2, display_h//2), TEST_CIRCLE_RADIUS, (255, 0, 0), 2)

            # 지표 계산
            if _prev_t is None:
                _prev_t = now
            dt = max(1e-3, now - _prev_t)
            speed_px = est_speed_px_per_s(gcx, gcy, _prev_cx, _prev_cy, dt)
            speed_cm = speed_px * CM_PER_PIXEL
            
            if _prev_cx is not None:
                dist = ((gcx - _prev_cx)**2 + (gcy - _prev_cy)**2) ** 0.5
                stab_buf.append((now, dist))
                while stab_buf and (now - stab_buf[0][0]) > STAB_WIN_SEC:
                    stab_buf.popleft()
                if stab_buf:
                    inside = sum(1 for (_, d) in stab_buf if d <= DT_THRESH_PX)
                    ratio = 100.0 * inside / len(stab_buf)
                    metric2_ratios.append(ratio)
            
            _prev_cx, _prev_cy, _prev_t = gcx, gcy, now
            metric1_speeds_px.append(speed_px)
            metric1_speeds_cm.append(speed_cm)

            # 지표3: ICR3
            if speed_px < STOP_SPEED_THR:
                if icr3_phase == "move":
                    icr3_phase = "stop and collect start"
                    icr3_center = (display_w//2, display_h//2)
                    icr3_t0 = now
                    icr3_inside = 0
                    icr3_total = 0
                    # debug_log(f"ICR3 수집 시작", "INFO")
                    if len(metric3_ratios)>0:
                        matric3_text = f"[지표3] ICR3={metric3_ratios[-1]:.1f}%"
                    else:
                        matric3_text = f"[지표3] Data 없음"
                elif icr3_phase == "stop and collect start":
                    r = ((gcx - icr3_center[0])**2 + (gcy - icr3_center[1])**2)**0.5
                    matric3_text = f"[지표3] 수집중... ({int(now-icr3_t0-STOP_HOLD_START)}s)"
                    if (now - icr3_t0) >= STOP_HOLD_START:
                        icr3_total += 1
                        if r <= ICR_RADIUS:
                            icr3_inside += 1
                        if (now - icr3_t0) >= STOP_HOLD_SEC+STOP_HOLD_START:
                            ratio = 100.0 * icr3_inside / max(1, icr3_total)
                            metric3_ratios.append(ratio)
                            # debug_log(f"ICR3 수집 완료: {ratio:.1f}%", "INFO")
                            icr3_phase = "idle"
                        cv2.circle(display, (display_w//2, display_h//2), ICR_RADIUS, (255,0,0), 2)
                else:
                    if len(metric3_ratios)>0:
                        matric3_text = f"[지표3] ICR3={metric3_ratios[-1]:.1f}%"
                    else:
                        matric3_text = f"[지표3] Data 없음"
            else:
                matric3_text = f"[지표3] 이동중"
                icr3_phase = "move"

            # ---- 오버레이 (쓰로틀) ----
            #if frame_idx % OVERLAY_EVERY == 0:
                #display = draw_text_kr(display, f"[FACE] offset=({gcx-display.shape[1]//2},{gcy-display.shape[0]//2})", (10, display_h-140), 25, 2)
                #if len(metric1_times)>0:
                   #display = draw_text_kr(display, f"[지표1] 재인식: {metric1_times[-1]:.3f}s", (10, display_h-110), 25, 2)
                #if len(metric2_ratios)>0:
                    #display = draw_text_kr(display, f"[지표2] 안정: {metric2_ratios[-1]:5.1f}%", (10, display_h-80), 25, 2)
                #display = draw_text_kr(display, matric3_text, (10, display_h-50), 25, 2)
            #else:
                # 쓰로틀되는 프레임에는 가벼운 cv2.putText만 (영문/숫자)
                #cv2.putText(display, f"FPS:{frame_per_sec} | Q:{q.qsize()}",
                            #(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

            # ⭐⭐⭐ 평가지표 3 테스트 진행 상태 표시
            if test_mode_active:
                elapsed_test = now - test_start_time
                if test_phase == "waiting":
                    test_text = f"[Test 3] 대기중... {elapsed_test:.1f}s"
                    cv2.putText(display, test_text, (display.shape[1]//2 - 150, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                elif test_phase == "moving":
                    move_elapsed = elapsed_test - 4
                    test_text = f"[Test 3] 움직임: {move_elapsed:.1f}s / 3.0s"
                    cv2.putText(display, test_text, (display.shape[1]//2 - 250, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                elif test_phase == "stopping":
                    stop_elapsed = now - test_stop_start_time
                    test_text = f"[Test 3] 정지: {stop_elapsed:.1f}s / 3.0s ({len(test_coordinates)})"
                    cv2.putText(display, test_text, (display.shape[1]//2 - 300, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

            # ⭐⭐⭐ 평가지표 1 테스트 진행 상태 표시
            if tracking_test_mode and test1_vars["movement_start_time"]:
                elapsed_move = now - test1_vars["movement_start_time"]
                test_text = f"[Test 1] 움직임: {elapsed_move:.1f}s / {test_duration}s (Check: {DETECTION_TIME}s)"
                cv2.putText(display, test_text, (display.shape[1]//2 - 350, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 디버깅 정보 화면 표시
            if DEBUG_MODE:
                info_text = f"FPS:{frame_per_sec} | Serial:{debug_counters['serial_sent']}/{debug_counters['serial_error']} | Queue:{q.qsize()}"
                cv2.putText(display, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

                # 추적 상태 표시
                if tracking_test_mode:
                    track_status = "TRACKING" if tracking_enabled else "WAITING"
                    status_color = (0, 255, 0) if tracking_enabled else (0, 165, 255)
                    cv2.putText(display, f"[Test 1] {track_status}", (display.shape[1] - 200, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            # ⭐ 대기 상태 표시
            if not tracking_test_mode and not test_mode_active:
                cv2.putText(display, "Press 'i' (Test 1) or 'o' (Test 3)",
                           (display.shape[1] - 380, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # 녹화중 표시
            if recording and msg_lt_display==False:
                msg_lt_text, msg_lt_until = "녹화 중!", now + 500.0
            
            # 메시지 표시
            if now < msg_lt_until and msg_lt_text:
                msg_lt_display = True
                display = draw_text_kr(display, msg_lt_text, (10, 60), 28, 2)
            else:
                msg_lt_display = False
            
            if now < msg_rt_until and msg_rt_text:
                msg_rt_display = True
                (tw, th), _ = cv2.getTextSize(msg_rt_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                display = draw_text_kr(display, msg_rt_text, (display.shape[1]-10-int(tw*1.2), 60), 28, 2)
            else:
                msg_rt_display = False

            # 연속 촬영
            if photo_shooting and next_shot_at is not None:
                t_rem = max(0.0, next_shot_at - now)
                sec = int(np.ceil(t_rem))
                
                if sec >= 1:
                    cd_text = str(sec)
                else:
                    cd_text = "cheese~!" if t_rem <= 0.4 else ""
                
                if cd_text:
                    display = draw_text_kr(display, cd_text, (10, 120), 42, 3)
                remain = max(0, photo_count - photo_taken)

                if now >= next_shot_at:
                    filename = get_new_image_filename()
                    cv2.imwrite(filename, frame)
                    photo_taken += 1
                    debug_log(f"사진 저장 #{photo_taken}/{photo_count}: {os.path.basename(filename)}", "INFO")

                    if photo_taken >= photo_count:
                        photo_shooting = False
                        next_shot_at = None
                        msg_lt_text, msg_lt_until = f"연속 사진 촬영 완료", now + 1.0
                        debug_log(f"연속 촬영 완료", "INFO")
                    else:
                        next_shot_at = now + photo_interval
                display = draw_text_kr(display, f"남은 장: {remain}", (display.shape[1]-220, 60), 28, 2)

            cv2.imshow("Face Tracking Robot - Method A (Debug Mode)", display)

            # 키 입력
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                debug_log("종료 키 입력됨", "INFO", force=True)
                break

            # ⭐⭐⭐ 'i' 키: 평가지표 1 테스트 시작 ⭐⭐⭐
            if key == ord('i'):
                if not tracking_test_mode and not test_mode_active:
                    try:
                        print("\n" + "=" * 70)
                        user_input = input("테스트 시간을 입력하세요 (0.1~2.0초, 기본값=1.5): ").strip()

                        if user_input == "":
                            duration = 1.5
                        else:
                            duration = float(user_input)
                            if duration <= 0 or duration > 2.0:
                                print("⚠️  입력값은 0.1~2.0초 사이여야 합니다. 기본값 1.5초로 설정합니다.")
                                duration = 1.5

                        print("=" * 70)
                        print(f"🧪 평가지표 1 - 추적 성능 테스트 시작")
                        print("=" * 70)
                        print(f"⏱️  움직임 시간: {duration}초")
                        print(f"⏱️  검출 체크: {DETECTION_TIME}초 후")
                        print("=" * 70)
                        print("📌 테스트 절차:")
                        print("  1. 카메라 앞에 얼굴을 위치시켜 주세요")
                        print("  2. 카운트다운이 시작되면 준비하세요")
                        print("  3. '움직임 시작' 신호 후 좌우로 움직이세요")
                        print(f"  4. {DETECTION_TIME}초 후 로봇팔이 따라왔는지 확인합니다")
                        print("=" * 70)
                        print()

                        test1_vars = reset_test_mode(duration)
                        debug_log(f"평가지표 1 테스트 시작 (움직임: {duration}초, 검출: {DETECTION_TIME}초)", "INFO", force=True)

                    except ValueError:
                        print("⚠️  입력값이 유효하지 않습니다. 기본값 1.5초로 설정합니다.\n")
                        duration = 1.5
                        test1_vars = reset_test_mode(duration)
                    except Exception as e:
                        print(f"⚠️  오류 발생: {e}. 기본값 1.5초로 설정합니다.\n")
                        duration = 1.5
                        test1_vars = reset_test_mode(duration)
                else:
                    print("\n⚠️  테스트가 이미 진행 중입니다. 완료될 때까지 기다려주세요.\n")

            # ⭐⭐⭐ 'o' 키: 평가지표 3 테스트 시작 ⭐⭐⭐
            if key == ord('o'):
                if not test_mode_active and not tracking_test_mode:
                    print("\n" + "=" * 70)
                    print("🧪 평가지표 3 - 추적 안정성 테스트 시작")
                    print("=" * 70)
                    print("📌 테스트 절차:")
                    print("  1. 카운트다운 후 좌우 또는 상하로 움직이세요 (3초)")
                    print("  2. '움직임 멈춤' 신호 후 정지하세요 (3초)")
                    print("  3. 추적 안정성을 측정합니다")
                    print(f"  4. 목표: 원 내부 비율 ≥ 80% (반지름: 화면 대각선의 3%)")
                    print("=" * 70)
                    print()

                    test_start_time = time.time()
                    test3_countdown_printed = {}
                    test_mode_active = True
                    test_phase = "waiting"
                    test_stop_start_time = 0
                    test_coordinates = []
                    test_reference_point = None
                    debug_log("평가지표 3 테스트 시작", "INFO", force=True)
                else:
                    print("\n⚠️  테스트가 이미 진행 중입니다. 완료될 때까지 기다려주세요.\n")

            if key == ord('s') and not recording and not photo_shooting:
                output_path = get_new_filename()
                debug_log(f"녹화 시작 시도: {os.path.basename(output_path)}", "INFO")
                record_w = out_frame.shape[1] if RECORD_USE_STAB else frame_w
                record_h = out_frame.shape[0] if RECORD_USE_STAB else frame_h
                out = cv2.VideoWriter(output_path, fourcc, frame_per_sec, (record_w, record_h))
                if not out.isOpened():
                    msg_lt_text, msg_lt_until = f"VideoWriter 열기 실패", now + 1.0
                    debug_log("VideoWriter 열기 실패", "ERROR")
                    out = None
                else:
                    recording = True
                    msg_lt_text, msg_lt_until = f"녹화 시작: {os.path.basename(output_path)}", now + 1.0
                    msg_lt_display = True
                    debug_log(f"녹화 시작: {record_w}x{record_h} @ {frame_per_sec}fps", "INFO")

            if key == ord('e') and recording:
                recording = False
                if out is not None:
                    out.release()
                    out = None
                debug_log("녹화 종료", "INFO")
                msg_lt_text, msg_lt_until = "녹화 종료!", now + 1.0
                msg_lt_display = True

            # 녹화 프레임 쓰기
            if recording and out is not None:
                clean = out_frame if RECORD_USE_STAB else frame
                out.write(clean)

            # 연속촬영 시작 (1~9)
            if (ord('1') <= key <= ord('9')) and not photo_shooting:
                photo_count = key - ord('0')
                photo_taken = 0
                photo_shooting = True
                next_shot_at = now + photo_interval
                msg_lt_text, msg_lt_until = f"{photo_count}장 연속 촬영 시작! ({photo_interval:.0f}초 간격)", now + 500
                debug_log(f"연속 촬영 시작: {photo_count}장, {photo_interval}초 간격", "INFO")
            
            pre_frame_time = now

    except KeyboardInterrupt:
        debug_log("KeyboardInterrupt 발생", "WARN", force=True)
    except Exception as e:
        debug_log(f"예외 발생: {e}", "ERROR", force=True)
        import traceback
        traceback.print_exc()
    finally:
        debug_log("리소스 정리 시작...", "INFO", force=True)
        try:
            if out is not None:
                out.release()
                debug_log("VideoWriter 해제 완료", "INFO")
        except Exception as e:
            debug_log(f"VideoWriter 해제 오류: {e}", "WARN")
        
        cap_thread.release()
        cv2.destroyAllWindows()
        q.put(None)

        # 지표 요약
        print("\n" + "=" * 70)
        print("📊 성능 지표 최종 요약")
        print("=" * 70)
        
        print(f"\n🔧 시스템 통계:")
        print(f"  총 프레임 처리: {debug_counters['frame_count']}")
        print(f"  얼굴 검출 성공: {debug_counters['face_detected']}회")
        print(f"  얼굴 손실: {debug_counters['face_lost']}회")
        print(f"  시리얼 전송: {serial_health['total_sent']}회")
        print(f"  시리얼 오류: {serial_health['total_errors']}회")
        if serial_health['total_sent'] > 0:
            error_rate = (serial_health['total_errors'] / serial_health['total_sent'] * 100)
            print(f"  시리얼 오류율: {error_rate:.2f}%")
        print(f"  모터 Freeze: {debug_counters['motor_frozen']}회")
        
        # ⭐ 시리얼 통신 진단
        if serial_health['connection_lost']:
            print(f"\n⚠️  시리얼 연결 문제 감지됨!")
            print(f"   - USB 연결 확인")
            print(f"   - 아두이노 상태 확인")
            print(f"   - Baud Rate 확인: {SERIAL_BAUD}")
        elif serial_health['total_sent'] == 0:
            print(f"\n⚠️  시리얼 데이터 전송 없음!")
            print(f"   - 얼굴이 검출되지 않았을 수 있음")
            print(f"   - 카메라 위치/조명 확인")
        elif serial_health['total_errors'] > 0:
            error_rate = (serial_health['total_errors'] / serial_health['total_sent'] * 100)
            if error_rate > 10:
                print(f"\n⚠️  시리얼 오류율 높음: {error_rate:.1f}%")
                print(f"   - USB 케이블 교체 권장")
                print(f"   - 아두이노 처리 속도 확인")
            else:
                print(f"\n✅ 시리얼 통신: 정상 (오류율 {error_rate:.1f}%)")
        else:
            print(f"\n✅ 시리얼 통신: 완벽 (오류 없음)")
        
        if len(metric1_times)>0:
            arr=np.array(metric1_times)
            print(f"\n📊 [지표1] 재인식 시간 (샘플: {len(arr)}개)")
            print(f"  평균: {arr.mean():.3f}s")
            print(f"  중앙값: {np.median(arr):.3f}s")
            print(f"  최소: {arr.min():.3f}s")
            print(f"  최대: {arr.max():.3f}s")
        else:
            print(f"\n📊 [지표1] 재인식 시간: 샘플 없음")

        if len(metric1_speeds_px)>0:
            ap=np.array(metric1_speeds_px)
            ac=np.array(metric1_speeds_cm)
            print(f"\n📊 [지표1-속도] 추적 속도 (샘플: {len(ap)}개)")
            print(f"  px/s - 평균: {ap.mean():.1f} | 중앙값: {np.median(ap):.1f} | 최대: {ap.max():.1f}")
            print(f"  cm/s - 평균: {ac.mean():.1f} | 중앙값: {np.median(ac):.1f} | 최대: {ac.max():.1f}")
        else:
            print(f"\n📊 [지표1-속도] 샘플 없음")

        if len(metric2_ratios)>0:
            arr=np.array(metric2_ratios)
            print(f"\n📊 [지표2] 추적 안정성 (샘플: {len(arr)}개)")
            print(f"  평균: {arr.mean():.1f}%")
            print(f"  중앙값: {np.median(arr):.1f}%")
            print(f"  최소: {arr.min():.1f}%")
            print(f"  최대: {arr.max():.1f}%")
        else:
            print(f"\n📊 [지표2] 추적 안정성: 샘플 없음")

        if len(metric3_ratios)>0:
            arr=np.array(metric3_ratios)
            print(f"\n📊 [지표3] ICR3 원내 비율 (샘플: {len(arr)}개)")
            print(f"  평균: {arr.mean():.1f}%")
            print(f"  중앙값: {np.median(arr):.1f}%")
            print(f"  최소: {arr.min():.1f}%")
            print(f"  최대: {arr.max():.1f}%")
        else:
            print(f"\n📊 [지표3] ICR3: 샘플 없음")
        
        print("=" * 70)
        print("✅ 프로그램 종료 완료")
        print("=" * 70)

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 프로그램 초기화")
    print("=" * 70)
    print(f"Python 버전: {sys.version.split()[0]}")
    print(f"OpenCV 버전: {cv2.__version__}")
    print(f"Numpy 버전: {np.__version__}")
    print(f"시리얼 포트: {SERIAL_PORT} @ {SERIAL_BAUD}bps")
    print("=" * 70)
    
    # ⭐ 시리얼 포트 존재 여부 확인
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    
    if ports:
        print("\n사용 가능한 포트:")
        port_found = False
        for p in ports:
            marker = "✅" if p.device == SERIAL_PORT else "  "
            print(f"  {marker} {p.device}: {p.description}")
            if p.device == SERIAL_PORT:
                port_found = True
        
        if not port_found:
            print(f"\n⚠️  경고: 설정된 포트 '{SERIAL_PORT}'를 찾을 수 없습니다!")
            print(f"   위 목록에서 올바른 포트를 선택하여 코드를 수정하세요.")
    else:
        print("\n❌ 사용 가능한 시리얼 포트가 없습니다!")
        print("   아두이노가 연결되어 있는지 확인하세요.")
    
    print("=" * 70)
    print()
    
    try:
        main()
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ 치명적 오류 발생!")
        print("=" * 70)
        print(f"오류 타입: {type(e).__name__}")
        print(f"오류 메시지: {e}")
        print("\n상세 스택:")
        import traceback
        traceback.print_exc()
        print("=" * 70)
        input("\n아무 키나 눌러 종료...")
    finally:
        print("\n프로그램 완전 종료")
