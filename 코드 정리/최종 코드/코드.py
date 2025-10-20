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
CAP_WIDTH, CAP_HEIGHT, CAP_FPS = 1920, 1080, 60
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

# 검출/추적
DETECT_EVERY = 1 # 1프레임 단위로 검출
LEAD_FACE_SEC = 0.12
CM_PER_PIXEL = 0.050

# 제어(로봇팔) - 방법 A
DESIRED_FACE_AREA = 35000
DEADZONE_XY = 20
DEADZONE_AREA = 12000
move_ready = threading.Event()
move_ready.set()
motor_freeze_time = {"x": 0, "y": 0, "z": 0}
FREEZE_DURATION_S = 0.6

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
        if DEBUG_DETAIL:
            debug_log(f"Freeze 활성: {axis}축", "DETAIL")
    return frozen

def update_freeze_timer(ddx, ddy, ddz, now):
    if ddx == 0:
        motor_freeze_time["x"] = now
        debug_log(f"Freeze 타이머 시작: X축", "DETAIL")
    if ddy == 0:
        motor_freeze_time["y"] = now
        debug_log(f"Freeze 타이머 시작: Y축", "DETAIL")
    if ddz == 0:
        motor_freeze_time["z"] = now
        debug_log(f"Freeze 타이머 시작: Z축", "DETAIL")

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
    
    # 거리별 스텝 크기 및 delay 조정
    if distance > 150:
        step = 5
        delay = 70
        dist_category = "아주 멀리"
    elif distance > 80:
        step = 3
        delay = 55
        dist_category = "중간"
    else:
        step = 1
        delay = 45
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
        "motor_3": ddy,
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
    def __init__(self, cam_index=0, backend=cv2.CAP_DSHOW):
        debug_log(f"카메라 초기화 시작: index={cam_index}", "INFO", force=True)
        self.cap = cv2.VideoCapture(cam_index, backend)

        # 포맷을 먼저 못박아 두는 게 협상 지연을 줄이는 데 도움됨
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS,          CAP_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        if not self.cap.isOpened():
            debug_log("카메라 열기 실패!", "ERROR", force=True)
            raise RuntimeError("카메라 열기 실패")

        # 워밍업: 캡이 실제 스트리밍을 시작하도록 첫 0.5~1초간 프레임 버림
        t0 = time.time()
        while time.time() - t0 < 0.7:
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
        # 불필요한 2회 grab 제거 → read 한 번이 더 안정적인 경우가 많음
        while self.running:
            ret, f = self.cap.read()
            if ret:
                with self.lock:
                    self.latest = f
                    self.frame_count += 1
            else:
                debug_log("프레임 읽기 실패", "WARN")
    
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

        while True:
            ok, frame = cap_thread.read()
            if not ok:
                debug_log("프레임 읽기 실패", "WARN")
                continue
            
            now = time.time()
            debug_counters["frame_count"] += 1
            
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
                ICR_RADIUS = int(((((frame_w/2)**2) + ((frame_h/2)**2))**0.5) * ICR_RATIO)
                debug_log(f"ICR 반경 설정: {ICR_RADIUS}px", "INFO")

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

            # 얼굴 상태 변화 로깅
            if face_found and reacquire_t0 is not None:
                reacq = now - reacquire_t0
                metric1_times.append(reacq)
                debug_log(f"얼굴 재인식 완료: {reacq:.3f}초", "INFO")
                reacquire_t0 = None
            elif not face_found and reacquire_t0 is None:
                reacquire_t0 = now
                debug_counters["face_lost"] += 1
                debug_log(f"얼굴 손실 (#{debug_counters['face_lost']})", "WARN")

            if face_found:
                face_boxes.sort(key=lambda b: b[2]*b[3], reverse=True)
                box_l, box_t, box_w, box_h = face_boxes[0]
                box_cx, box_cy = box_l + box_w//2, box_t + box_h//2
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
            ## ⚡ 로봇팔 제어
            ##-----------------------------------------------------------------
            if face_found and move_ready.is_set():
                debug_log(f"로봇팔 제어 시작", "DETAIL")
                
                angles = compute_motor_angles_safe(box_cx, box_cy, area, frame.shape)
                
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
            disp_kf_cx = int(cx_oe.filter(use_cx, now)) #칼만
            disp_kf_cy = int(cy_oe.filter(use_cy, now)) #칼만
 			#disp_kf_cx = frame_cx# original
            #disp_kf_cy = frame_cy# original
            #disp_kf_cx = box_cx#센터 고정
            #disp_kf_cy = box_cy#센터 고정
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
            
            if face_found:
                cv2.rectangle(display, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (0,200,0), 2)
                cv2.circle(display, (int(gcx), int(gcy)), 3, (0, 0, 255), -1)

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
                    debug_log(f"ICR3 수집 시작", "INFO")
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
                            debug_log(f"ICR3 수집 완료: {ratio:.1f}%", "INFO")
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

            # 오버레이
            display = draw_text_kr(display, f"[FACE] offset=({gcx-display.shape[1]//2},{gcy-display.shape[0]//2})", (10, display_h-140), 25, 2)
            if len(metric1_times)>0:
                display = draw_text_kr(display, f"[지표1] 재인식: {metric1_times[-1]:.3f}s", (10, display_h-110), 25, 2)
            if len(metric2_ratios)>0:
                display = draw_text_kr(display, f"[지표2] 안정: {metric2_ratios[-1]:5.1f}%", (10, display_h-80), 25, 2)
            display = draw_text_kr(display, matric3_text, (10, display_h-50), 25, 2)
            
            # 디버깅 정보 화면 표시
            if DEBUG_MODE:
                info_text = f"FPS:{frame_per_sec} | Serial:{debug_counters['serial_sent']}/{debug_counters['serial_error']} | Queue:{q.qsize()}"
                cv2.putText(display, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            
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