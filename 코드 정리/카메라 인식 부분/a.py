import cv2
import numpy as np
import threading
import queue
import serial
import os
import re
import time
from collections import deque

# ============================================================
# 통합본 + 모터 상세 로그(raw/clip/slew & 실제 TX)
# - 제어 기준: (칼만 보정된 얼굴중심 use_cx,use_cy) vs 프레임 중앙
# - raw: PD(+칼만 속도 선행 & 게인스케일/거리축동결) 출력
# - clip: 각도 제한(-90~90°) 적용 후
# - slew: 프레임간 변화 제한(SLEW) 적용 후 = 큐에 넣는 최종 값
# - SERIAL TX: 실제 아두이노로 전송된 값
# ============================================================

# ================== 기본 설정 ==================
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

# --- 미러/저장 정책 ---
PREVIEW_MIRROR = True
RECORD_MIRROR  = True
PHOTO_MIRROR   = True

RECORD_USE_EIS = True
PHOTO_USE_EIS  = False

OVERLAY_PREVIEW_ONLY = True

# --- 칼만/광류 파라미터 ---
FLOW_BETA = 0.35
FLOW_FEATURE_REFRESH = 15
FLOW_MIN_VALID_BASE = 80
DOWNSAMPLE_SCALE = 0.5

# --- DNN 호출 빈도 ---
DETECT_EVERY = 2

# --- PD/슬루/데드존 ---
DEAD_X = 0.02
DEAD_Y = 0.02
D_ALPHA = 0.25
D_CLAMP = 5.0
SLEW = 8.0  # deg/frame

# --- 픽셀↔cm (필수: 본인 환경 캘리브) ---
CM_PER_PIXEL = 0.050

# --- 모터 로그 옵션 ---
VERBOSE_MOTOR_LOG = True
MOTOR_LOG_INTERVAL = 0.20
_last_motor_log_ts = 0.0   # 전역에 한 번만!

# ================== 파일 이름 자동 증가 ==================
def get_new_filename(base_name="output", ext="avi"):
    existing_files = os.listdir(desktop_path)
    pattern = re.compile(rf"{re.escape(base_name)}_(\d+)\.{re.escape(ext)}")
    nums = [int(m.group(1)) for f in existing_files if (m := pattern.match(f))]
    n = max(nums, default=0) + 1
    return os.path.join(desktop_path, f"{base_name}_{n}.{ext}")

def get_new_picture_filename(base_name="picture", ext="jpg"):
    existing_files = os.listdir(desktop_path)
    pattern = re.compile(rf"{re.escape(base_name)}_(\d+)\.{re.escape(ext)}")
    nums = [int(m.group(1)) for f in existing_files if (m := pattern.match(f))]
    n = max(nums, default=0) + 1
    return os.path.join(desktop_path, f"{base_name}_{n}.{ext}")

# ================== 텍스트 드로잉 ==================
def draw_text(img, text, org, font_scale=0.8, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness, cv2.LINE_AA)

# ================== 시리얼 송신 스레드 ==================
def serial_worker(q, port='COM5', baud=115200):
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        print("시리얼 연결 완료")
    except Exception as e:
        print(f"시리얼 연결 실패: {e}")
        return

    try:
        while True:
            motor_cmds = q.get()
            if motor_cmds is None:
                break
            try:
                vals = [motor_cmds.get(f"motor_{i}", 0) for i in range(1, 7)]
                ser.write((','.join(map(str, vals)) + '\n').encode('utf-8'))
                if VERBOSE_MOTOR_LOG:
                    print(f"[SERIAL TX] {vals}")
            except Exception as e:
                print(f"[SerialWriteError] {e}")
    finally:
        ser.close()
        print("시리얼 종료")

# ================== 얼굴 검출 (OpenCV DNN) ==================
prototxt_path = r"C:\face_models\deploy.prototxt"
model_path   = r"C:\face_models\res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

def detect_faces_dnn(frame, conf_thresh=0.5):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    det = net.forward()
    boxes = []
    for i in range(det.shape[2]):
        conf = float(det[0, 0, i, 2])
        if conf > conf_thresh:
            x1, y1, x2, y2 = (det[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
            x1, y1, x2, y2 = max(0,x1), max(0,y1), min(w-1,x2), min(h-1,y2)
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes

# ================== 제어 (PD + 칼만 속도) ==================
prev_error_x = prev_error_y = prev_error_area = 0.0
last_control_time = time.time()
d_ex_ema = d_ey_ema = d_ea_ema = 0.0

def compute_motor_angles(center_x, center_y, area, frame_shape, desired_area=42000, kf=None, kalman_inited=False):
    global prev_error_x, prev_error_y, prev_error_area, last_control_time
    global d_ex_ema, d_ey_ema, d_ea_ema

    h, w = frame_shape[:2]

    # (핵심) 얼굴 중심 vs 화면 중앙 오차(정규화)
    error_x = (center_x - w/2) / (w/2)   # -1..1
    error_y = (center_y - h/2) / (h/2)   # -1..1
    error_area = (desired_area - area)   # 거리축

    # 데드존
    if abs(error_x) < DEAD_X: error_x = 0.0
    if abs(error_y) < DEAD_Y: error_y = 0.0

    # dt
    t = time.time()
    dt = max(1e-3, t - last_control_time)
    last_control_time = t

    # 미분
    d_error_x = (error_x - prev_error_x) / dt
    d_error_y = (error_y - prev_error_y) / dt
    d_error_area = (error_area - prev_error_area) / dt

    # 미분 EMA + 클램프
    d_ex_ema = (1 - D_ALPHA) * d_ex_ema + D_ALPHA * d_error_x
    d_ey_ema = (1 - D_ALPHA) * d_ey_ema + D_ALPHA * d_error_y
    d_ea_ema = (1 - D_ALPHA) * d_ea_ema + D_ALPHA * d_error_area
    d_ex_ema = float(np.clip(d_ex_ema, -D_CLAMP, D_CLAMP))
    d_ey_ema = float(np.clip(d_ey_ema, -D_CLAMP, D_CLAMP))

    # 칼만 속도 선행
    if kalman_inited and kf is not None:
        vx = kf.statePost[2,0] / (w/2)
        vy = kf.statePost[3,0] / (h/2)
        predict_ahead = 0.15
        error_x += vx * predict_ahead
        error_y += vy * predict_ahead

    # PD 게인
    Kp_xy = 90 * 0.35
    Kd_xy = 90 * 0.25
    Kp_area = 0.0020
    Kd_area = 0.0012

    # (raw) PD 출력
    out_x = Kp_xy * error_x + Kd_xy * d_ex_ema
    out_y = Kp_xy * error_y + Kd_xy * d_ey_ema
    out_a = Kp_area * error_area + Kd_area * d_ea_ema

    prev_error_x, prev_error_y, prev_error_area = error_x, error_y, error_area

    return {
        "motor_1": out_y,
        "motor_2": out_y,
        "motor_3": out_y * 0.5,
        "motor_4": out_a,
        "motor_5": out_y * 0.2,
        "motor_6": out_x * 0.5
    }

def clip_motor_angles(cmds, limits=(-90, 90)):
    return {k: int(np.clip(v, limits[0], limits[1])) for k, v in cmds.items()}

last_cmds = {f"motor_{i}": 0.0 for i in range(1,7)}
def apply_slew(cmds, max_delta=SLEW):
    global last_cmds
    out = {}
    for k, v in cmds.items():
        dv = float(np.clip(v - last_cmds[k], -max_delta, max_delta))
        out[k] = last_cmds[k] + dv
        last_cmds[k] = out[k]
    return out

# ================== Kalman 2D ==================
def init_kalman():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kf.processNoiseCov = np.diag([1e-2, 1e-2, 1e-1, 1e-1]).astype(np.float32)
    kf.measurementNoiseCov = np.diag([2.0, 2.0]).astype(np.float32)
    kf.errorCovPost = np.diag([10, 10, 10, 10]).astype(np.float32)
    return kf

def kalman_predict(kf, dt):
    kf.transitionMatrix = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], np.float32)
    pred = kf.predict()
    return pred[0,0], pred[1,0]

def kalman_correct(kf, x, y):
    meas = np.array([[np.float32(x)], [np.float32(y)]], np.float32)
    kf.correct(meas)

# ================== Affine EIS ==================
class AffineEIS:
    def __init__(self, beta=0.35, feature_refresh=15, scale=0.5, min_valid=80):
        self.prev_gray_small = None
        self.prev_pts = None
        self.beta = beta
        self.feature_refresh = feature_refresh
        self.frame_count = 0
        self.min_valid = int(min_valid)
        self.scale = scale
        self.Mc_ema = np.array([[1,0,0],[0,1,0]], np.float32)

    def _detect_features(self, gray_small, mask=None):
        return cv2.goodFeaturesToTrack(gray_small, maxCorners=800, qualityLevel=0.01,
                                       minDistance=12, blockSize=3, mask=mask)

    def update(self, gray_full, face_box=None):
        self.frame_count += 1
        gray_small = cv2.resize(gray_full, (0,0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        h, w = gray_small.shape[:2]

        mask = None
        if face_box is not None:
            x, y, bw, bh = face_box
            x = int(x*self.scale); y = int(y*self.scale)
            bw = int(bw*self.scale); bh = int(bh*self.scale)
            mask = np.full((h, w), 255, np.uint8)
            x0, y0 = max(0, x-8), max(0, y-8)
            x1, y1 = min(w-1, x+bw+8), min(h-1, y+bh+8)
            mask[y0:y1, x0:x1] = 0

        if self.prev_gray_small is None:
            self.prev_gray_small = gray_small
            self.prev_pts = self._detect_features(gray_small, mask)
            return self.Mc_ema.copy()

        if self.prev_pts is None or len(self.prev_pts) < 20:
            self.prev_pts = self._detect_features(self.prev_gray_small, mask)

        if self.prev_pts is None:
            self.prev_gray_small = gray_small
            return self.Mc_ema.copy()

        nxt_pts, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray_small, gray_small, self.prev_pts, None,
            winSize=(31,31), maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 40, 0.01)
        )
        if nxt_pts is None or st is None:
            self.prev_gray_small = gray_small
            self.prev_pts = self._detect_features(gray_small, mask)
            return self.Mc_ema.copy()

        good_new = nxt_pts[st==1]
        good_old = self.prev_pts[st==1]
        if len(good_new) < self.min_valid:
            self.prev_gray_small = gray_small
            self.prev_pts = self._detect_features(gray_small, mask)
            return self.Mc_ema.copy()

        M, inliers = cv2.estimateAffinePartial2D(
            good_old, good_new, method=cv2.RANSAC,
            ransacReprojThreshold=3.0, maxIters=3000, confidence=0.99
        )
        if M is None:
            M = np.array([[1,0,0],[0,1,0]], np.float32)

        Mc_new = cv2.invertAffineTransform(M.astype(np.float32))
        self.Mc_ema = (1.0 - self.beta) * self.Mc_ema + self.beta * Mc_new

        self.prev_gray_small = gray_small
        if (self.frame_count % self.feature_refresh == 0) or (inliers is not None and inliers.sum() < max(self.min_valid, 0.4*len(good_new))):
            self.prev_pts = self._detect_features(gray_small, mask)
        else:
            self.prev_pts = good_new.reshape(-1,1,2)

        Mc = self.Mc_ema.copy()
        Mc[:,2] *= (1.0 / self.scale)
        return Mc

# ================== FSM ==================
hist = deque(maxlen=12)
mode = "TRACKING"
lost_since = None
scan_phase = 0.0

history_len = 10
predict_horizon = 0.4
damping = 0.85
lost_predict_max = 0.5
lost_hold_max = 0.5
scan_period = 1.2
scan_amp_px = 80

def push_history(x, y, t):
    hist.append((float(x), float(y), float(t)))

def est_velocity():
    if len(hist) < 2:
        return np.array([0.0, 0.0])
    x0, y0, t0 = hist[max(0, len(hist)-history_len)]
    x1, y1, t1 = hist[-1]
    dt = max(1e-3, t1 - t0)
    return np.array([(x1 - x0)/dt, (y1 - y0)/dt])

def update_tracker(face_found, cx, cy, now, frame_shape, fps_hint=30.0):
    global mode, lost_since, scan_phase
    h, w = frame_shape[:2]

    if face_found:
        push_history(cx, cy, now)
        mode = "TRACKING"
        lost_since = None
        return (cx, cy), mode

    if lost_since is None:
        lost_since = now
    lost_dt = now - lost_since

    if lost_dt <= lost_predict_max and len(hist) >= 2:
        v = est_velocity()
        pred_dt = min(predict_horizon, lost_dt)
        last_x, last_y, _ = hist[-1]
        pred = np.array([last_x, last_y]) + v * pred_dt
        pred = damping * pred + (1 - damping) * np.array([w//2, h//2])
        pred = np.clip(pred, [0,0], [w-1, h-1])
        mode = "PREDICT"
        return (int(pred[0]), int(pred[1])), mode

    if lost_dt <= (lost_predict_max + lost_hold_max):
        last_x, last_y, _ = hist[-1] if len(hist) else (w//2, h//2, now)
        mode = "HOLD"
        return (int(last_x), int(last_y)), mode

    mode = "SCAN"
    scan_phase = (scan_phase + 2*np.pi*(1.0/scan_period)*(1.0/fps_hint)) % (2*np.pi)
    amp_y = max(6, int(0.015 * min(w, h)))
    sxp = (w//2) + int(scan_amp_px * np.sin(scan_phase))
    syp = (h//2) + int(amp_y * np.sin(scan_phase * 0.5))
    return (sxp, syp), mode

# ================== 유틸: 아핀 합성 ==================
def compose_affine(Ma, Mb):
    A = np.vstack([Ma, [0,0,1]])
    B = np.vstack([Mb, [0,0,1]])
    C = A @ B
    return C[:2, :]

# ================== 스레드/카메라/상태 ==================
q = queue.Queue()
serial_thread = threading.Thread(target=serial_worker, args=(q,), daemon=True)
serial_thread.start()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
if not cap.isOpened():
    print("카메라 열기 실패")
    q.put(None)
    raise SystemExit

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
recording = False
out = None

photo_shooting = False
photo_count = 0
photo_taken = 0
photo_interval = 3.0
next_shot_at = None

LOG_INTERVAL = 0.3
last_face_log_ts = 0.0

# === 정량지표 저장 ===
metric1_times = []
metric2_ratios = []
metric3_ratios = []

_face_prev = False
_face_lost_t = None

metric2_active = False
metric2_t0 = 0.0
metric2_frames_total = 0
metric2_frames_le10 = 0
_metric2_prev_center = None

metric3_collecting = False
metric3_t0 = 0.0
metric3_centers = []

# === 칼만/광류 상태 ===
kf = init_kalman()
kalman_inited = False
last_kf_ts = time.time()
flow_eis = None

# === FPS 추정 ===
fps_hint = 30.0
fps_ema = None
last_t = time.time()

# === 한 번이라도 인식했는지 ===
ever_locked = False

print("키: s/e=녹화, 1~9=연속사진, m/u/p=미러/EIS 토글, g=지표2(3초) 시작, q=종료")

try:
    last_center_used = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 캡처 실패")
            break

        now = time.time()
        h, w = frame.shape[:2]

        # FPS 추정
        inst_fps = 1.0 / max(1e-3, now - last_t); last_t = now
        fps_ema = inst_fps if fps_ema is None else 0.9*fps_ema + 0.1*inst_fps
        fps_hint = float(np.clip(fps_ema, 10, 120))

        # 해상도 비례 파라미터
        diag = np.hypot(w, h)
        crop_margin = int(max(60, 0.06 * min(w, h)))
        moving_thresh_px = 0.008 * diag
        flow_min_valid = int(FLOW_MIN_VALID_BASE * (w*h) / (640*480))

        # EIS 초기화
        if flow_eis is None:
            flow_eis = AffineEIS(beta=FLOW_BETA, feature_refresh=FLOW_FEATURE_REFRESH,
                                 scale=DOWNSAMPLE_SCALE, min_valid=flow_min_valid)

        # DNN 호출
        frame_idx += 1
        do_detect = (frame_idx % DETECT_EVERY == 0)

        dt = max(1e-3, now - last_kf_ts); last_kf_ts = now

        # 얼굴 검출
        face_boxes = detect_faces_dnn(frame) if do_detect else []
        face_found = len(face_boxes) > 0

        if face_found:
            face_boxes.sort(key=lambda b: b[2]*b[3], reverse=True)
            x, y, w0, h0 = face_boxes[0]
            center_x = x + w0 // 2
            center_y = y + h0 // 2
            area = w0 * h0

            if not kalman_inited:
                kf.statePost = np.array([[center_x],[center_y],[0],[0]], np.float32)
                kalman_inited = True

            kx, ky = kalman_predict(kf, dt)
            kalman_correct(kf, center_x, center_y)
            kx, ky = kf.statePost[0,0], kf.statePost[1,0]

            ever_locked = True

            if now - last_face_log_ts >= LOG_INTERVAL:
                vx_pix = float(kf.statePost[2,0]); vy_pix = float(kf.statePost[3,0])
                speed_cm_s = (vx_pix**2 + vy_pix**2) ** 0.5 * CM_PER_PIXEL
                print(f"[FACE] center=({center_x},{center_y}) kalman=({int(kx)},{int(ky)}) area={area}  speed≈{speed_cm_s:.1f}cm/s")
                last_face_log_ts = now
        else:
            x = y = w0 = h0 = 0
            center_x = center_y = None
            area = 42000
            if kalman_inited:
                kx, ky = kalman_predict(kf, dt)
            else:
                kx, ky = (w//2, h//2)

        # FSM
        (ux_fsm, uy_fsm), track_mode = update_tracker(face_found, center_x, center_y, now, frame.shape, fps_hint=fps_hint)

        # 사용할 중심(제어 좌표 = 원본기준)
        if face_found:
            use_cx, use_cy = int(kx), int(ky)
            face_box_for_flow = (x, y, w0, h0)
        else:
            if track_mode == "SCAN":
                use_cx, use_cy = ux_fsm, uy_fsm
            else:
                use_cx, use_cy = int(kx), int(ky)
            face_box_for_flow = None

        # 광류 보정
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        Mc = flow_eis.update(gray, face_box=face_box_for_flow)

        # 얼굴 중앙 평행이동
        target = np.array([w//2, h//2], np.float32)
        face_shift = target - np.array([use_cx, use_cy], np.float32)
        face_shift[0] = float(np.clip(face_shift[0], -(crop_margin-5), (crop_margin-5)))
        face_shift[1] = float(np.clip(face_shift[1], -(crop_margin-5), (crop_margin-5)))
        T_face = np.array([[1,0,face_shift[0]],[0,1,face_shift[1]]], np.float32)

        # 합성/워핑/크롭
        M_total = compose_affine(T_face, Mc)

        is_moving = False
        if last_center_used is not None:
            distm = np.hypot(use_cx - last_center_used[0], use_cy - last_center_used[1])
            is_moving = distm > moving_thresh_px
        last_center_used = (use_cx, use_cy)

        stabilized = cv2.warpAffine(frame, M_total, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        x1 = crop_margin; y1 = crop_margin
        x2 = w - crop_margin; y2 = h - crop_margin
        sx_scale = w / float(w - 2*crop_margin)
        sy_scale = h / float(h - 2*crop_margin)
        stab_base = cv2.resize(stabilized[y1:y2, x1:x2], (w, h), interpolation=cv2.INTER_LINEAR)

        # 저장 프레임(오버레이 없음)
        if recording and out is not None:
            clean = stab_base if RECORD_USE_EIS else frame
            if RECORD_MIRROR: clean = cv2.flip(clean, 1)
            out.write(clean)

        # 연속 사진
        if photo_shooting and next_shot_at is not None and now >= next_shot_at:
            filename = get_new_picture_filename()
            cleanp = (stab_base if PHOTO_USE_EIS else frame)
            if PHOTO_MIRROR: cleanp = cv2.flip(cleanp, 1)
            cv2.imwrite(filename, cleanp)
            photo_taken += 1
            print(f"{photo_taken}/{photo_count} 저장됨: {os.path.basename(filename)}")
            if photo_taken >= photo_count:
                photo_shooting = False; next_shot_at = None
                print("연속 사진 촬영 완료!")
            else:
                next_shot_at = now + photo_interval

        # 미리보기(오버레이 전용)
        display = stab_base.copy()
        if PREVIEW_MIRROR: display = cv2.flip(display, 1)

        def map_point(px, py):
            v = np.array([[px, py]], np.float32)
            v2 = cv2.transform(v.reshape(-1,1,2), M_total).reshape(-1,2)[0]
            vx = (v2[0] - crop_margin) * sx_scale
            vy = (v2[1] - crop_margin) * sy_scale
            if PREVIEW_MIRROR: vx = w - 1 - vx
            return int(np.clip(vx, 0, w-1)), int(np.clip(vy, 0, h-1))

        def map_rect(rx, ry, rw, rh):
            x0, y0 = map_point(rx, ry); x1, y1 = map_point(rx+rw, ry+rh)
            xa, xb = sorted([x0, x1]); ya, yb = sorted([y0, y1])
            return xa, ya, max(1, xb - xa), max(1, yb - ya)

        if OVERLAY_PREVIEW_ONLY:
            draw_text(display, f"MODE: {track_mode}{'  [ON]' if ever_locked else ''}", (10, 30), 0.7, 2)
            cx_disp, cy_disp = map_point(use_cx, use_cy)
            cv2.circle(display, (cx_disp, cy_disp), 3, (0, 200, 0), -1)
            if face_found:
                rx, ry, rwv, rhv = map_rect(x, y, w0, h0)
                cv2.rectangle(display, (rx, ry), (rx+rwv, ry+rhv), (0,200,0), 2)
                draw_text(display, f"Center(raw): ({center_x}, {center_y})", (10, 60), 0.7, 2)
            if recording:
                draw_text(display, "REC ●", (10, h-15), 0.8, 2)

        cv2.imshow("Face Tracker (Kalman + Affine EIS + PD Control; mirror; clean saves)", display)

        # ---------- 제어 ----------
        if not ever_locked:
            q.put({f"motor_{i}": 0 for i in range(1,7)})
        else:
            # 손실 상태별 게인/거리축 동결
            if not face_found:
                if lost_since is None: lost_since = now
                lost_dt = now - lost_since
            else:
                lost_since = None; lost_dt = 0.0

            gain_scale = 1.0; freeze_area = False
            if not face_found:
                if   lost_dt <= 0.5: gain_scale = 0.6; freeze_area = True   # PREDICT
                elif lost_dt <= 1.0: gain_scale = 0.4; freeze_area = True   # HOLD
                else:                gain_scale = 0.25; freeze_area = True   # SCAN

            # (1) raw
            raw_cmds = compute_motor_angles(use_cx, use_cy, area, frame.shape, kf=kf, kalman_inited=kalman_inited)
            if freeze_area: raw_cmds["motor_4"] = 0
            for k in ["motor_1","motor_2","motor_3","motor_5","motor_6"]:
                raw_cmds[k] *= gain_scale

            # (2) clip
            clipped_cmds = clip_motor_angles(raw_cmds, (-90, 90))

            # (3) slew (최종)
            final_cmds = apply_slew(clipped_cmds, max_delta=SLEW)

            # 상세 로그(★ global 빼고 단순 대입)
            if VERBOSE_MOTOR_LOG:
                if now - _last_motor_log_ts >= MOTOR_LOG_INTERVAL:
                    if kalman_inited:
                        vx_pix = float(kf.statePost[2,0]); vy_pix = float(kf.statePost[3,0])
                        speed_cm_s = (vx_pix**2 + vy_pix**2) ** 0.5 * CM_PER_PIXEL
                    else:
                        speed_cm_s = 0.0
                    err_x = (use_cx - w/2) / (w/2)
                    err_y = (use_cy - h/2) / (h/2)
                    def fmt(d): return ", ".join([f"{k.split('_')[1]}:{v:6.1f}" for k,v in d.items()])
                    print(f"[CTRL] mode={track_mode:<8} face={'Y' if face_found else 'N'} "
                          f"cx,cy=({use_cx:4d},{use_cy:4d}) err=({err_x:+.3f},{err_y:+.3f}) "
                          f"spd≈{speed_cm_s:5.1f}cm/s")
                    print(f"       raw   [{fmt(raw_cmds)}]")
                    print(f"       clip  [{fmt(clipped_cmds)}]")
                    print(f"       slew  [{fmt(final_cmds)}]")
                    _last_motor_log_ts = now  # 그냥 갱신

            # 큐 전송
            q.put(final_cmds)

        # ---------- 지표들 ----------
        # (1) 재인식시간
        if face_found and (not _face_prev):
            if _face_lost_t is not None:
                t_reacq = now - _face_lost_t
                metric1_times.append(t_reacq)
                print(f"[지표1] 재인식 시간 = {t_reacq:.3f}s (목표 ≤ 2s)")
            _face_lost_t = None
        if (not face_found) and _face_prev:
            _face_lost_t = now
        _face_prev = face_found

        # (2) 이동중 안정성(수동 g키 3초)
        if metric2_active:
            if _metric2_prev_center is not None:
                dx = use_cx - _metric2_prev_center[0]
                dy = use_cy - _metric2_prev_center[1]
                dt_pix = float(np.hypot(dx, dy))
                metric2_frames_total += 1
                if dt_pix <= 10.0: metric2_frames_le10 += 1
            _metric2_prev_center = (use_cx, use_cy)
            if now - metric2_t0 >= 3.0:
                ratio = (metric2_frames_le10 / metric2_frames_total) * 100.0 if metric2_frames_total>0 else 0.0
                metric2_ratios.append(ratio)
                print(f"[지표2] |Δ|≤10px 비율 = {ratio:.1f}% (목표 ≥ 85%)")
                metric2_active = False
                metric2_frames_total = 0; metric2_frames_le10 = 0; _metric2_prev_center = None
        else:
            _metric2_prev_center = (use_cx, use_cy)

        # (3) 정지 후 ICR3 (자동 3초)
        if not is_moving:
            if not metric3_collecting:
                metric3_collecting = True; metric3_t0 = now; metric3_centers = []
            metric3_centers.append((use_cx, use_cy))
            if now - metric3_t0 >= 3.0:
                if len(metric3_centers) >= 2:
                    cx0, cy0 = metric3_centers[0]
                    R = 0.015 * w  # 반지름(가로의 1.5%) → 지름 3%
                    inside = [(np.hypot(cx - cx0, cy - cy0) <= R) for (cx, cy) in metric3_centers]
                    icr3 = (np.sum(inside) / len(metric3_centers)) * 100.0
                    metric3_ratios.append(icr3)
                    print(f"[지표3] ICR3 = {icr3:.1f}% (목표 ≥ 80%)  (R={R:.1f}px)")
                metric3_collecting = False; metric3_centers = []
        else:
            metric3_collecting = False; metric3_centers = []

        # 키 입력
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s') and not recording:
            output_path = get_new_filename()
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))
            if not out.isOpened():
                print("VideoWriter 열기 실패"); out = None
            else:
                recording = True; print(f"녹화 시작! 저장 파일명: {os.path.basename(output_path)}")
        if key == ord('e') and recording:
            recording = False
            if out is not None: out.release(); out = None
            print("녹화 종료! 영상이 저장되었습니다.")
        if (ord('1') <= key <= ord('9')) and not photo_shooting:
            photo_count = key - ord('0'); photo_taken = 0
            photo_shooting = True; next_shot_at = now + photo_interval
            print(f"{photo_count}장의 사진 연속 촬영 시작! (간격 {photo_interval:.0f}초)")
        if key == ord('m'):
            PREVIEW_MIRROR = not PREVIEW_MIRROR; print(f"PREVIEW_MIRROR = {PREVIEW_MIRROR}")
        if key == ord('u'):
            RECORD_USE_EIS = not RECORD_USE_EIS; print(f"RECORD_USE_EIS = {RECORD_USE_EIS}")
        if key == ord('p'):
            PHOTO_USE_EIS = not PHOTO_USE_EIS; print(f"PHOTO_USE_EIS = {PHOTO_USE_EIS}")
        if key == ord('g') and (not metric2_active):
            metric2_active = True; metric2_t0 = now
            metric2_frames_total = 0; metric2_frames_le10 = 0
            _metric2_prev_center = (use_cx, use_cy)
            print("[지표2] 3초 측정 시작: 한 방향으로 30cm를 3초(10cm/s) 이동해 주세요!")

finally:
    if recording and out is not None:
        out.release()
    cap.release()
    cv2.destroyAllWindows()
    q.put(None)

    # 요약
    if len(metric1_times) > 0:
        arr = np.array(metric1_times)
        print(f"[지표1] 재인식시간 mean={arr.mean():.3f}s median={np.median(arr):.3f}s max={arr.max():.3f}s (목표 ≤ 2s)")
    else:
        print("[지표1] 샘플 없음")
    if len(metric2_ratios) > 0:
        arr = np.array(metric2_ratios)
        print(f"[지표2] |Δ|≤10px 비율 mean={arr.mean():.1f}% median={np.median(arr):.1f}% min={arr.min():.1f}% (목표 ≥ 85%)")
    else:
        print("[지표2] 샘플 없음")
    if len(metric3_ratios) > 0:
        arr = np.array(metric3_ratios)
        print(f"[지표3] ICR3 mean={arr.mean():.1f}% median={np.median(arr):.1f}% min={arr.min():.1f}% (목표 ≥ 80%)")
    else:
        print("[지표3] 샘플 없음")
