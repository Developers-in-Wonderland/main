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
# 기능 요약
# - DNN 얼굴검출(SSD ResNet10)
# - Kalman 2D(위치+속도)로 얼굴 중심 부드럽게/예측
# - Affine EIS: 다운샘플+LK 광류 → estimateAffinePartial2D(RANSAC) → 보상행렬 EMA
# - 하이브리드: (광류 보상 아핀) ∘ (얼굴 중심 맞추는 평행이동) 합성 후 워핑
# - 해상도 비례 튜닝: crop_margin, 임계값 등이 프레임 크기에 맞춰 자동 스케일
# - FSM: 얼굴 끊김 시 PREDICT→HOLD→SCAN, 튀지 않게
# - 미리보기/저장 모두 거울반전(설정), 저장본에는 오버레이 절대 X
# - 카운트다운 좌측 하단, 평가지표 자동 로깅
# ============================================================

# ================== 기본 설정 ==================
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

# --- 미러/저장 정책 ---
PREVIEW_MIRROR = True   # 미리보기 거울반전
RECORD_MIRROR  = True   # 저장 영상 거울반전
PHOTO_MIRROR   = True   # 저장 사진 거울반전

RECORD_USE_EIS = True   # 녹화본 EIS 적용
PHOTO_USE_EIS  = False  # 사진은 원본 저장(필요시 True)

OVERLAY_PREVIEW_ONLY = True  # 오버레이는 미리보기 전용

# --- 칼만/광류 파라미터(기본값; 해상도 비례로 재계산됨) ---
FLOW_BETA = 0.35          # 아핀 보상행렬 EMA 계수
FLOW_FEATURE_REFRESH = 15  # 피처 리프레시 주기(프레임)
FLOW_MIN_VALID_BASE = 80   # 최소 유효 포인트(해상도에 비례 상향)
DOWNSAMPLE_SCALE = 0.5     # 광류 추정 해상도 비율(0.5 추천)

# ================== 파일 이름 자동 증가 ==================
def get_new_filename(base_name="output", ext="avi"):
    existing_files = os.listdir(desktop_path)
    pattern = re.compile(rf"{re.escape(base_name)}_(\d+)\.{re.escape(ext)}")
    numbers = [int(m.group(1)) for f in existing_files if (m := pattern.match(f))]
    next_number = max(numbers, default=0) + 1
    return os.path.join(desktop_path, f"{base_name}_{next_number}.{ext}")

def get_new_picture_filename(base_name="picture", ext="jpg"):
    existing_files = os.listdir(desktop_path)
    pattern = re.compile(rf"{re.escape(base_name)}_(\d+)\.{re.escape(ext)}")
    numbers = [int(m.group(1)) for f in existing_files if (m := pattern.match(f))]
    next_number = max(numbers, default=0) + 1
    return os.path.join(desktop_path, f"{base_name}_{next_number}.{ext}")

# ================== 텍스트 드로잉(미리보기 전용) ==================
def draw_text(img, text, org, font_scale=0.8, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness, cv2.LINE_AA)

# ================== 시리얼 송신 스레드 ==================
def serial_worker(q, port='COM3', baud=115200):
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        print("시리얼 연결 완료")
    except Exception as e:
        print(f"시리얼 연결 실패: {e}")
        return

    while True:
        motor_cmds = q.get()
        if motor_cmds is None:
            break
        values = [motor_cmds.get(f"motor_{i}", 0) for i in range(1, 7)]
        message = ','.join(map(str, values)) + '\n'
        ser.write(message.encode('utf-8'))

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
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_thresh:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w-1, x2), min(h-1, y2)
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes

# ================== 로봇팔 제어식(정규화 기반) ==================
def compute_motor_angles(center_x, center_y, area, frame_shape, desired_area=42000):
    h, w = frame_shape[:2]
    dx_n = (center_x - w/2) / (w/2)  # -1..1
    dy_n = (center_y - h/2) / (h/2)
    dz   = (desired_area - area)
    return {
        "motor_1": dy_n * 90 * 0.10,
        "motor_2": dy_n * 90 * 0.10,
        "motor_3": dy_n * 90 * 0.05,
        "motor_4": dz   * 0.0005,
        "motor_5": dy_n * 90 * 0.02,
        "motor_6": dx_n * 90 * 0.05
    }

def clip_motor_angles(motor_cmds, limits=(-90, 90)):
    return {k: int(np.clip(v, limits[0], limits[1])) for k, v in motor_cmds.items()}

# ================== Kalman 2D (x, y, vx, vy) ==================
def init_kalman():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]], dtype=np.float32)
    kf.processNoiseCov = np.diag([1e-2, 1e-2, 1e-1, 1e-1]).astype(np.float32)
    kf.measurementNoiseCov = np.diag([2.0, 2.0]).astype(np.float32)
    kf.errorCovPost = np.diag([10, 10, 10, 10]).astype(np.float32)
    return kf

def kalman_predict(kf, dt):
    kf.transitionMatrix = np.array([[1,0,dt,0],
                                    [0,1,0,dt],
                                    [0,0,1, 0],
                                    [0,0,0, 1]], dtype=np.float32)
    pred = kf.predict()
    return pred[0,0], pred[1,0]

def kalman_correct(kf, x, y):
    meas = np.array([[np.float32(x)], [np.float32(y)]], dtype=np.float32)
    kf.correct(meas)

# ================== Affine EIS (멀티스케일 LK + RANSAC 아핀) ==================
class AffineEIS:
    def __init__(self, beta=0.35, feature_refresh=15, scale=0.5, min_valid=80):
        self.prev_gray_small = None
        self.prev_pts = None
        self.beta = beta
        self.feature_refresh = feature_refresh
        self.frame_count = 0
        self.min_valid = int(min_valid)
        self.scale = scale
        self.Mc_ema = np.array([[1,0,0],[0,1,0]], dtype=np.float32)  # 보상행렬 EMA

    def _detect_features(self, gray_small, mask=None):
        return cv2.goodFeaturesToTrack(gray_small, maxCorners=800, qualityLevel=0.01,
                                       minDistance=12, blockSize=3, mask=mask)

    def update(self, gray_full, face_box=None):
        self.frame_count += 1
        gray_small = cv2.resize(gray_full, (0,0), fx=self.scale, fy=self.scale,
                                interpolation=cv2.INTER_AREA)
        h, w = gray_small.shape[:2]

        # 얼굴 영역 마스크(배경 위주로 광류)
        mask = None
        if face_box is not None:
            x, y, bw, bh = face_box
            x = int(x*self.scale); y = int(y*self.scale)
            bw = int(bw*self.scale); bh = int(bh*self.scale)
            mask = np.full((h, w), 255, dtype=np.uint8)
            x0 = max(0, x-8); y0 = max(0, y-8)
            x1 = min(w-1, x+bw+8); y1 = min(h-1, y+bh+8)
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
            M = np.array([[1,0,0],[0,1,0]], dtype=np.float32)

        # (prev->curr) 아핀의 역행렬 = curr에 적용할 보상행렬
        Mc_new = cv2.invertAffineTransform(np.hstack([M[:,:2], M[:,2:]]))
        Mc_new = Mc_new.astype(np.float32)

        # 행렬 EMA
        self.Mc_ema = (1.0 - self.beta) * self.Mc_ema + self.beta * Mc_new

        # 다음 프레임 준비
        self.prev_gray_small = gray_small
        if (self.frame_count % self.feature_refresh == 0) or (inliers is not None and inliers.sum() < self.min_valid):
            self.prev_pts = self._detect_features(gray_small, mask)
        else:
            self.prev_pts = good_new.reshape(-1,1,2)

        # 다운샘플 좌표계에서 추정했으므로, transl. 항을 스케일업
        s = 1.0 / self.scale
        Mc = self.Mc_ema.copy()
        Mc[:,2] *= s
        return Mc

# ================== FSM (예측/정지/스캔) ==================
hist = deque(maxlen=12)     # (x, y, t)
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
    """
    얼굴 없음: PREDICT(<=0.5s) -> HOLD(<=0.5s) -> SCAN
    """
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
    sxp = (w//2) + int(scan_amp_px * np.sin(scan_phase))
    syp = (h//2)
    return (sxp, syp), mode

# ================== 유틸: 아핀 합성( M_total = Ma ∘ Mb ) ==================
def compose_affine(Ma, Mb):
    """
    Ma, Mb: 2x3
    적용 순서: 먼저 Mb 적용, 그 다음 Ma 적용
    """
    A = np.vstack([Ma, [0,0,1]])
    B = np.vstack([Mb, [0,0,1]])
    C = A @ B
    return C[:2, :]

# ================== 스레드/카메라/상태 ==================
q = queue.Queue()
serial_thread = threading.Thread(target=serial_worker, args=(q,), daemon=True)
serial_thread.start()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # 1080p 권장
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

# === 평가지표 수집 ===
prev_face_found = False
face_lost_ts = None
reacquire_times = []   # 지표1
move_dists = []        # 지표2
still_centers = []     # 지표3
still_start_ts = None

# === 칼만/광류 상태 ===
kf = init_kalman()
kalman_inited = False
last_kf_ts = time.time()
flow_eis = None  # 프레임 크기 알아야 초기화

print("키 안내: s=녹화 시작, e=녹화 종료, 1~9=연속사진(3초 간격), q=종료")

try:
    last_center_used = None
    fps_hint = 30.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 캡처 실패")
            break

        now = time.time()
        h, w = frame.shape[:2]

        # === 해상도 비례 파라미터(프레임 크기 기반) ===
        diag = np.hypot(w, h)
        crop_margin = int(max(60, 0.06 * min(w, h)))          # 화면의 ~6%
        moving_thresh_px = 0.008 * diag                       # 대각선의 ~0.8%
        flow_min_valid = int(FLOW_MIN_VALID_BASE * (w*h) / (640*480))

        # 광류 EIS 지연 초기화(해상도 알게 된 후)
        if flow_eis is None:
            flow_eis = AffineEIS(beta=FLOW_BETA, feature_refresh=FLOW_FEATURE_REFRESH,
                                 scale=DOWNSAMPLE_SCALE, min_valid=flow_min_valid)

        dt = max(1e-3, now - last_kf_ts)
        last_kf_ts = now

        # ---------- 얼굴 검출 ----------
        face_boxes = detect_faces_dnn(frame)
        face_found = len(face_boxes) > 0

        if face_found:
            face_boxes.sort(key=lambda b: b[2]*b[3], reverse=True)
            x, y, w0, h0 = face_boxes[0]
            center_x = x + w0 // 2
            center_y = y + h0 // 2
            area = w0 * h0
            if not kalman_inited:
                kf.statePost = np.array([[center_x],[center_y],[0],[0]], dtype=np.float32)
                kalman_inited = True
            kx, ky = kalman_predict(kf, dt)
            kalman_correct(kf, center_x, center_y)
            kx, ky = kf.statePost[0,0], kf.statePost[1,0]
            if now - last_face_log_ts >= LOG_INTERVAL:
                print(f"[FACE] center=({center_x}, {center_y}) kalman=({int(kx)}, {int(ky)}) area={area}")
                last_face_log_ts = now
        else:
            x = y = w0 = h0 = 0
            center_x = center_y = None
            area = 42000
            kx, ky = kalman_predict(kf, dt) if kalman_inited else (w//2, h//2)

        # ---------- FSM ----------
        (ux_fsm, uy_fsm), track_mode = update_tracker(face_found, center_x, center_y, now, frame.shape, fps_hint=fps_hint)

        # ---------- 사용할 중심 선택 ----------
        if face_found:
            use_cx, use_cy = int(kx), int(ky)     # 얼굴 보이면 칼만 추정
            face_box_for_flow = (x, y, w0, h0)    # 광류용 마스크
        else:
            if track_mode == "SCAN":
                use_cx, use_cy = ux_fsm, uy_fsm   # 스캔 중심
            else:
                use_cx, use_cy = int(kx), int(ky) # PREDICT/HOLD: 칼만 예측
            face_box_for_flow = None

        # ---------- 광류 기반 보상 아핀행렬 ----------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        Mc = flow_eis.update(gray, face_box=face_box_for_flow)   # 2x3

        # ---------- 얼굴 중심을 화면 중앙으로 평행이동(추가 보정) ----------
        target = np.array([w//2, h//2], dtype=np.float32)
        face_shift = target - np.array([use_cx, use_cy], dtype=np.float32)
        # 너무 큰 평행이동은 크롭 여백 내로 제한
        face_shift[0] = float(np.clip(face_shift[0], -(crop_margin-5), (crop_margin-5)))
        face_shift[1] = float(np.clip(face_shift[1], -(crop_margin-5), (crop_margin-5)))
        T_face = np.array([[1,0,face_shift[0]],[0,1,face_shift[1]]], dtype=np.float32)

        # ---------- 합성 아핀 행렬: (얼굴 평행이동) ∘ (광류 보상) ----------
        M_total = compose_affine(T_face, Mc)

        # ---------- 평가지표용 이동량 ----------
        is_moving = False
        if last_center_used is not None:
            dxm = use_cx - last_center_used[0]
            dym = use_cy - last_center_used[1]
            distm = np.hypot(dxm, dym)
            is_moving = distm > moving_thresh_px
            move_dists.append(distm)
        last_center_used = (use_cx, use_cy)

        # ---------- 안정화 워핑 + 중앙 크롭/리사이즈 ----------
        stabilized = cv2.warpAffine(frame, M_total, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        x1 = crop_margin; y1 = crop_margin
        x2 = w - crop_margin; y2 = h - crop_margin
        sx_scale = w / float(w - 2*crop_margin)
        sy_scale = h / float(h - 2*crop_margin)
        stab_base = cv2.resize(stabilized[y1:y2, x1:x2], (w, h), interpolation=cv2.INTER_LINEAR)

        # ---------- 저장(깨끗한 프레임) ----------
        if recording and out is not None:
            clean_for_record = stab_base if RECORD_USE_EIS else frame
            if RECORD_MIRROR:
                clean_for_record = cv2.flip(clean_for_record, 1)
            out.write(clean_for_record)

        # 연속 촬영(사진)
        if photo_shooting and next_shot_at is not None:
            if now >= next_shot_at:
                filename = get_new_picture_filename()
                clean_for_photo = (stab_base if PHOTO_USE_EIS else frame)
                if PHOTO_MIRROR:
                    clean_for_photo = cv2.flip(clean_for_photo, 1)
                cv2.imwrite(filename, clean_for_photo)
                photo_taken += 1
                print(f"{photo_taken}/{photo_count} 저장됨: {os.path.basename(filename)}")
                if photo_taken >= photo_count:
                    photo_shooting = False
                    next_shot_at = None
                    print("연속 사진 촬영 완료!")
                else:
                    next_shot_at = now + photo_interval

        # ---------- 미리보기(오버레이 전용) ----------
        display = stab_base.copy()
        if PREVIEW_MIRROR:
            display = cv2.flip(display, 1)

        # === 좌표 매핑(아핀 → 크롭/스케일 → 미러) ===
        def map_point(px, py):
            # 아핀 변환
            v = np.array([[px, py]], dtype=np.float32)
            v2 = cv2.transform(v.reshape(-1,1,2), M_total).reshape(-1,2)[0]
            # 크롭/스케일
            vx = (v2[0] - crop_margin) * sx_scale
            vy = (v2[1] - crop_margin) * sy_scale
            # 미러
            if PREVIEW_MIRROR:
                vx = w - 1 - vx
            return int(np.clip(vx, 0, w-1)), int(np.clip(vy, 0, h-1))

        def map_rect(rx, ry, rw0, rh0):
            p1 = map_point(rx, ry)
            p2 = map_point(rx+rw0, ry+rh0)
            x0m, y0m = p1
            x1m, y1m = p2
            x_rect = min(x0m, x1m)
            y_rect = min(y0m, y1m)
            w_rect = max(1, abs(x1m - x0m))
            h_rect = max(1, abs(y1m - y0m))
            return x_rect, y_rect, w_rect, h_rect

        if OVERLAY_PREVIEW_ONLY:
            # 모드
            draw_text(display, f"MODE: {track_mode}", (10, 30), 0.7, 2)
            # 중심(칼만/스캔)
            cx_disp, cy_disp = map_point(use_cx, use_cy)
            cv2.circle(display, (cx_disp, cy_disp), 3, (0, 200, 0), -1)
            # 얼굴 박스/텍스트
            if face_found:
                rx, ry, rwv, rhv = map_rect(x, y, w0, h0)
                cv2.rectangle(display, (rx, ry), (rx + rwv, ry + rhv), (0, 200, 0), 2)
                draw_text(display, f"Center: ({center_x}, {center_y})", (10, 60), 0.7, 2)
                draw_text(display, f"Area: {area}", (10, 84), 0.7, 2)
            # REC
            if recording:
                draw_text(display, "REC ●", (10, h-15), 0.8, 2)
            # 카운트다운(좌하단)
            if photo_shooting and next_shot_at is not None:
                remain_sec = max(0.0, next_shot_at - now)
                remain_ceil = int(np.ceil(remain_sec))
                cd = "3" if remain_ceil >= 3 else "2" if remain_ceil == 2 else "1 cheese~!" if remain_ceil == 1 else "cheese~!"
                y_cd = (h - 50) if recording else (h - 20)
                draw_text(display, cd, (20, y_cd), 1.2, 2)
                # 잔여 장수(우상단)
                shots_left = max(0, photo_count - photo_taken)
                (tw, th), _ = cv2.getTextSize(f"Shots left: {shots_left}", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                draw_text(display, f"Shots left: {shots_left}", (w - tw - 20, 40), 0.8, 2)

        cv2.imshow("Face Tracker (Kalman + Affine EIS; mirror; clean saves)", display)

        # ---------- 로봇팔 제어 ----------
        cmds = compute_motor_angles(use_cx, use_cy, area, frame.shape)
        cmds = clip_motor_angles(cmds, (-90, 90))
        q.put(cmds)

        # ---------- 평가지표 ----------
        if face_found and prev_face_found is False and face_lost_ts is not None:
            reacquire_times.append(now - face_lost_ts)
        if not face_found and prev_face_found is True:
            face_lost_ts = now
        prev_face_found = face_found

        # 정지 3초(반경 9px) 안정도
        if not is_moving:
            if still_start_ts is None:
                still_start_ts = now
                still_centers = []
            still_centers.append(last_center_used)
            if (now - still_start_ts) >= 3.0:
                c0 = still_centers[0]; r = 9.0
                inside = [(np.hypot(c[0]-c0[0], c[1]-c0[1]) <= r) for c in still_centers]
                ratio_inside = (np.sum(inside) / len(still_centers)) * 100.0
                print(f"[정지 3초 떨림 보정] 원(9px) 안 비율 = {ratio_inside:.1f}% (목표 ≥ 80%)")
                still_start_ts = None; still_centers = []
        else:
            still_start_ts = None; still_centers = []

        # ---------- 키 입력 ----------
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # 녹화 시작
        if key == ord('s') and not recording:
            output_path = get_new_filename()
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))
            if not out.isOpened():
                print("VideoWriter 열기 실패"); out = None
            else:
                recording = True
                print(f"녹화 시작! 저장 파일명: {os.path.basename(output_path)}")

        # 녹화 종료
        if key == ord('e') and recording:
            recording = False
            if out is not None:
                out.release(); out = None
            print("녹화 종료! 영상이 저장되었습니다.")

        # 연속촬영 시작 (1~9)
        if (ord('1') <= key <= ord('9')) and not photo_shooting:
            photo_count = key - ord('0')
            photo_taken = 0
            photo_shooting = True
            next_shot_at = now + photo_interval
            print(f"{photo_count}장의 사진 연속 촬영 시작! (간격 {photo_interval:.0f}초)")

finally:
    if recording and out is not None:
        out.release()
    cap.release()
    cv2.destroyAllWindows()
    q.put(None)

    # ==== 평가지표 요약 출력 ====
    if len(reacquire_times) > 0:
        arr = np.array(reacquire_times)
        print(f"[지표1 재획득시간] mean={arr.mean():.2f}s, median={np.median(arr):.2f}s, max={arr.max():.2f}s (목표 ≤ 2s)")
    if len(move_dists) > 0:
        arr2 = np.array(move_dists)
        ratio_under_10 = (np.sum(arr2 <= 10.0) / len(arr2)) * 100.0
        print(f"[지표2 이동중 |Δ|≤10px 비율] {ratio_under_10:.1f}% (실험구간 기준)")
