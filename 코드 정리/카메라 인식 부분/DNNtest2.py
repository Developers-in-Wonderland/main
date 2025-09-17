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
# - Kalman 2D 추정(위치+속도)로 얼굴 중심 부드럽게/예측
# - 광학 흐름(LK) 기반 EIS: 배경 모션 추정 → 흔들림 상쇄(EMA)
# - 하이브리드 안정화: 얼굴(칼만) + 광류 결합 shift
# - 예측/정지/스캔 FSM: 미검출 시 튀지 않게 처리
# - 미리보기/저장 모두 거울반전, 저장본은 오버레이 없음
# - 평가지표 로깅: 재획득시간, 이동중 흔들림, 정지 3초 안정도
# ============================================================

# ================== 기본 설정 ==================
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

# --- 반전/저장 정책 ---
PREVIEW_MIRROR = True   # 미리보기 거울반전
RECORD_MIRROR  = True   # 저장 영상 거울반전
PHOTO_MIRROR   = True   # 저장 사진 거울반전

RECORD_USE_EIS = True   # 녹화본 EIS 적용
PHOTO_USE_EIS  = False  # 사진은 원본 저장(필요시 True)

OVERLAY_PREVIEW_ONLY = True  # 오버레이는 미리보기 전용

# --- EIS/광류/칼만 파라미터 ---
crop_margin = 70            # 안정화 크롭 여유(px)
moving_thresh_px = 7.0      # 프레임간 이동량 임계(움직임 판단)
flow_beta = 0.4             # 광류 보정벡터 EMA 계수
flow_feature_refresh = 15    # 광류 피처 주기적 리프레시(프레임 단위)
flow_min_valid = 40          # 유효 광류 포인트 최소 개수
flow_dx_cap = 50             # 이상치 제거(한 프레임 당 최대 이동)
lambda_face_found = (1.0, 0.30)   # (face, flow) 가중치 (얼굴 보일 때)
lambda_predict = (0.8, 0.6)       # 미검출 단기 예측/홀드
lambda_scan = (0.0, 1.0)          # 스캔 중

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

# ================== 텍스트 그리기(미리보기 전용) ==================
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
            boxes.append((x1, y1, x2 - x1, y2 - y2 + y2 - y1))  # robust
            # 위 줄은 안전을 위해 재계산: w = x2-x1, h = y2-y1
            boxes[-1] = (x1, y1, x2 - x1, y2 - y1)
    return boxes

# ================== 로봇팔 제어식 ==================
def compute_motor_angles(center_x, center_y, area, frame_shape, desired_area=42000):
    frame_h, frame_w = frame_shape[:2]
    dx = center_x - (frame_w // 2)
    dy = center_y - (frame_h // 2)
    dz = desired_area - area
    return {
        "motor_1": dy * 0.1,
        "motor_2": dy * 0.1,
        "motor_3": dy * 0.05,
        "motor_4": dz * 0.0005,
        "motor_5": dy * 0.02,
        "motor_6": dx * 0.05
    }

def clip_motor_angles(motor_cmds, limits=(-90, 90)):
    return {k: int(np.clip(v, limits[0], limits[1])) for k, v in motor_cmds.items()}

# ================== Kalman 2D (x, y, vx, vy) ==================
def init_kalman():
    kf = cv2.KalmanFilter(4, 2)
    # 상태: [x, y, vx, vy]^T, 측정: [x, y]^T
    kf.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]], dtype=np.float32)
    # 공분산/노이즈(대략값; 환경에 맞게 튜닝)
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
    return pred[0,0], pred[1,0]  # x, y

def kalman_correct(kf, x, y):
    meas = np.array([[np.float32(x)], [np.float32(y)]], dtype=np.float32)
    kf.correct(meas)

# ================== 광학 흐름 기반 EIS ==================
class OptFlowEIS:
    def __init__(self, beta=0.4, feature_refresh=15, min_valid=40):
        self.prev_gray = None
        self.prev_pts = None
        self.comp_ema = np.zeros(2, dtype=np.float32)  # 누적 보정벡터(EMA)
        self.beta = beta
        self.frame_count = 0
        self.feature_refresh = feature_refresh
        self.min_valid = min_valid

    def _detect_features(self, gray, mask=None):
        return cv2.goodFeaturesToTrack(
            gray, maxCorners=300, qualityLevel=0.01, minDistance=15,
            blockSize=3, mask=mask
        )

    def update(self, gray, face_box=None):
        """
        gray: 현재 그레이스케일
        face_box: (x,y,w,h) — 있으면 그 영역은 마스크로 제외(배경 중심)
        반환: 보정 shift 벡터 (dx, dy) — 흔들림 보상용 (이미 부호 보정 & EMA 적용)
        """
        self.frame_count += 1
        h, w = gray.shape[:2]

        # 얼굴 마스크 제외(배경 중심으로 광류)
        mask = None
        if face_box is not None:
            mask = np.full((h, w), 255, dtype=np.uint8)
            x, y, bw, bh = face_box
            x0 = max(0, x-10); y0 = max(0, y-10)
            x1 = min(w-1, x+bw+10); y1 = min(h-1, y+bh+10)
            mask[y0:y1, x0:x1] = 0

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = self._detect_features(gray, mask)
            return self.comp_ema.copy()

        if self.prev_pts is None or len(self.prev_pts) < 10:
            self.prev_pts = self._detect_features(self.prev_gray, mask)

        if self.prev_pts is None:
            self.prev_gray = gray
            return self.comp_ema.copy()

        nxt_pts, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None,
                                                    winSize=(21,21), maxLevel=3,
                                                    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        if nxt_pts is None or st is None:
            self.prev_gray = gray
            self.prev_pts = self._detect_features(gray, mask)
            return self.comp_ema.copy()

        good_new = nxt_pts[st==1]
        good_old = self.prev_pts[st==1]

        if len(good_new) < self.min_valid:
            # 피처 부족 → 재검출
            self.prev_gray = gray
            self.prev_pts = self._detect_features(gray, mask)
            return self.comp_ema.copy()

        # 이동 벡터(이상치 제거)
        flow = (good_new - good_old).reshape(-1, 2)
        flow = np.clip(flow, -flow_dx_cap, flow_dx_cap)
        dx_med = np.median(flow[:,0])
        dy_med = np.median(flow[:,1])

        # 흔들림 보상 벡터(관측 반대방향)
        comp = np.array([-dx_med, -dy_med], dtype=np.float32)
        # EMA로 부드럽게
        self.comp_ema = (1.0 - self.beta) * self.comp_ema + self.beta * comp

        # 다음 프레임 준비
        self.prev_gray = gray
        # 주기적으로/부족하면 리프레시, 아니면 현재 피처 이어가기
        if (self.frame_count % self.feature_refresh == 0) or len(good_new) < self.min_valid:
            self.prev_pts = self._detect_features(gray, mask)
        else:
            self.prev_pts = good_new.reshape(-1,1,2)

        return self.comp_ema.copy()

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
    항상 추적 중심을 반환 (예측/스캔 포함)
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

# ================== 스레드/카메라/상태 ==================
q = queue.Queue()
serial_thread = threading.Thread(target=serial_worker, args=(q,), daemon=True)
serial_thread.start()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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
flow_eis = OptFlowEIS(beta=flow_beta, feature_refresh=flow_feature_refresh, min_valid=flow_min_valid)

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
                # 초기 상태 세팅
                kf.statePost = np.array([[center_x],[center_y],[0],[0]], dtype=np.float32)
                kalman_inited = True
            # 칼만: 예측→보정
            kx, ky = kalman_predict(kf, dt)
            kalman_correct(kf, center_x, center_y)
            kx, ky = kf.statePost[0,0], kf.statePost[1,0]

            if now - last_face_log_ts >= LOG_INTERVAL:
                print(f"[FACE] center=({center_x}, {center_y}) kalman=({int(kx)}, {int(ky)}) area={area}")
                last_face_log_ts = now
        else:
            x = y = w0 = h0 = 0
            center_x = center_y = None
            area = 42000  # 목표 면적 유지
            # 칼만: 측정 없음 → 예측만
            kx, ky = kalman_predict(kf, dt) if kalman_inited else (frame.shape[1]//2, frame.shape[0]//2)

        # ---------- FSM (예측/정지/스캔) ----------
        (ux_fsm, uy_fsm), track_mode = update_tracker(face_found, center_x, center_y, now, frame.shape, fps_hint=fps_hint)

        # ---------- 추적 중심 선택(칼만/스캔 혼합) ----------
        # 얼굴 보이면: 칼만 추정 사용, 아니면 FSM 결과(스캔 포함)
        if face_found:
            use_cx, use_cy = int(kx), int(ky)
            w_face, w_flow = lambda_face_found
            face_box_for_flow = (x, y, w0, h0)
        else:
            if track_mode == "SCAN":
                use_cx, use_cy = ux_fsm, uy_fsm  # 스캔 중심
                w_face, w_flow = lambda_scan
            else:
                # PREDICT/HOLD: 칼만 예측 위주
                use_cx, use_cy = int(kx), int(ky)
                w_face, w_flow = lambda_predict
            face_box_for_flow = None  # 얼굴 없으니 마스크X

        # ---------- 광류 기반 흔들림 보상벡터 ----------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow_shift = flow_eis.update(gray, face_box=face_box_for_flow)  # (dx, dy) — 화면 보정용

        # ---------- 하이브리드 안정화: 얼굴(칼만) + 광류 결합 shift ----------
        h, w = frame.shape[:2]
        target = np.array([w//2, h//2], dtype=np.float32)
        face_shift = target - np.array([use_cx, use_cy], dtype=np.float32)  # 얼굴을 중앙으로
        total_shift = (w_face * face_shift) + (w_flow * flow_shift)

        # 너무 큰 shift는 crop 여유 내로 클립(검은 테두리 방지)
        total_shift[0] = float(np.clip(total_shift[0], -(crop_margin-5), (crop_margin-5)))
        total_shift[1] = float(np.clip(total_shift[1], -(crop_margin-5), (crop_margin-5)))

        # 평가지표용 이동량/정지 판단
        is_moving = False
        if last_center_used is not None:
            dxm = use_cx - last_center_used[0]
            dym = use_cy - last_center_used[1]
            distm = np.hypot(dxm, dym)
            is_moving = distm > moving_thresh_px
            move_dists.append(distm)
        last_center_used = (use_cx, use_cy)

        # ---------- 안정화 워핑 + 크롭/리사이즈 ----------
        M = np.float32([[1, 0, total_shift[0]], [0, 1, total_shift[1]]])
        shifted = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        x1 = crop_margin; y1 = crop_margin
        x2 = w - crop_margin; y2 = h - crop_margin
        cropped = shifted[y1:y2, x1:x2]
        sx_scale = w / float(w - 2*crop_margin)
        sy_scale = h / float(h - 2*crop_margin)
        stab_base = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

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

        # ---------- 미리보기 화면(오버레이 전용) ----------
        display = stab_base.copy()
        if PREVIEW_MIRROR:
            display = cv2.flip(display, 1)

        # 좌표 매핑 함수(워프+크롭+스케일+미러 반영)
        def map_point_to_display(px, py):
            sxp = px + total_shift[0]          # 워프 평행이동
            syp = py + total_shift[1]
            sxp = (sxp - crop_margin) * sx_scale  # 크롭/스케일 반영
            syp = (syp - crop_margin) * sy_scale
            if PREVIEW_MIRROR:
                sxp = w - 1 - sxp
            return int(np.clip(sxp, 0, w-1)), int(np.clip(syp, 0, h-1))

        def map_rect_to_display(rx, ry, rw0, rh0):
            x1d = (rx + total_shift[0] - crop_margin) * sx_scale
            y1d = (ry + total_shift[1] - crop_margin) * sy_scale
            rw_s = rw0 * sx_scale
            rh_s = rh0 * sy_scale
            if PREVIEW_MIRROR:
                x1d = w - (x1d + rw_s)
            return (int(np.clip(x1d, 0, w-1)),
                    int(np.clip(y1d, 0, h-1)),
                    int(np.clip(rw_s, 1, w)),
                    int(np.clip(rh_s, 1, h)))

        if OVERLAY_PREVIEW_ONLY:
            # 모드/중심점
            draw_text(display, f"MODE: {track_mode}", (10, 30), 0.7, 2)
            cx_disp, cy_disp = map_point_to_display(use_cx, use_cy)
            cv2.circle(display, (cx_disp, cy_disp), 3, (0, 200, 0), -1)

            # 얼굴 박스/텍스트(실제 검출 시)
            if face_found:
                rx, ry, rwv, rhv = map_rect_to_display(x, y, w0, h0)
                cv2.rectangle(display, (rx, ry), (rx + rwv, ry + rhv), (0, 200, 0), 2)
                draw_text(display, f"Center: ({int(center_x)}, {int(center_y)})", (10, 60), 0.7, 2)
                draw_text(display, f"Area: {area}", (10, 84), 0.7, 2)

            # 녹화중 표시는 미리보기 전용(좌하단)
            if recording:
                draw_text(display, "REC ●", (10, h-15), 0.8, 2)

            # 카운트다운(좌하단)
            if photo_shooting and next_shot_at is not None:
                remain_sec = max(0.0, next_shot_at - now)
                remain_ceil = int(np.ceil(remain_sec))
                cd = "3" if remain_ceil >= 3 else "2" if remain_ceil == 2 else "1 cheese~!" if remain_ceil == 1 else "cheese~!"
                y_cd = (h - 50) if recording else (h - 20)
                draw_text(display, cd, (20, y_cd), 1.2, 2)
                # 남은 장수(우상단)
                shots_left = max(0, photo_count - photo_taken)
                (tw, th), _ = cv2.getTextSize(f"Shots left: {shots_left}", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                draw_text(display, f"Shots left: {shots_left}", (w - tw - 20, 40), 0.8, 2)

        # ---------- 미리보기 출력 ----------
        cv2.imshow("Face Tracker (Kalman + OpticalFlow EIS; mirror; clean saves)", display)

        # ---------- 로봇팔 제어 ----------
        cmds = compute_motor_angles(use_cx, use_cy, area, frame.shape)
        cmds = clip_motor_angles(cmds, (-90, 90))
        q.put(cmds)

        # ---------- 평가지표 이벤트 기록 ----------
        if face_found and prev_face_found is False and face_lost_ts is not None:
            reacquire_times.append(now - face_lost_ts)
        if not face_found and prev_face_found is True:
            face_lost_ts = now
        prev_face_found = face_found

        # 지표3: 정지 3초(반경 9px 원 내부 비율)
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
            h0, w0v = frame.shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (w0v, h0))
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
