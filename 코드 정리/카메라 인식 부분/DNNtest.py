import cv2
import numpy as np
import threading
import queue
import serial
import os
import re
import time
from collections import deque

##칼만 필터 공부해서 바꿔보기
## 얼굴이 가까워지거나 너무 옆이나 위로 빠지면 화면 늘어짐 해결해야함

# ============================================================
# 기능 요약
# - DNN 얼굴검출(SSD ResNet10) → 로봇팔 제어값 계산/전송(스레드)
# - EIS(가상 짐벌): 얼굴 중심 EMA 스무딩 → 평행이동 + 중앙 크롭/리사이즈
# - 예측/정지/스캔 FSM: 미검출 시 튀지 않게 PREDICT→HOLD→SCAN
# - 미리보기/저장 모두 거울반전(옵션) & 저장본에는 오버레이 미포함
# - 평가지표 로깅: 재획득시간, 이동중 프레임간 이동량, 정지 3초 떨림
# - 카운트다운(3/2/1/cheese~) 좌측 하단 배치(REC와 안 겹치게)
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
            boxes.append((x1, y1, x2 - x1, y2 - y1))
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

# ================== EIS(가상 짐벌) ==================
eis_target = None
alpha_move, alpha_still = 0.35, 0.20  # 움직일 때/정지 때 EMA
crop_margin = 70                      # 테두리 여유
moving_thresh_px = 7.0                # 움직임 판정 임계

def ema_update(prev, curr, alpha):
    if prev is None:
        return curr
    return (1 - alpha) * prev + alpha * curr

def stabilize_frame(frame, face_center, is_moving=True, return_params=False):
    """
    얼굴 중심을 EMA로 스무딩 → 평행이동 → 중앙 크롭 → 원본크기 리사이즈
    return_params=True: (stabilized, shift, sx, sy) 반환
      - shift: warpAffine에 쓰인 (dx, dy)
      - sx, sy: 가로/세로 스케일(크롭 후 리사이즈 배율)
    """
    global eis_target
    h, w = frame.shape[:2]
    cx, cy = face_center
    target_center = np.array([w//2, h//2], dtype=np.float32)
    curr = np.array([cx, cy], dtype=np.float32)
    alpha = alpha_move if is_moving else alpha_still
    eis_target = ema_update(eis_target, curr, alpha)

    shift = (target_center - eis_target)
    M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    shifted = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # 중앙 크롭 후 원본 크기로 리사이즈
    x1 = crop_margin; y1 = crop_margin
    x2 = w - crop_margin; y2 = h - crop_margin
    cropped = shifted[y1:y2, x1:x2]

    # 크롭→리사이즈 스케일
    sx = w / float(w - 2*crop_margin)
    sy = h / float(h - 2*crop_margin)
    stabilized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    if return_params:
        return stabilized, shift, sx, sy
    return stabilized

# ================== 미검출 시 예측/정지/스캔 FSM ==================
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

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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

print("키 안내: s=녹화 시작, e=녹화 종료, 1~9=연속사진(3초 간격), q=종료")

try:
    last_center_shown = None
    fps_hint = 30.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 캡처 실패")
            break

        now = time.time()

        # ---------- 얼굴 검출 ----------
        face_boxes = detect_faces_dnn(frame)
        face_found = len(face_boxes) > 0

        if face_found:
            face_boxes.sort(key=lambda b: b[2]*b[3], reverse=True)
            x, y, w0, h0 = face_boxes[0]
            center_x = x + w0 // 2
            center_y = y + h0 // 2
            area = w0 * h0
            if now - last_face_log_ts >= LOG_INTERVAL:
                print(f"[FACE] center=({center_x}, {center_y}), area={area}")
                last_face_log_ts = now
        else:
            x = y = w0 = h0 = 0
            center_x = center_y = None
            area = 42000  # 목표 면적 유지

        # ---------- 추적 상태(FSM) ----------
        (ux, uy), track_mode = update_tracker(face_found, center_x, center_y, now, frame.shape, fps_hint=fps_hint)

        # ---------- EIS: 저장용 베이스 프레임 생성 ----------
        is_moving = False
        if last_center_shown is not None:
            dxm = ux - last_center_shown[0]
            dym = uy - last_center_shown[1]
            distm = np.hypot(dxm, dym)
            is_moving = distm > moving_thresh_px
            move_dists.append(distm)
        last_center_shown = (ux, uy)

        stab_base, shift, sx_scale, sy_scale = stabilize_frame(
            frame, (ux, uy), is_moving=is_moving, return_params=True
        )

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

        # ---------- 미리보기 화면 구성(오버레이 전용) ----------
        # 1) 기본 베이스: stab_base
        display = stab_base.copy()

        # 2) 미리보기 거울반전
        if PREVIEW_MIRROR:
            display = cv2.flip(display, 1)

        # 3) 오버레이 좌표 보정(안정화 + 크롭 스케일 + 반전)
        h, w = display.shape[:2]
        def map_point_to_display(px, py):
            # (a) 안정화 평행이동 반영
            sxp = px + shift[0]
            syp = py + shift[1]
            # (b) 크롭 제거 + 스케일 적용
            sxp = (sxp - crop_margin) * sx_scale
            syp = (syp - crop_margin) * sy_scale
            # (c) 반전
            if PREVIEW_MIRROR:
                sxp = w - 1 - sxp
            return int(np.clip(sxp, 0, w-1)), int(np.clip(syp, 0, h-1))

        def map_rect_to_display(rx, ry, rw, rh):
            # 좌상단 변환 + 폭/높이 스케일
            x1 = (rx + shift[0] - crop_margin) * sx_scale
            y1 = (ry + shift[1] - crop_margin) * sy_scale
            rw_s = rw * sx_scale
            rh_s = rh * sy_scale
            if PREVIEW_MIRROR:
                # 거울반전: 좌상단 x' = w - (x + w_rect)
                x1 = w - (x1 + rw_s)
            return (int(np.clip(x1, 0, w-1)),
                    int(np.clip(y1, 0, h-1)),
                    int(np.clip(rw_s, 1, w)),
                    int(np.clip(rh_s, 1, h)))

        if OVERLAY_PREVIEW_ONLY:
            # 모드 표시 (좌상단)
            draw_text(display, f"MODE: {track_mode}", (10, 30), 0.7, 2)

            # 추적 중심(예측/스캔 포함)
            cx_disp, cy_disp = map_point_to_display(ux, uy)
            cv2.circle(display, (cx_disp, cy_disp), 3, (0, 200, 0), -1)

            # 얼굴 박스/텍스트(실제 검출 시)
            if face_found:
                rx, ry, rwv, rhv = map_rect_to_display(x, y, w0, h0)
                cv2.rectangle(display, (rx, ry), (rx + rwv, ry + rhv), (0, 200, 0), 2)
                draw_text(display, f"Center: ({center_x}, {center_y})", (10, 60), 0.7, 2)
                draw_text(display, f"Area: {area}", (10, 84), 0.7, 2)

            # 녹화중 표시는 미리보기 전용(좌하단)
            if recording:
                draw_text(display, "REC ●", (10, h-15), 0.8, 2)

            # === 카운트다운/잔여 장수 ===
            if photo_shooting and next_shot_at is not None:
                remain_sec = max(0.0, next_shot_at - now)
                remain_ceil = int(np.ceil(remain_sec))
                cd = "3" if remain_ceil >= 3 else "2" if remain_ceil == 2 else "1 cheese~!" if remain_ceil == 1 else "cheese~!"
                # 좌측 하단(REC와 겹치지 않게 녹화 중이면 살짝 위)
                y_cd = (h - 50) if recording else (h - 20)
                draw_text(display, cd, (20, y_cd), 1.2, 2)

                # 남은 장수(우측 상단)
                shots_left = max(0, photo_count - photo_taken)
                (tw, th), _ = cv2.getTextSize(f"Shots left: {shots_left}", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                draw_text(display, f"Shots left: {shots_left}", (w - tw - 20, 40), 0.8, 2)

        # ---------- 미리보기 출력 ----------
        cv2.imshow("Face Tracker (Mirror preview; clean mirrored recordings)", display)

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
            still_centers.append(last_center_shown)
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
