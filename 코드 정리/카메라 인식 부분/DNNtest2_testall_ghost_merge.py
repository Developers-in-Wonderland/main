import cv2
import numpy as np
import threading
import queue
import serial
import os
import re
import time
from collections import deque

# ====== (추가) 한글 텍스트 유지를 위한 PIL 사용 ======
from PIL import ImageFont, ImageDraw, Image

FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"   # 한글 폰트 경로 (Windows 기본: 맑은 고딕)
def draw_text_kr(img, text, org, font_size=26, thickness=2):
    """
    PIL로 한글 렌더링 후 OpenCV 이미지에 반영.
    org는 좌상단 기준 좌표 (x, y)
    """
    if not text:
        return img
    # PIL 이미지/드로우
    img_pil = Image.fromarray(img)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except:
        # 폰트 로드 실패시: OpenCV 기본 폰트로 대체(영문만 정상)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, max(0.5, font_size/26.0),
                    (255,255,255), thickness+2, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, max(0.5, font_size/26.0),
                    (0,0,0), thickness, cv2.LINE_AA)
        return img
    draw = ImageDraw.Draw(img_pil)
    x, y = org
    # 외곽 흰색 테두리 효과 흉내
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        draw.text((x+dx, y+dy), text, font=font, fill=(255,255,255))
    draw.text((x, y), text, font=font, fill=(0,0,0))
    return np.array(img_pil)

# ============================================================
# 중앙 고정 + 무왜곡(균일 스케일) 줌 + 평행이동 안정화
# ============================================================

# ------------------ 기본/저장 ------------------
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
CAP_WIDTH, CAP_HEIGHT, CAP_FPS = 1920, 1080, 60
RECORD_USE_STAB = True

# ------------------ 검출/추적 ------------------
DETECT_EVERY = 1
LEAD_FACE_SEC = 0.12
CM_PER_PIXEL = 0.050

# ------------------ 제어(로봇팔) ------------------
DEAD_X, DEAD_Y = 0.02, 0.02
D_ALPHA, D_CLAMP = 0.25, 5.0
SLEW = 8.0

# ------------------ 중앙고정 & 줌 ------------------
TARGET_FACE_FRAC = 0.26
ZOOM_MIN, ZOOM_MAX = 1.00, 1.80
ZOOM_SLEW_PER_S = 0.60
ZOOM_QUANT = 0.02
ZOOM_HYST = 0.06
RATIO_TRANSLATE = 0.3
SAFETY_PX = 6

# ------------------ 정량지표 설정 ------------------
# [지표1] 얼굴 재인식 시간
reacquire_t0 = None
metric1_times = []              # sec
# [지표1-부가] 초속(px/s, cm/s)
metric1_speeds_px = []          # px/s
metric1_speeds_cm = []          # cm/s (CM_PER_PIXEL 사용)

# [지표2] 추적 안정성 (프레임 간 이동 ≤ DT_THRESH_PX 비율)
DT_THRESH_PX = 10.0
STAB_WIN_SEC = 3.0
stab_buf = deque()              # (t, dist)
metric2_ratios = []

# [지표3] ICR3 (정지 구간 원내 비율)
STOP_SPEED_THR = 10.0            # px/s 이하면 정지 간주
STOP_HOLD_START= 0.5
STOP_HOLD_SEC  = 3.0            # 정지 유지 구간
icr3_phase = "idle"             # 'idle' | 'collect'
icr3_center = None              # (cx, cy)
icr3_t0 = 0.0
icr3_inside = 0
icr3_total  = 0
ICR_RATIO = 0.03                # 원 반경 Spec (3%니까 0.03)
ICR_RADIUS = 0.0                # 원 반경(px) — ㅁ티에서 계산
metric3_ratios = []
matric3_text = ""

_prev_cx, _prev_cy = None, None
_prev_t = None

# ------------------ 도우미 ------------------
def get_new_filename(base_name="output", ext="avi"):
    existing = os.listdir(desktop_path)
    pat = re.compile(rf"{re.escape(base_name)}_(\d+)\.{re.escape(ext)}")
    nums = [int(m.group(1)) for f in existing if (m := pat.match(f))]
    n = max(nums, default=0) + 1
    return os.path.join(desktop_path, f"{base_name}_{n}.{ext}")


def get_new_image_filename(base_name="shot", ext="jpg"):
    return get_new_filename(base_name, ext)

def est_speed_px_per_s(cx, cy, prev_cx, prev_cy, dt):
    if prev_cx is None or dt <= 0:
        return 0.0
    dx = float(cx - prev_cx)
    if dx<5 :
        dx = 0
    dy = float(cy - prev_cy)
    if dy<5 :
        dy = 0

    return (dx*dx + dy*dy) ** 0.5 / max(dt, 1e-6)

# ------------------ One Euro ------------------
class OneEuro:
    def __init__(self, min_cutoff=0.8, beta=0.04, dcutoff=1.0):
        self.min_cutoff = float(min_cutoff); self.beta = float(beta); self.dcutoff = float(dcutoff)
        self.x_prev = None; self.dx_prev = 0.0; self.t_prev = None
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

# ------------------ 캡처 스레드 ------------------
class CaptureThread:
    def __init__(self, cam_index=0, backend=cv2.CAP_DSHOW):
        self.cap = cv2.VideoCapture(cam_index, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAP_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try: self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except: pass
        if not self.cap.isOpened(): raise RuntimeError("카메라 열기 실패")
        try: self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        except: pass
        try: self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        except: pass
        try: self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        except: pass
        self.lock = threading.Lock(); self.latest = None; self.running = True
        self.th = threading.Thread(target=self.loop, daemon=True); self.th.start()
    def loop(self):
        while self.running:
            for _ in range(2): self.cap.grab()
            ret, f = self.cap.retrieve()
            if ret:
                with self.lock:
                    self.latest = f
    def read(self):
        with self.lock:
            if self.latest is None: return False, None
            return True, self.latest.copy()
    def release(self):
        self.running = False
        self.th.join(timeout=0.5)
        self.cap.release()

# ------------------ 시리얼(로봇팔) ------------------
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
            if motor_cmds is None: break
            try:
                vals = [motor_cmds.get(f"motor_{i}", 0) for i in range(1,7)]
                ser.write((','.join(map(str, vals)) + '\n').encode('utf-8'))
            except Exception as e:
                print(f"[SerialWriteError] {e}")
    finally:
        ser.close(); print("시리얼 종료")

# ------------------ 얼굴 DNN ------------------
prototxt_path = r"C:\face_models\deploy.prototxt"
model_path   = r"C:\face_models\res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

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
    return boxes

# ------------------ 칼만 ------------------
def init_kalman():
    kf = cv2.KalmanFilter(4,2)
    kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kf.processNoiseCov   = np.diag([1e-2,1e-2,1e-1,1e-1]).astype(np.float32)
    kf.measurementNoiseCov = np.diag([2.0,2.0]).astype(np.float32)
    kf.errorCovPost = np.diag([10,10,10,10]).astype(np.float32)
    return kf

def kalman_predict(kf, dt):
    kf.transitionMatrix = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], np.float32)
    pred = kf.predict(); return int(pred[0,0]), int(pred[1,0])

def kalman_correct(kf, x, y):
    kf.correct(np.array([[np.float32(x)],[np.float32(y)]], np.float32))

# ================== PD 제어 ==================
prev_error_x = prev_error_y = prev_error_area = 0.0
last_control_time = time.time()
d_ex_ema = d_ey_ema = d_ea_ema = 0.0

def compute_motor_angles(center_x, center_y, area, frame_shape, desired_area=42000, kf=None, kalman_inited=False):
    global prev_error_x, prev_error_y, prev_error_area, last_control_time
    global d_ex_ema, d_ey_ema, d_ea_ema
    frame_h, frame_w = frame_shape[:2]
    error_x = (center_x - frame_w/2) / (frame_w/2)
    error_y = (center_y - frame_h/2) / (frame_h/2)
    error_area = (desired_area - area)
    if abs(error_x) < DEAD_X: error_x = 0.0
    if abs(error_y) < DEAD_Y: error_y = 0.0
    t=time.time(); dt=max(1e-3, t-last_control_time); last_control_time=t
    d_error_x=(error_x-prev_error_x)/dt; d_error_y=(error_y-prev_error_y)/dt; d_error_area=(error_area-prev_error_area)/dt
    d_ex_ema=(1-D_ALPHA)*d_ex_ema + D_ALPHA*d_error_x
    d_ey_ema=(1-D_ALPHA)*d_ey_ema + D_ALPHA*d_error_y
    d_ea_ema=(1-D_ALPHA)*d_ea_ema + D_ALPHA*d_error_area
    d_ex_ema=float(np.clip(d_ex_ema, -D_CLAMP, D_CLAMP))
    d_ey_ema=float(np.clip(d_ey_ema, -D_CLAMP, D_CLAMP))
    if kalman_inited and kf is not None:
        vx = kf.statePost[2,0] / (frame_w/2); vy = kf.statePost[3,0] / (frame_h/2)
        error_x += vx * LEAD_FACE_SEC; error_y += vy * LEAD_FACE_SEC
    Kp_xy = 90*0.35; Kd_xy = 90*0.25
    Kp_area = 0.0020; Kd_area = 0.0012
    out_x = Kp_xy*error_x + Kd_xy*d_ex_ema
    out_y = Kp_xy*error_y + Kd_xy*d_ey_ema
    out_a = Kp_area*error_area + Kd_area*d_ea_ema
    prev_error_x,prev_error_y,prev_error_area = error_x,error_y,error_area
    return {"motor_1": out_y,"motor_2": out_y,"motor_3": out_y*0.5,"motor_4": out_a,"motor_5": out_y*0.2,"motor_6": out_x*0.5}

def clip_motor_angles(cmds, limits=(-90,90)):
    return {k:int(np.clip(v, limits[0], limits[1])) for k,v in cmds.items()}

last_cmds = {f"motor_{i}": 0.0 for i in range(1,7)}
def apply_slew(cmds, max_delta=SLEW):
    global last_cmds
    out={}
    for k,v in cmds.items():
        dv=float(np.clip(v - last_cmds[k], -max_delta, max_delta))
        out[k]=last_cmds[k]+dv; last_cmds[k]=out[k]
    return out

# ------------------ 줌 스무더 ------------------
class ZoomSmooth:
    def __init__(self, z0=1.0, quant=0.02, hyst=0.06):
        self.z = float(z0)
        self.quant = float(quant)
        self.hyst = float(hyst)
        self.oe = OneEuro(min_cutoff=0.6, beta=0.03, dcutoff=1.0)
        self.t_prev = None
        self.z_hold = float(z0)
    def update(self, z_desired, t, slew_per_s=0.6):
        z_desired = float(np.clip(z_desired, ZOOM_MIN, ZOOM_MAX))
        if abs(z_desired - self.z_hold) < self.hyst:
            z_desired = self.z_hold
        else:
            self.z_hold = z_desired
        z_hat = self.oe.filter(z_desired, t)
        z_hat = round(z_hat / self.quant) * self.quant
        if self.t_prev is None:
            self.t_prev = t; self.z = z_hat; return self.z
        dt = max(1e-3, t - self.t_prev)
        dz_max = slew_per_s * dt
        dz = np.clip(z_hat - self.z, -dz_max, dz_max)
        self.z += dz; self.t_prev = t
        return float(self.z)

# ------------------ 메인 ------------------
def main():
    global icr3_phase, icr3_center, icr3_t0, icr3_inside, icr3_total
    global _prev_cx, _prev_cy, _prev_t, reacquire_t0

    # 스레드 준비
    q = queue.Queue()
    threading.Thread(target=serial_worker, args=(q,), daemon=True).start()
    cap_thread = CaptureThread()

    # 비디오 저장
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    recording, out = False, None

    # 상태
    kf = init_kalman(); kalman_inited = False
    last_kf_ts = time.time()
    zoom_smooth = ZoomSmooth(z0=1.0, quant=ZOOM_QUANT, hyst=ZOOM_HYST)

    # 표시 스무딩(센터)
    cx_oe = OneEuro(0.9, 0.04, 1.2)
    cy_oe = OneEuro(0.9, 0.05, 1.2)

    ever_locked = False
    LOG_INTERVAL, last_log = 0.3, 0.0

    # ----[연속촬영 & 오버레이 상태]----
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

    print("키: s/e=녹화 시작/종료, 1~9=연속촬영(장수), q=종료")
    try:
        frame_idx = 0
        frame_per_sec = 0
        frame_idx_per_sec = 0
        sum_time_per_sec = 0

        box_l=box_t=box_w=box_h=box_cx=box_cy=0
        pre_frame_time = 0
        ICR_RADIUS = 0.0

        while True:
            ok, frame = cap_thread.read()
            if not ok: continue
            now = time.time()
            
            sum_time_per_sec += (now-pre_frame_time)
            frame_idx_per_sec = frame_idx_per_sec+1
            if sum_time_per_sec > 1.0 :
                frame_per_sec = frame_idx_per_sec
                print(f"frame per sec : {frame_per_sec}")
                sum_time_per_sec = 0
                frame_idx_per_sec = 0

            #교수님께서 원하시는 해상도로 resize (카메라 해상도가 1920,1080이 안됨)
            #frame = cv2.resize(frame, (int(CAP_WIDTH*(1/(1-RATIO_TRANSLATE))), int(CAP_HEIGHT*(1/(1-RATIO_TRANSLATE)))))
            frame = cv2.flip(frame,1)
            frame_h, frame_w = frame.shape[:2]

            #지표3 spec Setting
            if ICR_RADIUS<= 0 :
                ICR_RADIUS = int( ((((frame_w/2)**2) + ((frame_h/2))**2)**0.5) * ICR_RATIO )

            frame_idx += 1
            do_detect = (frame_idx % DETECT_EVERY == 0)

            dt_kf = max(1e-3, now - last_kf_ts); last_kf_ts = now

            # 얼굴 검출
            if do_detect:
                face_boxes = detect_faces_dnn(frame)
                face_boxes_preFrame = face_boxes  
            else:
                face_boxes = face_boxes_preFrame
                #face_boxes = []

            face_found = len(face_boxes) > 0

            # --- 지표1: 재인식 시간 측정(미검출→검출 전환) ---
            if face_found and reacquire_t0 is not None:
                reacq = now - reacquire_t0
                metric1_times.append(reacq)
                reacquire_t0 = None
                #print(f"[지표1] 재인식 {reacq:.3f}s")
            elif not face_found and reacquire_t0 is None:
                reacquire_t0 = now

            if face_found:
                face_boxes.sort(key=lambda b: b[2]*b[3], reverse=True)
                box_l, box_t, box_w, box_h = face_boxes[0]
                box_cx, box_cy = box_l + box_w//2, box_t + box_h//2
                area = box_w*box_h
                ever_locked = True

                if not kalman_inited:
                    kf.statePost = np.array([[box_cx],[box_cy],[0],[0]], np.float32)
                    kalman_inited = True
                kpx, kpy = kalman_predict(kf, dt_kf)
                kalman_correct(kf, box_cx, box_cy)
                kpx, kpy = int(kf.statePost[0,0]), int(kf.statePost[1,0])
            else:
                if kalman_inited: kpx, kpy = kalman_predict(kf, dt_kf)
                else:             kpx, kpy = (frame_w//2, frame_h//2)
            
            # 중앙 고정 목표
            use_cx, use_cy = kpx, kpy
            if kalman_inited:
                use_cx += int(kf.statePost[2,0] * LEAD_FACE_SEC)
                use_cy += int(kf.statePost[3,0] * LEAD_FACE_SEC)
            
            ##-----------------------------------------------------------------
            # 로봇팔 제어가 여기 들어가야함...

            ##-----------------------------------------------------------------
            # 화면 표시용 스무딩
            disp_kf_cx = int(cx_oe.filter(use_cx, now))
            disp_kf_cy = int(cy_oe.filter(use_cy, now))
            disp_ori_cx = box_cx
            disp_ori_cy = box_cy

            #disp_kf_cx=use_cx=box_cx
            #disp_kf_cy=use_cy=box_cy

            # --- 줌 목표 --- 
            #if face_found and box_h > 0:
            #    desired_face_px = TARGET_FACE_FRAC * min(frame_w, frame_h)
            #    z_desired = float(np.clip(desired_face_px / float(box_h), ZOOM_MIN, ZOOM_MAX))
            #else:
            #    z_desired = 1.0
            #z_now = zoom_smooth.update(z_desired, now, slew_per_s=ZOOM_SLEW_PER_S)

            # --- 중앙 평행이동 + 크롭 ---
            diff_x = (frame_w/2)-disp_kf_cx
            diff_y = (frame_h/2)-disp_kf_cy
            min_x = -(frame_w*RATIO_TRANSLATE/2); max_x = frame_w*RATIO_TRANSLATE/2
            min_y = -(frame_h*RATIO_TRANSLATE/2); max_y = frame_h*RATIO_TRANSLATE/2
            fx = float(np.clip(diff_x, min_x, max_x))
            fy = float(np.clip(diff_y, min_y, max_y))

            display_w = int(frame_w * (1-RATIO_TRANSLATE))
            display_h = int(frame_h * (1-RATIO_TRANSLATE))
            crop_t = int(disp_kf_cy-(display_h/2)); crop_b = int(disp_kf_cy+(display_h/2))
            crop_l = int(disp_kf_cx-(display_w/2)); crop_r = int(disp_kf_cx+(display_w/2))
            if crop_t < 0: crop_t = 0; crop_b = crop_t + display_h
            elif crop_b >= frame_h-1: crop_b = frame_h-1; crop_t = crop_b-display_h
            if crop_l < 0: crop_l = 0; crop_r = crop_l + display_w
            elif crop_r >= frame_w-1: crop_r = frame_w-1; crop_l = crop_r-display_w
            
            shifted = frame[int(crop_t):int(crop_b), int(crop_l):int(crop_r)]
            #shifted = frame[int(frame_h*RATIO_TRANSLATE/2):int(frame_h*RATIO_TRANSLATE/2 + display_h), int(frame_w*RATIO_TRANSLATE/2):int(frame_w*RATIO_TRANSLATE/2 + display_w)]
            #임시로 보정 안한거 저장용
            #croped = frame[int(frame_h*RATIO_TRANSLATE/2):int(frame_h*RATIO_TRANSLATE/2 + display_h), int(frame_w*RATIO_TRANSLATE/2):int(frame_w*RATIO_TRANSLATE/2 + display_w)]
            
            disp_addapt_size_kf_cx = disp_kf_cx - crop_l
            disp_addapt_size_kf_cy = disp_kf_cy - crop_t
            disp_addapt_size_ori_cx = disp_ori_cx - crop_l
            disp_addapt_size_ori_cy = disp_ori_cy - crop_t
            
            out_frame = shifted

            # --- 저장/미리보기 ---
            display = out_frame.copy()

            # 가이드 박스 + 센터 점
            guide_w = box_w; guide_h = box_h
            gx1 = int(disp_addapt_size_ori_cx - (guide_w/2))
            gx2 = int(gx1+guide_w)
            gy1 = int(disp_addapt_size_ori_cy - (guide_h/2))
            gy2 = int(gy1+guide_h)
            gcx = disp_addapt_size_ori_cx#int(gx1+(guide_w/2)); 
            gcy = disp_addapt_size_ori_cy#int(gy1+(guide_h/2))
            gx1=max(3,gx1); 
            gy1=max(3,gy1)
            gx2=min(display.shape[1]-3,gx2); 
            gy2=min(display.shape[0]-3,gy2)
            if face_found:
                cv2.rectangle(display, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (0,200,0), 2)
                cv2.circle(display, (int(gcx), int(gcy)), 3, (0, 0, 255), -1)

            #################################################################################
            # --- 지표 용 속도 계산 ---
            if _prev_t is None: _prev_t = now
            dt = max(1e-3, now - _prev_t)
            #speed_px = est_speed_px_per_s(disp_kf_cx, disp_kf_cy, _prev_cx, _prev_cy, dt)
            speed_px = est_speed_px_per_s(gcx, gcy, _prev_cx, _prev_cy, dt)
            speed_cm = speed_px * CM_PER_PIXEL
            if _prev_cx is not None:
                # 지표2: 최근 STAB_WIN_SEC 동안 dist<=DT_THRESH_PX 비율
                #dist = ((disp_kf_cx - _prev_cx)**2 + (disp_kf_cy - _prev_cy)**2) ** 0.5
                dist = ((gcx - _prev_cx)**2 + (gcy - _prev_cy)**2) ** 0.5
                stab_buf.append((now, dist))
                while stab_buf and (now - stab_buf[0][0]) > STAB_WIN_SEC:
                    stab_buf.popleft()
                if stab_buf:
                    inside = sum(1 for (_, d) in stab_buf if d <= DT_THRESH_PX)
                    ratio = 100.0 * inside / len(stab_buf)
                    metric2_ratios.append(ratio)  # 계속 로그(원하면 샘플링 간헐화)
            _prev_cx, _prev_cy, _prev_t = gcx, gcy, now

            # 지표1 부가: 초속 기록
            metric1_speeds_px.append(speed_px)
            metric1_speeds_cm.append(speed_cm)

            # --- 지표3: ICR3 (정지구간 원내 비율) ---
            if speed_px < STOP_SPEED_THR:
                if icr3_phase == "move":
                    icr3_phase = "stop and collect start"
                    #icr3_center = (disp_kf_cx, disp_kf_cy)
                    icr3_center = (display_w//2, display_h//2)
                    icr3_t0 = now
                    icr3_inside = 0; icr3_total = 0
                    if len(metric3_ratios)>0:
                        matric3_text = f"[지표3] ICR3(원내 비율) = {metric3_ratios[len(metric3_ratios)-1]:.1f}%"
                    else:
                        matric3_text = f"[지표3] Data 없음 (원내 {ICR_RADIUS}px)"
                elif icr3_phase == "stop and collect start":
                    
                    #r = ((disp_kf_cx - icr3_center[0])**2 + (disp_kf_cy - icr3_center[1])**2)**0.5
                    r = ((gcx - icr3_center[0])**2 + (gcy - icr3_center[1])**2)**0.5
                    
                    matric3_text = f"[지표3] 움직이다 멈춤!! 수집중... ({STOP_HOLD_SEC}/{int(now - icr3_t0-STOP_HOLD_START)}s) -> spec({ICR_RADIUS}px 이하 (현 {r:.1f}px))"
                    if (now - icr3_t0) >= STOP_HOLD_START:
                        icr3_total += 1
                        if r <= ICR_RADIUS: 
                            icr3_inside += 1
                        if (now - icr3_t0) >= STOP_HOLD_SEC+STOP_HOLD_START:
                            ratio = 100.0 * icr3_inside / max(1, icr3_total)
                            metric3_ratios.append(ratio)
                            #print(f"[지표3] ICR3(원내 비율) = {ratio:.1f}%")
                            icr3_phase = "idle"

                        cv2.circle(display, (int(display_w//2)-1, int(display_h/2)-1), ICR_RADIUS, (255, 0, 0), 2)
                    
                else:
                    if len(metric3_ratios)>0:
                        matric3_text = f"[지표3] ICR3(원내 비율) = {metric3_ratios[len(metric3_ratios)-1]:.1f}%"
                    else:
                        matric3_text = f"[지표3] Data 없음 (원내 {ICR_RADIUS}px)"
            else:
                matric3_text = f"[지표3] 이동중!!"
                icr3_phase = "move"

            #################################################################################
            # 메세지 출력

            # ===== 오버레이: 지표,  =====
            # 좌상단: ZOOM, 속도
            #display = draw_text_kr(display, f"ZOOM x{z_now:.2f}", (10, 10), 26, 2)
            #print(f"[FACE] raw=({gcx},{gcy}) kalman=({kpx},{kpy}) size=({box_w}x{box_h}) speed≈{speed:.1f}cm/s")
            display = draw_text_kr(display, f"[FACE] offset from image center=({gcx-display.shape[1]//2},{gcy-display.shape[0]//2})", (10, display_h-140), 25, 2)
            #if len(metric1_speeds_px)>0 and len(metric1_speeds_cm)>0 and len(metric1_times)>0 :
            #    display = draw_text_kr(display, f"[지표1] 속도: {metric1_speeds_px[len(metric1_speeds_px)-1]:5.1f}px/s  ({metric1_speeds_cm[len(metric1_speeds_cm)-1]:4.1f}cm/s) 재인식 {metric1_times[len(metric1_times)-1]:.3f}s", (10, display_h-110), 25, 2)
            if  len(metric1_times)>0 :
                display = draw_text_kr(display, f"[지표1] 재인식 시간 : {metric1_times[len(metric1_times)-1]:.3f}s", (10, display_h-110), 25, 2)
            if len(metric2_ratios)>0:
                display = draw_text_kr(display, f"[지표2] 비율: {metric2_ratios[len(metric2_ratios)-1]:5.1f}%", (10, display_h-80), 25, 2)
            #if len(metric3_ratios)>0:
            display = draw_text_kr(display, matric3_text, (10, display_h-50), 25, 2)
            
            # 녹화중일 경우
            if recording and msg_lt_display==False :
                msg_lt_text, msg_lt_until = "녹화 중!", now + 500.0
                
            #if recording==False and photo_shooting==False:
            #    msg_lt_text = False
            #    msg_lt_until = 0

            # 상단 메시지들(시작/종료/연속촬영 안내)
            if now < msg_lt_until and msg_lt_text:
                msg_lt_display = True
                display = draw_text_kr(display, msg_lt_text, (10, 10), 28, 2)
            else:
                msg_lt_display = False
            if now < msg_rt_until and msg_rt_text:
                msg_rt_display = True
                (tw, th), _ = cv2.getTextSize(msg_rt_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                display = draw_text_kr(display, msg_rt_text, (display.shape[1]-10-int(tw*1.2), 10), 28, 2)
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
                    display = draw_text_kr(display, cd_text, (10, 100), 42, 3)
                remain = max(0, photo_count - photo_taken)

                if now >= next_shot_at:
                    filename = get_new_image_filename()
                    # ★ 사진 저장은 '원본 frame'으로 — 오버레이 없음 ★
                    cv2.imwrite(filename, frame)
                    photo_taken += 1
                    print(f"{photo_taken}/{photo_count} 저장됨: {os.path.basename(filename)}")

                    if photo_taken >= photo_count:
                        photo_shooting = False
                        next_shot_at = None
                        msg_lt_text, msg_lt_until = f"연속 사진 촬영 완료)", now + 1.0
                    else:
                        next_shot_at = now + photo_interval
                display = draw_text_kr(display, f"남은 장 수: {remain}", (display.shape[1]-220, 10), 28, 2)

            cv2.imshow("Center-Locked (No Distortion / No Wobble Zoom)", display)

            # 로그(속도)
            if face_found and now - last_log >= LOG_INTERVAL:
                vx_pix, vy_pix = float(kf.statePost[2,0]), float(kf.statePost[3,0])
                speed = (vx_pix**2 + vy_pix**2)**0.5 * CM_PER_PIXEL
                out_h, out_w = out_frame.shape[:2]
                #print(f"[FACE] raw=({gcx},{gcy}) kalman=({kpx},{kpy}) size=({box_w}x{box_h}) speed≈{speed:.1f}cm/s")
                #print(f"[image] framew,h=({frame_w},{frame_h}) displayw,h=({display_w},{display_h}) frame[{int(disp_kf_cy-(display_h/2))}:{int(disp_kf_cy+(display_h/2))} , {int(disp_kf_cx-(display_w/2))}:{int(disp_kf_cx+(display_w/2))}]")
                last_log = now

            # --- 로봇팔 제어 ---
            if not ever_locked:
                q.put({f"motor_{i}": 0 for i in range(1,7)})
            else:
                cmds = compute_motor_angles(use_cx, use_cy, area, frame.shape, kf=kf, kalman_inited=kalman_inited)
                #if now - last_log >= LOG_INTERVAL:
                #    print(f"[compute_motor_angles_in] use=({use_cx},{use_cy}) frame_c=({frame_w//2},{frame_h//2}) area={area}")
                #    print(f"[compute_motor_angles_out] {cmds}")
                #    last_log = now
                cmds = clip_motor_angles(cmds, (-90, 90))
                cmds = apply_slew(cmds, max_delta=SLEW)
                q.put(cmds)

            # --- 키 입력/녹화/연속촬영 ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            if key == ord('s') and not recording and not photo_shooting:
                output_path = get_new_filename()

                record_w = out_frame.shape[1] if RECORD_USE_STAB else frame_w
                record_h = out_frame.shape[0] if RECORD_USE_STAB else frame_h
                out = cv2.VideoWriter(output_path, fourcc, frame_per_sec, (record_w, record_h))
                if not out.isOpened():
                    #print("VideoWriter 열기 실패"); out = None
                    msg_lt_text, msg_lt_until = f"VideoWriter 열기 실패", now + 1.0
                    out = None
                else:
                    recording = True
                    #print(f"녹화 시작: {os.path.basename(output_path)}")
                    msg_lt_text, msg_lt_until = f"녹화 시작: {os.path.basename(output_path)}", now + 1.0
                    msg_lt_display = True

            if key == ord('e') and recording:
                recording = False
                if out is not None: out.release(); out = None
                print("녹화 종료! 저장 완료")
                msg_lt_text, msg_lt_until = "녹화 종료!", now + 1.0
                msg_lt_display = True

            # 녹화 프레임 쓰기
            if recording and out is not None:
                clean = out_frame if RECORD_USE_STAB else frame
                #clean = croped if RECORD_USE_STAB else frame
                #clean = cv2.flip(clean,1)
                out.write(clean)

            # 연속촬영 시작 (1~9)
            if (ord('1') <= key <= ord('9')) and not photo_shooting:
                photo_count = key - ord('0')
                photo_taken = 0
                photo_shooting = True
                next_shot_at = now + photo_interval  # 3초 뒤 촬영
                #print()
                msg_lt_text, msg_lt_until = f"{photo_count}장의 사진 연속 촬영 시작! (간격 {photo_interval:.0f}초)", now + 500
                
            
            pre_frame_time = now

    finally:
        try:
            if out is not None: out.release()
        except: pass
        cap_thread.release()
        cv2.destroyAllWindows()
        q.put(None)

        # ====== 지표 요약 출력 ======
        if len(metric1_times)>0:
            arr=np.array(metric1_times)
            print(f"[지표1] 재인식 mean={arr.mean():.3f}s median={np.median(arr):.3f}s max={arr.max():.3f}s")
        else:
            print("[지표1] 샘플 없음")

        if len(metric1_speeds_px)>0:
            ap=np.array(metric1_speeds_px); ac=np.array(metric1_speeds_cm)
            print(f"[지표1-속도] px/s mean={ap.mean():.1f} median={np.median(ap):.1f} max={ap.max():.1f}")
            print(f"[지표1-속도] cm/s mean={ac.mean():.1f} median={np.median(ac):.1f} max={ac.max():.1f}")
        else:
            print("[지표1-속도] 샘플 없음")

        if len(metric2_ratios)>0:
            arr=np.array(metric2_ratios)
            print(f"[지표2] 이동안정 mean={arr.mean():.1f}% median={np.median(arr):.1f}% min={arr.min():.1f}%")
        else:
            print("[지표2] 샘플 없음")

        if len(metric3_ratios)>0:
            arr=np.array(metric3_ratios)
            print(f"[지표3] ICR3 mean={arr.mean():.1f}% median={np.median(arr):.1f}% min={arr.min():.1f}%")
        else:
            print("[지표3] 샘플 없음")

if __name__ == "__main__":
    main()
