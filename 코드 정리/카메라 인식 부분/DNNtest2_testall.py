import cv2
import numpy as np
import threading
import queue
import serial
import os
import re
import time
from collections import deque

# ================== 기본 ==================
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
CAP_WIDTH, CAP_HEIGHT, CAP_FPS = 1920, 1080, 60

DETECT_EVERY = 2
DEAD_X, DEAD_Y = 0.02, 0.02
D_ALPHA, D_CLAMP = 0.25, 5.0
SLEW = 8.0
CM_PER_PIXEL = 0.050
LEAD_FACE_SEC = 0.12

# === 무왜곡 중앙 고정 & 줌 ===
TARGET_FACE_FRAC = 0.26
ZOOM_MIN, ZOOM_MAX = 1.00, 1.70          # 과업스케일 억제
ZOOM_SLEW_PER_S = 0.55                   # 줌 변화 한계(초당)
ZOOM_QUANT = 0.015                       # 줌 양자화(미세 떨림 억제)
ZOOM_HYST  = 0.06                        # 줌 히스테리시스
SAFETY_PX = 8

# === 엣지 안정화(소프트 클램프/히스테리시스) ===
EDGE_SOFT_PX = 90                         # 경계 완충폭
EDGE_Z_HYST  = 0.08                       # 에지 제한 히스테리시스

# === 센터 이동 제한 ===
CENTER_SLEW_PX_S = 900                    # 초당 픽셀 이동 한계
CENTER_ACCEL_PX_S2 = 7000                 # 초당^2 가속 한계

# ================== 파일 이름 ==================
def get_new_filename(base_name="output", ext="avi"):
    existing_files = os.listdir(desktop_path)
    pattern = re.compile(rf"{re.escape(base_name)}_(\d+)\.{re.escape(ext)}")
    nums = [int(m.group(1)) for f in existing_files if (m := pattern.match(f))]
    n = max(nums, default=0) + 1
    return os.path.join(desktop_path, f"{base_name}_{n}.{ext}")

# ================== 텍스트 ==================
def draw_text(img, text, org, font_scale=0.8, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness, cv2.LINE_AA)

# ================== 시리얼 ==================
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

# ================== DNN 얼굴 ==================
prototxt_path = r"C:\face_models\deploy.prototxt"
model_path   = r"C:\face_models\res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

def detect_faces_dnn(frame, conf_thresh=0.5):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104.0,177.0,123.0))
    net.setInput(blob); det = net.forward()
    boxes = []
    for i in range(det.shape[2]):
        conf = float(det[0,0,i,2])
        if conf > conf_thresh:
            x1,y1,x2,y2 = (det[0,0,i,3:7] * np.array([w,h,w,h])).astype(int)
            x1,y1,x2,y2 = max(0,x1),max(0,y1),min(w-1,x2),min(h-1,y2)
            boxes.append((x1,y1,x2-x1,y2-y1))
    return boxes

# ================== OneEuro ==================
class OneEuro:
    def __init__(self, min_cutoff=0.9, beta=0.05, dcutoff=1.2):
        self.min_cutoff=float(min_cutoff); self.beta=float(beta); self.dcutoff=float(dcutoff)
        self.x_prev=None; self.dx_prev=0.0; self.t_prev=None
    @staticmethod
    def alpha(cutoff, dt):
        tau = 1.0/(2.0*np.pi*cutoff)
        return 1.0/(1.0 + tau/dt)
    def filter(self, x, t):
        if self.t_prev is None:
            self.t_prev=t; self.x_prev=float(x); return float(x)
        dt=max(1e-3, t-self.t_prev)
        dx=(x-self.x_prev)/dt
        a_d=OneEuro.alpha(self.dcutoff, dt)
        dx_hat=a_d*dx + (1-a_d)*self.dx_prev
        cutoff=self.min_cutoff + self.beta*abs(dx_hat)
        a=OneEuro.alpha(cutoff, dt)
        x_hat=a*x + (1-a)*self.x_prev
        self.t_prev=t; self.x_prev=x_hat; self.dx_prev=dx_hat
        return float(x_hat)

# ================== 줌 스무드 ==================
class ZoomSmooth:
    def __init__(self, z0=1.0, quant=0.02, hyst=0.06):
        self.z=float(z0); self.quant=float(quant); self.hyst=float(hyst)
        self.oe=OneEuro(min_cutoff=0.6, beta=0.03, dcutoff=1.0)
        self.t_prev=None; self.z_hold=float(z0)
    def update(self, z_desired, t, slew_per_s=0.6):
        # 히스테리시스
        if abs(z_desired - self.z_hold) < self.hyst:
            z_desired = self.z_hold
        else:
            self.z_hold = z_desired
        z_hat = self.oe.filter(z_desired, t)
        z_hat = round(z_hat / ZOOM_QUANT) * ZOOM_QUANT
        if self.t_prev is None:
            self.t_prev=t; self.z=z_hat; return self.z
        dt=max(1e-3, t-self.t_prev)
        dz_max = slew_per_s * dt
        dz = np.clip(z_hat - self.z, -dz_max, dz_max)
        self.z += dz; self.t_prev=t
        return float(self.z)

# ================== Kalman 2D ==================
def init_kalman():
    kf=cv2.KalmanFilter(4,2)
    kf.measurementMatrix=np.array([[1,0,0,0],[0,1,0,0]],np.float32)
    kf.processNoiseCov=np.diag([1e-2,1e-2,1e-1,1e-1]).astype(np.float32)
    kf.measurementNoiseCov=np.diag([2.0,2.0]).astype(np.float32)
    kf.errorCovPost=np.diag([10,10,10,10]).astype(np.float32)
    return kf

def kalman_predict(kf, dt):
    kf.transitionMatrix=np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]],np.float32)
    pred=kf.predict()
    return int(pred[0,0]), int(pred[1,0])

def kalman_correct(kf,x,y):
    kf.correct(np.array([[np.float32(x)],[np.float32(y)]],np.float32))

# ================== PD 제어 ==================
prev_error_x = prev_error_y = prev_error_area = 0.0
last_control_time = time.time()
d_ex_ema = d_ey_ema = d_ea_ema = 0.0

def compute_motor_angles(center_x, center_y, area, frame_shape, desired_area=42000, kf=None, kalman_inited=False):
    global prev_error_x, prev_error_y, prev_error_area, last_control_time
    global d_ex_ema, d_ey_ema, d_ea_ema
    h, w = frame_shape[:2]
    error_x = (center_x - w/2) / (w/2)
    error_y = (center_y - h/2) / (h/2)
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
        vx = kf.statePost[2,0] / (w/2); vy = kf.statePost[3,0] / (h/2)
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

# ================== FSM ==================
hist = deque(maxlen=12)
mode = "TRACKING"; lost_since=None; scan_phase=0.0
history_len=10; predict_horizon=0.4; damping=0.85
lost_predict_max=0.5; lost_hold_max=0.5; scan_period=1.2; scan_amp_px=80

def push_history(x,y,t): hist.append((float(x),float(y),float(t)))
def est_velocity():
    if len(hist)<2: return np.array([0.0,0.0])
    x0,y0,t0 = hist[max(0,len(hist)-history_len)]
    x1,y1,t1 = hist[-1]; dt=max(1e-3,t1-t0)
    return np.array([(x1-x0)/dt,(y1-y0)/dt])

def update_tracker(face_found,cx,cy,now,frame_shape,fps_hint=30.0):
    global mode,lost_since,scan_phase
    h,w=frame_shape[:2]
    if face_found:
        push_history(cx,cy,now); mode="TRACKING"; lost_since=None; return (cx,cy),mode
    if lost_since is None: lost_since=now
    lost_dt=now-lost_since
    if lost_dt<=lost_predict_max and len(hist)>=2:
        v=est_velocity(); pred_dt=min(predict_horizon,lost_dt)
        last_x,last_y,_=hist[-1]; pred=np.array([last_x,last_y])+v*pred_dt
        pred=damping*pred + (1-damping)*np.array([w//2,h//2])
        pred=np.clip(pred,[0,0],[w-1,h-1]); mode="PREDICT"
        return (int(pred[0]),int(pred[1])),mode
    if lost_dt<= (lost_predict_max+lost_hold_max):
        last_x,last_y,_ = hist[-1] if len(hist) else (w//2,h//2,now)
        mode="HOLD"; return (int(last_x),int(last_y)),mode
    mode="SCAN"
    scan_phase=(scan_phase + 2*np.pi*(1.0/scan_period)*(1.0/fps_hint))%(2*np.pi)
    amp_y=max(6,int(0.015*min(w,h)))
    sxp=(w//2)+int(scan_amp_px*np.sin(scan_phase))
    syp=(h//2)+int(amp_y*np.sin(scan_phase*0.5))
    return (sxp,syp),mode

# ================== 유틸 ==================
def clamp(v, lo, hi): return lo if v<lo else hi if v>hi else v

def soft_clamp(v, lo, hi, soft):
    if v < lo + soft:
        t = clamp((v - lo)/max(1e-6, soft), 0.0, 1.0)
        t = t*t*(3 - 2*t)  # smoothstep
        return lo + t*soft
    if v > hi - soft:
        t = clamp((hi - v)/max(1e-6, soft), 0.0, 1.0)
        t = t*t*(3 - 2*t)
        return hi - t*soft
    return v

class MotionLimiter:
    def __init__(self, v_max_px_s, a_max_px_s2):
        self.vx=0.0; self.vy=0.0; self.prev_t=None
        self.v_max=v_max_px_s; self.a_max=a_max_px_s2
        self.x=None; self.y=None
    def step(self, x_des, y_des, t):
        if self.prev_t is None:
            self.prev_t=t; self.x=float(x_des); self.y=float(y_des); self.vx=0.0; self.vy=0.0
            return self.x, self.y
        dt=max(1e-3, t-self.prev_t); self.prev_t=t
        ax = clamp((x_des - self.x)/dt - self.vx, -self.a_max*dt, self.a_max*dt)
        ay = clamp((y_des - self.y)/dt - self.vy, -self.a_max*dt, self.a_max*dt)
        self.vx = clamp(self.vx + ax, -self.v_max*dt, self.v_max*dt)
        self.vy = clamp(self.vy + ay, -self.v_max*dt, self.v_max*dt)
        self.x += self.vx; self.y += self.vy
        return self.x, self.y

# ================== 캡처/상태 ==================
q = queue.Queue()
threading.Thread(target=serial_worker, args=(q,), daemon=True).start()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
cap.set(cv2.CAP_PROP_FPS,          CAP_FPS)
if not cap.isOpened():
    print("카메라 열기 실패"); q.put(None); raise SystemExit

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
recording=False; out=None

LOG_INTERVAL=0.3; last_face_log_ts=0.0
metric1_times=[]; metric2_ratios=[]; metric3_ratios=[]
_face_prev=False; _face_lost_t=None
metric2_active=False; metric2_t0=0.0; metric2_frames_total=0; metric2_frames_le10=0; _metric2_prev_center=None
metric3_collecting=False; metric3_t0=0.0; metric3_centers=[]
kf=init_kalman(); kalman_inited=False; last_kf_ts=time.time()
fps_hint=30.0; fps_ema=None; last_t=time.time()
cx_oe=OneEuro(0.9,0.04,1.2); cy_oe=OneEuro(0.9,0.05,1.2)
zoom_smooth=ZoomSmooth(z0=1.0, quant=ZOOM_QUANT, hyst=ZOOM_HYST)
center_limiter = MotionLimiter(CENTER_SLEW_PX_S, CENTER_ACCEL_PX_S2)
ever_locked=False

def guide_box(w,h):
    gw=int(0.32*w); gh=int(gw*1.15)
    gx1=w//2-gw//2; gy1=h//2-gh//2
    return gx1,gy1,gx1+gw,gy1+gh

print("키: s/e=녹화 시작/종료, g=지표2(3초) 시작, q=종료")

try:
    last_center_used=None; frame_idx=0
    edge_z_hold = 1.0

    while True:
        ret, frame = cap.read()
        if not ret: print("프레임 캡처 실패"); break

        now=time.time(); h,w=frame.shape[:2]
        inst_fps=1.0/max(1e-3, now-last_t); last_t=now
        fps_ema = inst_fps if fps_ema is None else 0.9*fps_ema+0.1*inst_fps
        fps_hint=float(np.clip(fps_ema, 10, 120))

        diag=np.hypot(w,h); moving_thresh_px=0.008*diag

        frame_idx+=1; do_detect=(frame_idx%DETECT_EVERY==0)
        dt=max(1e-3, now-last_kf_ts); last_kf_ts=now

        # 얼굴 검출
        face_boxes = detect_faces_dnn(frame) if do_detect else []
        face_found = len(face_boxes)>0 # face box의 갯수가 1개 이상일 경우

        if face_found:
            face_boxes.sort(key=lambda b:b[2]*b[3], reverse=True)
            x,y,bw,bh=face_boxes[0] # left top width height
            cx,cy = x+bw//2, y+bh//2 # center position
            area=bw*bh 
            if not kalman_inited:
                kf.statePost=np.array([[cx],[cy],[0],[0]],np.float32); kalman_inited=True
            kpx,kpy=kalman_predict(kf,dt); kalman_correct(kf,cx,cy)
            kpx,kpy=int(kf.statePost[0,0]), int(kf.statePost[1,0])
            ever_locked=True
            if now-last_face_log_ts>=LOG_INTERVAL:
                vx_pix=float(kf.statePost[2,0]); vy_pix=float(kf.statePost[3,0])
                speed_cm_s=((vx_pix**2+vy_pix**2)**0.5)*CM_PER_PIXEL
                print(f"[FACE] raw=({cx},{cy}) kal=({kpx},{kpy}) size=({bw}x{bh}) v≈{speed_cm_s:.1f}cm/s")
                last_face_log_ts=now
        else:
            x=y=bw=bh=0; area=42000
            if kalman_inited: kpx,kpy=kalman_predict(kf,dt)
            else: kpx,kpy=(w//2,h//2)

        # 목표 센터(선행)
        use_cx,use_cy=kpx,kpy
        if kalman_inited:
            use_cx += int(kf.statePost[2,0]*LEAD_FACE_SEC)
            use_cy += int(kf.statePost[3,0]*LEAD_FACE_SEC)

        # 부드러운 센터 + 속도/가속 제한
        filt_cx=int(cx_oe.filter(use_cx, now))
        filt_cy=int(cy_oe.filter(use_cy, now))
        lim_cx, lim_cy = center_limiter.step(filt_cx, filt_cy, now)

        # 엣지 안전한 목표 줌
        if face_found and bh>0:
            desired_face_px = TARGET_FACE_FRAC * min(w,h)
            z_face = float(np.clip(desired_face_px/float(bh), ZOOM_MIN, ZOOM_MAX))
        else:
            z_face = 1.0

        # 현재 센터에서 엣지 안전 줌 한계 계산(하드)
        eps=1e-3
        left_space   = max(lim_cx - SAFETY_PX, eps)
        right_space  = max((w-1 - lim_cx) - SAFETY_PX, eps)
        top_space    = max(lim_cy - SAFETY_PX, eps)
        bottom_space = max((h-1 - lim_cy) - SAFETY_PX, eps)
        max_z_left   = (w/2.0)/left_space
        max_z_right  = (w/2.0)/right_space
        max_z_top    = (h/2.0)/top_space
        max_z_bottom = (h/2.0)/bottom_space
        edge_limit_hard = max(1.0, min(max_z_left, max_z_right, max_z_top, max_z_bottom, ZOOM_MAX))

        # 소프트 히스테리시스 적용된 엣지 제한
        if abs(edge_limit_hard - edge_z_hold) > EDGE_Z_HYST:
            edge_z_hold = edge_limit_hard
        edge_limit_soft = edge_z_hold

        z_target = float(np.clip(min(z_face, edge_limit_soft), ZOOM_MIN, ZOOM_MAX))
        z_now = zoom_smooth.update(z_target, now, slew_per_s=ZOOM_SLEW_PER_S)

        # 현재 줌에서 허용되는 센터 영역 계산
        half_w = (w/2.0)/z_now; half_h=(h/2.0)/z_now
        min_x = half_w + SAFETY_PX; max_x = w - half_w - SAFETY_PX
        min_y = half_h + SAFETY_PX; max_y = h - half_h - SAFETY_PX

        # 소프트 클램프(가장자리 울렁임 제거)
        scx = soft_clamp(lim_cx, min_x, max_x, EDGE_SOFT_PX)
        scy = soft_clamp(lim_cy, min_y, max_y, EDGE_SOFT_PX)

        # 크롭 ROI 계산(무왜곡)
        x1 = int(round(scx - half_w)); y1 = int(round(scy - half_h))
        x2 = int(round(scx + half_w)); y2 = int(round(scy + half_h))
        x1 = clamp(x1, 0, w-2); y1 = clamp(y1, 0, h-2)
        x2 = clamp(x2, x1+2, w); y2 = clamp(y2, y1+2, h)

        roi = frame[y1:y2, x1:x2]
        # 업스케일 시 CUBIC, 다운스케일 시 AREA
        interp = cv2.INTER_AREA if (roi.shape[1] >= w and roi.shape[0] >= h) else cv2.INTER_CUBIC
        out_frame = cv2.resize(roi, (w, h), interpolation=interp)

        # 이동 판정(정량지표용)
        is_moving=False
        if last_center_used is not None:
            distm=np.hypot(use_cx-last_center_used[0], use_cy-last_center_used[1])
            is_moving = distm > moving_thresh_px
        last_center_used=(use_cx,use_cy)

        # 프리뷰(중앙 박스만 고정)
        display=out_frame.copy()
        gx1,gy1,gx2,gy2 = guide_box(w,h)
        cv2.rectangle(display,(gx1,gy1),(gx2,gy2),(0,200,0),2)
        draw_text(display, f"ZOOM x{z_now:.2f} | MODE: {mode}", (10,30), 0.7, 2)
        cv2.imshow("Center-Locked (Edge-Stable, No Distortion)", display)

        # 제어
        if not ever_locked:
            q.put({f"motor_{i}":0 for i in range(1,7)})
        else:
            if not face_found:
                if lost_since is None: lost_since=now
                lost_dt = now - lost_since
            else:
                lost_since=None; lost_dt=0.0
            gain_scale=1.0; freeze_area=False
            if not face_found:
                if lost_dt<=0.5: gain_scale=0.6; freeze_area=True
                elif lost_dt<=1.0: gain_scale=0.4; freeze_area=True
                else: gain_scale=0.25; freeze_area=True
            cmds=compute_motor_angles(use_cx,use_cy,area,frame.shape,kf=kf,kalman_inited=kalman_inited)
            if freeze_area: cmds["motor_4"]=0
            for k in ["motor_1","motor_2","motor_3","motor_5","motor_6"]:
                cmds[k]*=gain_scale
            cmds=clip_motor_angles(cmds,(-90,90)); cmds=apply_slew(cmds,max_delta=SLEW)
            q.put(cmds)

        # 지표 (재인식/이동안정/정지ICR3)
        if face_found and (not _face_prev):
            if _face_lost_t is not None:
                t_reacq=now-_face_lost_t; metric1_times.append(t_reacq)
                print(f"[지표1] 재인식 시간 = {t_reacq:.3f}s (≤2.0s 목표)")
            _face_lost_t=None
        if (not face_found) and _face_prev: _face_lost_t=now
        _face_prev=face_found

        if metric2_active:
            if _metric2_prev_center is not None:
                dx=use_cx - _metric2_prev_center[0]; dy=use_cy - _metric2_prev_center[1]
                dpx=float(np.hypot(dx,dy)); metric2_frames_total+=1
                if dpx<=10.0: metric2_frames_le10+=1
            _metric2_prev_center=(use_cx,use_cy)
            if now-metric2_t0>=3.0:
                ratio=(metric2_frames_le10/metric2_frames_total*100.0) if metric2_frames_total>0 else 0.0
                metric2_ratios.append(ratio)
                print(f"[지표2] 이동중 |Δ|≤10px 비율 = {ratio:.1f}% (≥85% 목표)")
                metric2_active=False; metric2_frames_total=0; metric2_frames_le10=0; _metric2_prev_center=None
        else:
            _metric2_prev_center=(use_cx,use_cy)

        if not is_moving:
            if not metric3_collecting:
                metric3_collecting=True; metric3_t0=now; metric3_centers=[]
            metric3_centers.append((use_cx,use_cy))
            if now-metric3_t0>=3.0:
                if len(metric3_centers)>=2:
                    cx0,cy0=metric3_centers[0]; R=0.015*w
                    inside=[(np.hypot(cx-cx0, cy-cy0)<=R) for (cx,cy) in metric3_centers]
                    icr3=np.sum(inside)/len(metric3_centers)*100.0
                    metric3_ratios.append(icr3)
                    print(f"[지표3] 정지 후 ICR3 = {icr3:.1f}% (≥80% 목표) (R={R:.1f}px)")
                metric3_collecting=False; metric3_centers=[]
        else:
            metric3_collecting=False; metric3_centers=[]

        # 키/녹화
        key=cv2.waitKey(1)&0xFF
        if key==ord('q'): break
        if key==ord('s') and not recording:
            path=get_new_filename(); out=cv2.VideoWriter(path, fourcc, 20.0, (w,h))
            if not out.isOpened(): print("VideoWriter 열기 실패"); out=None
            else: recording=True; print(f"녹화 시작: {os.path.basename(path)}")
        if key==ord('e') and recording:
            recording=False
            if out is not None: out.release(); out=None
            print("녹화 종료! 저장 완료")
        if key==ord('g') and (not metric2_active):
            metric2_active=True; metric2_t0=now; metric2_frames_total=0; metric2_frames_le10=0
            _metric2_prev_center=(use_cx,use_cy)
            print("[지표2] 3초 측정 시작: 한 방향으로 30cm를 3초(10cm/s)에 이동!")

        if recording and out is not None:
            out.write(out_frame)

finally:
    if recording and out is not None: out.release()
    cap.release(); cv2.destroyAllWindows(); q.put(None)
    if len(metric1_times)>0:
        arr=np.array(metric1_times)
        print(f"[지표1] 재인식 mean={arr.mean():.3f}s median={np.median(arr):.3f}s max={arr.max():.3f}s")
    else:
        print("[지표1] 샘플 없음")
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
