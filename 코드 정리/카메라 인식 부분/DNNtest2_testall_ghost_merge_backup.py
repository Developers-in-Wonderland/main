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
# 중앙 고정 + 무왜곡(균일 스케일) 줌 + 평행이동 안정화 (EIS 제거판)
# - 왜곡 없음: 회전/스케일 보정 금지, 오직 translation + uniform zoom
# - 줌 울렁임 제거: 히스테리시스 + 양자화 + OneEuro + 슬루 제한
# - 얼굴은 항상 화면 정중앙 (화면이 따라감)
# - 저지연 캡처 스레드(720p@60fps, MJPG, 버퍼 드롭)
# - DNN(SSD ResNet10) + 칼만(x,y,vx,vy) + 예측 lead
# - 저장본/미리보기 모두 미러 없음
# ============================================================

# ------------------ 기본/저장 ------------------
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
CAP_WIDTH, CAP_HEIGHT, CAP_FPS = 1920, 1080, 60
RECORD_USE_STAB = True  # 저장본에도 중앙고정/줌 적용 (False면 원본 저장)

# ------------------ 검출/추적 ------------------
DETECT_EVERY = 2
LEAD_FACE_SEC = 0.12  # 칼만 속도 선행
CM_PER_PIXEL = 0.050  # 로그용(대충)

# ------------------ 제어(로봇팔) ------------------
DEAD_X, DEAD_Y = 0.02, 0.02
D_ALPHA, D_CLAMP = 0.25, 5.0
SLEW = 8.0
CM_PER_PIXEL = 0.050
LEAD_FACE_SEC = 0.12

# ------------------ 중앙고정 & 줌 ------------------
TARGET_FACE_FRAC = 0.26   # 얼굴이 화면의 이 비율에 오도록 줌 (min(w,h) 기준)
ZOOM_MIN, ZOOM_MAX = 1.00, 1.80
ZOOM_SLEW_PER_S = 0.60    # 초당 줌 변화 한계(더 낮추면 울렁임 더 줄어듦)
ZOOM_QUANT = 0.02         # 줌 양자화(계단)로 미세 떨림 제거
ZOOM_HYST = 0.06          # 히스테리시스 밴드(±6%)

BASE_MARGIN_FRAC = 0.10    # 기본 여백 (초기 안정)
SAFETY_PX = 6              # 크롭 경계 여유

# ------------------ 도우미 ------------------
def draw_text(img, text, org, font_scale=0.8, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255,255,255), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0,0,0), thickness, cv2.LINE_AA)

def get_new_filename(base_name="output", ext="avi"):
    existing = os.listdir(desktop_path)
    pat = re.compile(rf"{re.escape(base_name)}_(\d+)\.{re.escape(ext)}")
    nums = [int(m.group(1)) for f in existing if (m := pat.match(f))]
    n = max(nums, default=0) + 1
    return os.path.join(desktop_path, f"{base_name}_{n}.{ext}")

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
        # 오토 기능은 장치에 따라 조절
        try: self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1);  # 1=manual/const on MSMF
        except: pass
        try: self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        except: pass
        try: self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        except: pass
        self.lock = threading.Lock(); self.latest = None; self.running = True
        self.th = threading.Thread(target=self.loop, daemon=True); self.th.start()
    def loop(self):
        while self.running:
            for _ in range(2): self.cap.grab()  # 오래된 프레임 드랍
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
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104.0,177.0,123.0))
    net.setInput(blob)
    det = net.forward()
    boxes = []
    for i in range(det.shape[2]):
        conf = float(det[0,0,i,2])
        if conf > conf_thresh:
            x1,y1,x2,y2 = (det[0,0,i,3:7] * np.array([w,h,w,h])).astype(int)
            x1,y1,x2,y2 = max(0,x1),max(0,y1),min(w-1,x2),min(h-1,y2)
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

# ------------------ 줌 스무더(히스테리시스/양자화/슬루) ------------------
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
        # 히스테리시스: 이전 hold와의 차가 밴드 내면 유지
        if abs(z_desired - self.z_hold) < self.hyst:
            z_desired = self.z_hold
        else:
            self.z_hold = z_desired
        # OneEuro로 스무딩
        z_hat = self.oe.filter(z_desired, t)
        # 양자화로 미세 떨림 제거
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

    print("키: s/e=녹화 시작/종료, q=종료")
    try:
        frame_idx = 0
        while True:
            ok, frame = cap_thread.read()
            if not ok: continue
            now = time.time()
            h, w = frame.shape[:2]

            frame_idx += 1
            do_detect = (frame_idx % DETECT_EVERY == 0)

            dt = max(1e-3, now - last_kf_ts); last_kf_ts = now

            # 얼굴 검출
            face_boxes = detect_faces_dnn(frame) if do_detect else []
            face_found = len(face_boxes) > 0

            if face_found:
                face_boxes.sort(key=lambda b: b[2]*b[3], reverse=True)
                box_l, box_t, box_w, box_h = face_boxes[0]
                box_cx, box_cy = x + box_w//2, y + box_h//2
                area = box_w*box_h
                ever_locked = True

                if not kalman_inited:
                    kf.statePost = np.array([[box_cx],[box_cy],[0],[0]], np.float32)
                    kalman_inited = True
                kpx, kpy = kalman_predict(kf, dt)
                kalman_correct(kf, box_cx, box_cy)
                kpx, kpy = int(kf.statePost[0,0]), int(kf.statePost[1,0])

                # 로그(속도)
                if now - last_log >= LOG_INTERVAL:
                    vx_pix, vy_pix = float(kf.statePost[2,0]), float(kf.statePost[3,0])
                    speed = (vx_pix**2 + vy_pix**2)**0.5 * CM_PER_PIXEL
                    print(f"[FACE] raw=({box_cx},{box_cy}) kalman=({kpx},{kpy}) size=({box_w}x{box_h}) speed≈{speed:.1f}cm/s")
                    last_log = now
            else:
                x=y=box_w=box_h=0; area=42000
                if kalman_inited: kpx, kpy = kalman_predict(kf, dt)
                else:             kpx, kpy = (w//2, h//2)

            # --- 중앙 고정용 목표 센터(칼만 + 선행) ---
            use_cx, use_cy = kpx, kpy
            # 선행(리드)
            if kalman_inited:
                use_cx += int(kf.statePost[2,0] * LEAD_FACE_SEC)
                use_cy += int(kf.statePost[3,0] * LEAD_FACE_SEC)

            # 화면 표시용 스무딩(깜빡임 최소화)
            disp_cx = int(cx_oe.filter(use_cx, now))
            disp_cy = int(cy_oe.filter(use_cy, now))

            # --- 줌 목표 계산(균일 스케일) ---
            # 얼굴 높이가 화면에 차지할 목표 비율에 맞추기
            if face_found and box_h > 0:
                desired_face_px = TARGET_FACE_FRAC * min(w, h)
                z_desired = float(np.clip(desired_face_px / float(box_h), ZOOM_MIN, ZOOM_MAX))
            else:
                # 얼굴 없으면 천천히 1.0으로
                z_desired = 1.0

            z_now = zoom_smooth.update(z_desired, now, slew_per_s=ZOOM_SLEW_PER_S)

            # --- 순수 평행이동으로 중앙에 맞춘 뒤, 균일 줌(=대칭 크롭) ---
            # 1) 중앙이 되도록 평행이동
            fx = float(np.clip((w/2) - disp_cx, -(w*0.5), (w*0.5)))
            fy = float(np.clip((h/2) - disp_cy, -(h*0.5), (h*0.5)))
            T = np.array([[1,0,fx],[0,1,fy]], np.float32)

            #shifted = frame
			#shifted = cv2.zeros(h, w)
            shifted = cv2.warpAffine(frame, T, (w, h), flags=cv2.INTER_LINEAR)#,
                                     #borderMode=cv2.BORDER_REFLECT101)

            # 2) 균일 줌 = 대칭 크롭 → 리사이즈 (Aspect 유지, 왜곡 0)
            #    margin = w*(1-1/z)/2  (가로/세로 동일 margin)
            margin = int(np.clip(w*(1.0 - 1.0/float(z_now))/2.0, 0, min(w,h)//2 - SAFETY_PX))
            x1 = margin; y1 = margin
            x2 = w - margin; y2 = h - margin
            out_frame = cv2.resize(shifted[y1:y2, x1:x2], (w, h), interpolation=cv2.INTER_LINEAR)

            # --- 저장/미리보기 --- 
            display = out_frame.copy()
            
            # 미리보기 거울 모드
            display = cv2.flip(display,1)
            
            # 중앙 고정 가이드(움직이지 않는 박스)
            guide_w = int(0.32 * w)
            guide_h = int(guide_w * 1.15)  # 대충 얼굴 비율
            gx1 = w//2 - guide_w//2; gy1 = h//2 - guide_h//2
            gx2 = gx1 + guide_w;     gy2 = gy1 + guide_h
            cv2.rectangle(display, (gx1, gy1), (gx2, gy2), (0,200,0), 2)
            draw_text(display, f"ZOOMx={z_now:.2f}", (10, 30), 0.7, 2)

            cv2.imshow("Center-Locked (No Distortion / No Wobble Zoom)", display)

            # --- 로봇팔 제어(원본 좌표 기준) ---
            if not ever_locked:
                q.put({f"motor_{i}": 0 for i in range(1,7)})
            else:
                cmds = compute_motor_angles(use_cx, use_cy, area, frame.shape, kf=kf, kalman_inited=kalman_inited)
                cmds = clip_motor_angles(cmds, (-90, 90))
                cmds = apply_slew(cmds, max_delta=SLEW)
                q.put(cmds)

            # --- 키 입력/녹화 ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s') and not recording:
                output_path = get_new_filename()
                out = cv2.VideoWriter(output_path, fourcc, CAP_FPS, (w, h))
                if not out.isOpened():
                    print("VideoWriter 열기 실패"); out = None
                else: 
                    recording = True; print(f"녹화 시작: {os.path.basename(output_path)}")
            if key == ord('e') and recording:
                recording = False
                if out is not None: out.release(); out = None
                print("녹화 종료! 저장 완료")

            if recording and out is not None:
                clean = out_frame if RECORD_USE_STAB else frame
                #저장본 거울모드
                clean = cv2.flip(clean,1)
                out.write(clean)

    finally:
        try:
            if out is not None: out.release()
        except: pass
        cap_thread.release()
        cv2.destroyAllWindows()
        q.put(None)

if __name__ == "__main__":
    main()
