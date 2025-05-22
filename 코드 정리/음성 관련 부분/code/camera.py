import cv2
import os
import re
import time
import threading
import queue
import serial
import numpy as np

# 프로젝트 디렉터리 기준 경로 설정
project_root   = os.path.dirname(os.path.abspath(__file__))
video_folder   = os.path.join(project_root, "videos")
photo_folder   = os.path.join(project_root, "photos")
os.makedirs(video_folder, exist_ok=True)
os.makedirs(photo_folder, exist_ok=True)

# 파일명 생성 유틸
def get_new_filename(folder, base_name="output", ext="avi"):
    existing = os.listdir(folder)
    pattern  = re.compile(rf"{re.escape(base_name)}_(\d+)\.{re.escape(ext)}")
    nums      = [int(m.group(1)) for f in existing if (m := pattern.match(f))]
    return os.path.join(folder, f"{base_name}_{max(nums,default=0)+1}.{ext}")

def get_new_picture_filename(folder, base_name="picture", ext="jpg"):
    existing = os.listdir(folder)
    pattern  = re.compile(rf"{re.escape(base_name)}_(\d+)\.{re.escape(ext)}")
    nums      = [int(m.group(1)) for f in existing if (m := pattern.match(f))]
    return os.path.join(folder, f"{base_name}_{max(nums,default=0)+1}.{ext}")

# 얼굴 검출용 Haar cascade
frontal_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_profileface.xml')

# 모터 지연 계산
def compute_delay(dx, dy, min_delay=10, max_delay=20):
    dist       = np.hypot(dx, dy)
    normalized = min(dist/400, 1.0)
    return int(max_delay - (max_delay-min_delay)*normalized)

def compute_motor_angles(cx, cy, area, frame_shape, desired_area=50000):
    h, w = frame_shape[:2]
    dx, dy = cx - w//2, cy - h//2
    dz     = desired_area - area
    ddx    = 0 if abs(dx)<=50 else (-1 if dx>0 else 1)
    ddy    = 0 if abs(dy)<=80 else (-1 if dy>0 else 1)
    ddz    = 0  # 단순화
    delay  = compute_delay(dx, dy)
    return {
        "motor_1": ddx,
        "motor_2": -ddy,
        "motor_3": 2*ddy,
        "motor_4": -ddy+ddz,
        "motor_5": -2*ddz,
        "motor_6": ddz,
        "motor_7": delay
    }

def clip_motor_angles(cmds, limits=(-90,90)):
    return {k:int(np.clip(v, limits[0], limits[1])) for k,v in cmds.items()}

# 시리얼 워커
def serial_worker(q, port='COM5', baud=115200):
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        print("[Serial] 연결 완료")
    except Exception as e:
        print(f"[Serial] 연결 실패: {e}")
        return
    while True:
        cmds = q.get()
        if cmds is None:
            break
        # 최신 명령만
        while not q.empty():
            nxt = q.get_nowait()
            if nxt is not None:
                cmds = nxt
        vals    = [cmds[f"motor_{i}"] for i in range(1,8)]
        msg     = ','.join(map(str,vals))+'\n'
        ser.write(msg.encode())
        print(f"[Serial] Sent: {msg.strip()}")
        time.sleep(cmds["motor_7"]/1000)
    ser.close()
    print("[Serial] 종료")

class CameraRecorder:
    def __init__(self, device_index=1, fps=20.0):
        # 카메라 & 비디오세팅
        self.cap       = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("카메라 열기 실패")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        self.cap.set(cv2.CAP_PROP_FPS,30)

        self.fourcc    = cv2.VideoWriter_fourcc(*'MJPG')
        self.recording = False
        self.out       = None
        self.fps       = fps

        # 촬영 모드 변수
        self.photo_mode       = False
        self.photo_total      = 0
        self.photo_taken      = 0
        self.photo_interval   = 0
        self.photo_start_time = 0

        # 얼굴 추적 모터 큐
        self.q = queue.Queue()
        self.serial_thread = threading.Thread(
            target=serial_worker, args=(self.q,), daemon=True)
        self.serial_thread.start()

        # 캡처 백그라운드 스레드
        self._stop   = False
        self.prev_cx = self.prev_cy = None
        self.alpha   = 0.8
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def _capture_loop(self):
        while not self._stop:
            ret, frame = self.cap.read()
            if not ret: continue

            # 녹화
            if self.recording and self.out:
                self.out.write(frame)

            # 얼굴 추적 & 모터 제어
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            f1   = frontal_cascade.detectMultiScale(gray,1.1,8, minSize=(60,60))
            f2   = profile_cascade.detectMultiScale(gray,1.1,8, minSize=(60,60))
            flipped = cv2.flip(gray,1)
            f3   = profile_cascade.detectMultiScale(flipped,1.1,8, minSize=(60,60))
            # right 얼굴 좌표 보정
            for (x,y,w,h) in f3:
                f2 = list(f2)+[(frame.shape[1]-x-w, y, w, h)]
            faces = list(f1)+list(f2)
            if faces:
                faces.sort(key=lambda r:r[2]*r[3], reverse=True)
                x,y,w,h = faces[0]
                cx,cy   = x+w//2, y+h//2
                # 보간
                if self.prev_cx is not None:
                    cx = int(self.alpha*self.prev_cx + (1-self.alpha)*cx)
                    cy = int(self.alpha*self.prev_cy + (1-self.alpha)*cy)
                self.prev_cx, self.prev_cy = cx, cy
                area = w*h
                cmds = clip_motor_angles(
                    compute_motor_angles(cx, cy, area, frame.shape))
                self.q.put(cmds)
                # 시각화
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.circle(frame,(cx,cy),5,(0,0,255),-1)

            # 연속 촬영  
            if self.photo_mode:
                elapsed = time.time() - self.photo_start_time
                if elapsed < self.photo_interval:
                    sec = int(self.photo_interval - elapsed)
                    disp=frame.copy()
                    cv2.putText(disp,str(sec),(50,100),
                        cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,255),5)
                    cv2.imshow('Live Camera',disp)
                else:
                    fname = get_new_picture_filename(photo_folder)
                    cv2.imwrite(fname, frame)
                    self.photo_taken+=1
                    disp=frame.copy()
                    cv2.putText(disp,'Cheese!',(50,100),
                        cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),4)
                    cv2.imshow('Live Camera',disp)
                    cv2.waitKey(1000)
                    if self.photo_taken>=self.photo_total:
                        self.photo_mode=False
                        print("[Camera] 연속 촬영 완료")
                    else:
                        self.photo_start_time=time.time()
            else:
                cv2.imshow('Live Camera', frame)

            if cv2.waitKey(1)&0xFF==ord('q'):
                self._stop=True

    def start_recording(self):
        if self.recording: return
        path = get_new_filename(video_folder)
        h,w  = map(int,self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), \
               map(int,self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.out = cv2.VideoWriter(path,self.fourcc,self.fps,(w,h))
        self.recording=True
        print(f"[Camera] 녹화 시작 → {os.path.basename(path)}")

    def stop_recording(self):
        if not self.recording: return
        self.recording=False
        self.out.release()
        print("[Camera] 녹화 종료")

    def delayed_recording(self, delay=5):
        print(f"[Camera] {delay}초 후 녹화 예약")
        threading.Timer(delay,self.start_recording).start()

    def continuous_photo(self, count=3, interval=5):
        if self.recording: return
        self.photo_mode=True
        self.photo_total=count
        self.photo_taken=0
        self.photo_interval=interval
        self.photo_start_time=time.time()
        print(f"[Camera] 연속촬영: {count}장, {interval}초 간격")

    def close(self):
        self._stop=True
        self.cap.release()
        if self.out: self.out.release()
        self.q.put(None)
        self.serial_thread.join()
        cv2.destroyAllWindows()
