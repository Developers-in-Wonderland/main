# camera.py

import cv2
import os
import re
import time
import threading

# 프로젝트 디렉터리 기준 경로 설정
project_root = os.path.dirname(os.path.abspath(__file__))
video_folder  = os.path.join(project_root, "videos")
photo_folder  = os.path.join(project_root, "photos")
os.makedirs(video_folder, exist_ok=True)
os.makedirs(photo_folder, exist_ok=True)

# 영상 파일 자동 이름 생성
# ex) output_1.avi, output_2.avi ...
def get_new_filename(folder, base_name="output", ext="avi"):
    existing = os.listdir(folder)
    pattern = re.compile(rf"{re.escape(base_name)}_(\d+)\.{re.escape(ext)}")
    nums = [int(m.group(1)) for f in existing if (m := pattern.match(f))]
    return os.path.join(folder, f"{base_name}_{max(nums, default=0)+1}.{ext}")

# 사진 파일 자동 이름 생성
# ex) picture_1.jpg, picture_2.jpg ...
def get_new_picture_filename(folder, base_name="picture", ext="jpg"):
    existing = os.listdir(folder)
    pattern = re.compile(rf"{re.escape(base_name)}_(\d+)\.{re.escape(ext)}")
    nums = [int(m.group(1)) for f in existing if (m := pattern.match(f))]
    return os.path.join(folder, f"{base_name}_{max(nums, default=0)+1}.{ext}")

class CameraRecorder:
    def __init__(self, device_index=1, fps=20.0):  # USB 캠 인덱스 1로 기본 설정
        self.cap       = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"카메라 열기 실패 (device_index={device_index})")
        self.fourcc    = cv2.VideoWriter_fourcc(*'MJPG')
        self.recording = False
        self.out       = None
        self.fps       = fps
        self._stop     = False
        self.frame     = None
        # 연속 촬영 상태 변수
        self.photo_mode      = False
        self.photo_total     = 0
        self.photo_taken     = 0
        self.photo_interval  = 0
        self.photo_start_time= 0
        # 백그라운드 프레임 읽기 스레드 시작
        self.thread    = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        while not self._stop:
            ret, frame = self.cap.read()
            if not ret:
                continue
            self.frame = frame.copy()
            # 녹화 중이면 파일에 기록
            if self.recording and self.out:
                self.out.write(self.frame)

            # 연속 촬영 중 카운트다운 및 촬영 처리
            if self.photo_mode:
                now = time.time()
                elapsed = now - self.photo_start_time
                if elapsed < self.photo_interval:
                    # 화면에 카운트다운 표시
                    sec_left = int(self.photo_interval - elapsed)
                    disp = self.frame.copy()
                    cv2.putText(disp, str(sec_left), (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5)
                    cv2.imshow('Live Camera', disp)
                else:
                    # 촬영 시점
                    filename = get_new_picture_filename(photo_folder)
                    cv2.imwrite(filename, self.frame)
                    print(f"{self.photo_taken+1}번째 사진 저장됨: {os.path.basename(filename)}")
                    # Cheat sheet overlay
                    disp = self.frame.copy()
                    cv2.putText(disp, 'Cheese!', (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                    cv2.imshow('Live Camera', disp)
                    cv2.waitKey(1000)
                    # 준비 다음 촬영
                    self.photo_taken += 1
                    if self.photo_taken >= self.photo_total:
                        self.photo_mode = False
                        print("[Camera] 연속 촬영 완료")
                    else:
                        self.photo_start_time = now
            else:
                # 일반 라이브 뷰 표시
                cv2.imshow('Live Camera', self.frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self._stop = True

    def start_recording(self):
        if self.recording:
            print("이미 녹화 중입니다.")
            return
        if self.frame is None:
            print("아직 프레임을 가져오지 못했습니다.")
            return
        path = get_new_filename(video_folder)
        h, w = self.frame.shape[:2]
        self.out = cv2.VideoWriter(path, self.fourcc, self.fps, (w, h))
        if not self.out.isOpened():
            print("VideoWriter 열기 실패")
            return
        self.recording = True
        print(f"[Camera] 녹화 시작 → {os.path.basename(path)}")

    def stop_recording(self):
        if not self.recording:
            print("녹화 중이 아닙니다.")
            return
        self.recording = False
        self.out.release()
        self.out = None
        print("[Camera] 녹화 종료")

    def delayed_recording(self, delay=5):
        print(f"[Camera] {delay}초 후 녹화 시작 예약")
        threading.Timer(delay, self.start_recording).start()

    def continuous_photo(self, count=3, interval=5):
        print(f"[Camera] 연속 촬영 시작: 총 {count}장, {interval}초 간격")
        if self.frame is None:
            print("프레임이 준비되지 않았습니다.")
            return
        self.photo_mode       = True
        self.photo_total      = count
        self.photo_taken      = 0
        self.photo_interval   = interval
        self.photo_start_time = time.time()

    def close(self):
        self._stop = True
        self.thread.join()
        self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()
