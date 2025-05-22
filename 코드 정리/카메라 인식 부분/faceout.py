import cv2
import serial
import time
import numpy as np
import threading
import queue
import os
import re

# 바탕화면 경로 설정
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

move_ready = threading.Event()
move_ready.set()

# 파일 이름 자동 생성 함수들
def get_new_filename(base_name="output", ext="avi"):
    existing_files = os.listdir(desktop_path)
    pattern = re.compile(rf"{re.escape(base_name)}_(\d+)\.{re.escape(ext)}")
    numbers = [int(match.group(1)) for file in existing_files if (match := pattern.match(file))]
    next_number = max(numbers, default=0) + 1
    return os.path.join(desktop_path, f"{base_name}_{next_number}.{ext}")

def get_new_picture_filename(base_name="picture", ext="jpg"):
    existing_files = os.listdir(desktop_path)
    pattern = re.compile(rf"{re.escape(base_name)}_(\d+)\.{re.escape(ext)}")
    numbers = [int(match.group(1)) for file in existing_files if (match := pattern.match(file))]
    next_number = max(numbers, default=0) + 1
    return os.path.join(desktop_path, f"{base_name}_{next_number}.{ext}")

# 시리얼 전송 스레드
def serial_worker(q, port='COM5', baud=115200):
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

        while not q.empty():
            latest = q.get_nowait()
            if latest is not None:
                motor_cmds = latest

        values = [motor_cmds[f"motor_{i}"] for i in range(1, 8)]
        message = ','.join(map(str, values)) + '\n'
        ser.write(message.encode('utf-8'))
        print(f"[Serial] Sent: {message.strip()}")

        delay_ms = motor_cmds["motor_7"]
        move_ready.clear()
        time.sleep(delay_ms / 1000.0)
        move_ready.set()
    ser.close()
    print("시리얼 종료")

frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

def compute_delay(dx, dy, min_delay=10, max_delay=20):
    distance = np.sqrt(dx**2 + dy**2)
    normalized = min(distance / 400, 1.0)
    delay = int(max_delay - (max_delay - min_delay) * normalized)
    return delay

def compute_motor_angles(center_x, center_y, area, frame_shape, desired_area=50000):
    frame_h, frame_w = frame_shape[:2]
    dx = center_x - (frame_w // 2)
    dy = center_y - (frame_h // 2)
    dz = desired_area - area

    ddx = 0 if abs(dx) <= 50 else (-1 if dx > 0 else 1)
    ddy = 0 if abs(dy) <= 80 else (-1 if dy > 0 else 1)
    ddz = 0 if abs(dz) <= 3000 else (1 if dz > 0 else -1)
    ddz = 0

    delay = compute_delay(dx, dy)
    return {
        "motor_1": ddx,
        "motor_2": -ddy,
        "motor_3": 2 * ddy,
        "motor_4": -ddy + ddz,
        "motor_5": -(2 * ddz),
        "motor_6": ddz,
        "motor_7": delay
    }

def clip_motor_angles(motor_cmds, limits=(-90, 90)):
    return {k: int(np.clip(v, limits[0], limits[1])) for k, v in motor_cmds.items()}

def dynamic_alpha(dx, dy, base_alpha=0.7, max_alpha=0.95):
    distance = np.sqrt(dx**2 + dy**2)
    normalized = min(distance / 150, 1.0)
    return base_alpha + (max_alpha - base_alpha) * (1 - normalized)

def shift_frame_to_center(frame, cx, cy, alpha=0.8, prev_center=None):
    frame_h, frame_w = frame.shape[:2]
    target_x, target_y = frame_w // 2, frame_h // 2
    if prev_center is not None:
        cx = int(alpha * prev_center[0] + (1 - alpha) * cx)
        cy = int(alpha * prev_center[1] + (1 - alpha) * cy)
    dx = target_x - cx
    dy = target_y - cy
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(frame, M, (frame_w, frame_h))
    return shifted, (cx, cy)

q = queue.Queue()
serial_thread = threading.Thread(target=serial_worker, args=(q,), daemon=True)
serial_thread.start()

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("카메라 열기 실패")
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

prev_cx, prev_cy = None, None
recording = False
out = None
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
photo_shooting = False
photo_count = 0
photo_taken = 0
photo_interval = 5
countdown_start_time = 0

last_cmd = None
no_face_frames = 0
max_no_face_frames = 30

print("실행 중: 's'=녹화 시작, 'e'=녹화 종료, 숫자=연속 사진촬영, 'q'=종료")

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 읽기 실패")
        break

    now = time.time()
    key = cv2.waitKey(1) & 0xFF

    if key == ord('e') and recording:
        print("녹화 종료! 영상이 저장되었습니다.")
        recording = False
        out.release()
        out = None

    if key == ord('s') and not recording and not photo_shooting:
        output_path = get_new_filename()
        print(f"녹화 시작! 저장 파일명: {os.path.basename(output_path)}")
        height, width = frame.shape[:2]
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
        if not out.isOpened():
            print("VideoWriter 열기 실패")
            break
        recording = True

    if recording:
        out.write(frame)
        cv2.putText(frame, "Recording...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if ord('0') < key <= ord('9') and not photo_shooting:
        photo_count = key - ord('0')
        photo_taken = 0
        countdown_start_time = now
        photo_shooting = True
        print(f"{photo_count}장의 사진 연속 촬영 시작!")

    if photo_shooting:
        elapsed = now - countdown_start_time
        seconds_left = photo_interval - int(elapsed)
        clean_frame = frame.copy()

        if seconds_left > 0:
            cv2.putText(frame, f"{seconds_left}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
        elif seconds_left == 0 and elapsed < photo_interval + 1:
            cv2.putText(frame, "Cheese~!", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)

        shots_left = photo_count - photo_taken
        cv2.putText(frame, f"{shots_left}", (frame.shape[1] - 60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3)

        if elapsed >= photo_interval:
            filename = get_new_picture_filename()
            cv2.imwrite(filename, clean_frame)
            photo_taken += 1
            print(f"{photo_taken}번째 저장됨: {os.path.basename(filename)}")

            if photo_taken >= photo_count:
                photo_shooting = False
                print("연속 사진 촬영 완료!")
            else:
                countdown_start_time = now

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_frontal = frontal_cascade.detectMultiScale(gray, 1.1, 8, minSize=(60, 60))
    faces_profile = profile_cascade.detectMultiScale(gray, 1.1, 8, minSize=(60, 60))
    flipped = cv2.flip(gray, 1)
    faces_right = profile_cascade.detectMultiScale(flipped, 1.1, 8, minSize=(60, 60))
    for (x, y, w, h) in faces_right:
        x_corr = frame.shape[1] - x - w
        faces_profile = list(faces_profile)
        faces_profile.append((x_corr, y, w, h))

    all_faces = list(faces_frontal) + list(faces_profile)
    if all_faces and move_ready.is_set():
        all_faces.sort(key=lambda r: r[2] * r[3], reverse=True)
        x, y, w, h = all_faces[0]
        cx, cy = x + w // 2, y + h // 2
        area = w * h

        dx = cx - (frame.shape[1] // 2)
        dy = cy - (frame.shape[0] // 2)
        alpha = dynamic_alpha(dx, dy)
        frame, new_center = shift_frame_to_center(frame, cx, cy, alpha, (prev_cx, prev_cy))
        prev_cx, prev_cy = new_center

        print(f"[얼굴 추적] 중심 좌표: ({cx},{cy}) | 넓이: {area} | alpha: {round(alpha,2)}")

        angles = compute_motor_angles(cx, cy, area, frame.shape)
        clipped = clip_motor_angles(angles)
        q.put(clipped)
        last_cmd = clipped
        no_face_frames = 0
    else:
        print("얼굴 없음")
        no_face_frames += 1
        if last_cmd and no_face_frames <= max_no_face_frames:
            print("→ 예측 이동 실행 중")
            q.put(last_cmd)
        elif no_face_frames > max_no_face_frames:
            print("→ 예측 기간 종료, 정지")

    cv2.imshow('Live Camera', frame)
    if key == ord('q'):
        print("프로그램 종료")
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
q.put(None)
serial_thread.join()
