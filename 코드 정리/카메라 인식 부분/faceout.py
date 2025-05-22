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

# 영상 파일 자동 이름 생성
def get_new_filename(base_name="output", ext="avi"):
    existing_files = os.listdir(desktop_path)
    pattern = re.compile(rf"{re.escape(base_name)}_(\d+)\.{re.escape(ext)}")
    numbers = [int(match.group(1)) for file in existing_files if (match := pattern.match(file))]
    next_number = max(numbers, default=0) + 1
    return os.path.join(desktop_path, f"{base_name}_{next_number}.{ext}")

# 사진 파일 자동 이름 생성
def get_new_picture_filename(base_name="picture", ext="jpg"):
    existing_files = os.listdir(desktop_path)
    pattern = re.compile(rf"{re.escape(base_name)}_(\d+)\.{re.escape(ext)}")
    numbers = [int(match.group(1)) for file in existing_files if (match := pattern.match(file))]
    next_number = max(numbers, default=0) + 1
    return os.path.join(desktop_path, f"{base_name}_{next_number}.{ext}")

# 시리얼 통신 스레드
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

def compute_motor_angles(center_x, center_y, area, frame_shape, desired_area=30000):
    frame_h, frame_w = frame_shape[:2]
    dx = center_x - (frame_w // 2)
    dy = center_y - (frame_h // 2)
    dz = desired_area - area

    ddx = 0 if abs(dx) <= 50 else (-1 if dx > 0 else 1)
    ddy = 0 if abs(dy) <= 50 else (-1 if dy > 0 else 1)
    ddz = 0 if abs(dz) <= 10000 else (1 if dz > 0 else -1)

    delay = compute_delay(dx, dy)

    return {
        "motor_1": 0.5 * ddx,
        "motor_2": -0.5 * ddy,
        "motor_3": ddy,
        "motor_4": -0.5 * ddy + 0.5 * ddz,
        "motor_5": -ddz,
        "motor_6": 0.5 * ddz,
        "motor_7": delay
    }

motor_freeze_time = {"x": 0, "y": 0, "z": 0}

def should_freeze(axis, now):
    return now - motor_freeze_time[axis] < 0.7

def update_freeze_timer(ddx, ddy, ddz, now):
    if ddx == 0: motor_freeze_time["x"] = now
    if ddy == 0: motor_freeze_time["y"] = now
    if ddz == 0: motor_freeze_time["z"] = now

def clip_motor_angles(motor_cmds, limits=(-90, 90)):
    return {k: int(np.clip(v, limits[0], limits[1])) for k, v in motor_cmds.items()}

q = queue.Queue()
serial_thread = threading.Thread(target=serial_worker, args=(q,), daemon=True)
serial_thread.start()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라 열기 실패")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

last_face_info = {
    "cx": 320, "cy": 240, "area": 30000,
    "dx": 0, "dy": 0, "dz": 0,
    "frames_since_lost": 0
}

print("실행 중: 's'=녹화 시작, 'e'=녹화 종료, 'q'=종료")

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 읽기 실패")
        break

    now = time.time()
    key = cv2.waitKey(1) & 0xFF

    frame = cv2.flip(frame, 1)  # 좌우반전하여 거울처럼 보이도록 함
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

        print(f"[얼굴 추적] 중심 좌표: ({cx},{cy}) | 넓이: {area}")
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        angles = compute_motor_angles(cx, cy, area, frame.shape)

        ddx = angles["motor_1"]
        ddy = -angles["motor_2"]
        ddz = angles["motor_6"]
        update_freeze_timer(ddx, ddy, ddz, now)

        last_face_info.update({
            "cx": cx,
            "cy": cy,
            "area": area,
            "dx": 1 if cx - last_face_info["cx"] > 5 else (-1 if cx - last_face_info["cx"] < -5 else 0),
            "dy": 1 if cy - last_face_info["cy"] > 5 else (-1 if cy - last_face_info["cy"] < -5 else 0),
            "dz": ddz,
            "frames_since_lost": 0
        })

        if should_freeze("x", now): angles["motor_1"] = 0
        if should_freeze("y", now): angles["motor_2"] = 0; angles["motor_3"] = 0
        if should_freeze("z", now): angles["motor_4"] = 0; angles["motor_5"] = 0; angles["motor_6"] = 0

        clipped = clip_motor_angles(angles)
        q.put(clipped)
    else:
        last_face_info["frames_since_lost"] += 1

        last_face_info["cx"] += 5 * last_face_info["dx"]
        last_face_info["cy"] += 5 * last_face_info["dy"]

        print(f"[예측 추적] 얼굴 없음 | 예측 좌표: ({last_face_info['cx']},{last_face_info['cy']})")

        angles = {
            "motor_1": 0.5 * last_face_info["dx"],
            "motor_2": -0.5 * last_face_info["dy"],
            "motor_3": last_face_info["dy"],
            "motor_4": -0.5 * last_face_info["dy"] + 0.5 * last_face_info["dz"],
            "motor_5": -last_face_info["dz"],
            "motor_6": 0.5 * last_face_info["dz"],
            "motor_7": 15
        }
        clipped = clip_motor_angles(angles)
        q.put(clipped)

    cv2.imshow('Live Camera', frame)
    if key == ord('q'):
        print("프로그램 종료")
        break

cap.release()
cv2.destroyAllWindows()
q.put(None)
serial_thread.join()
