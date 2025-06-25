import cv2
import numpy as np
import threading # 파이썬 비동기 처리, 시리얼 통신을 영상과 동시에 돌림
import queue
import serial
import os # 경로 조작에 사용됨
import re # 정규표현식 사용을 위함, 파일 이름에서 숫자를 추출할 때 사용
import time # 연속 촬영때 사용, 시간 측정 및 대기 기능능

# 바탕화면 경로 설정
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

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

############ 시리얼 전송 스레드 ###############
def serial_worker(q, port='COM3', baud=115200):
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)  # 아두이노 리셋 대기
        print("시리얼 연결 완료")
    except Exception as e:
        print(f"시리얼 연결 실패: {e}")
        return

    while True:
        motor_cmds = q.get()
        if motor_cmds is None:
            break
        values = [motor_cmds[f"motor_{i}"] for i in range(1, 7)]
        message = ','.join(map(str, values)) + '\n'
        ser.write(message.encode('utf-8'))
        print(f"[Serial] Sent: {message.strip()}")

    ser.close()
    print("시리얼 종료")

prototxt_path = r"C:\face_models\deploy.prototxt"
model_path = r"C:\face_models\res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

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

q = queue.Queue()
serial_thread = threading.Thread(target=serial_worker, args=(q,), daemon=True)
serial_thread.start()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("카메라 열기 실패")
    exit()

# 변수 초기화
recording = False
out = None
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
photo_shooting = False
photo_count = 0
photo_taken = 0
photo_interval = 5
countdown_start_time = 0

# 조작 설명
print("실행 중: 's'=녹화 시작, 'e'=녹화 종료, 숫자=연속 사진촬영, 'q'=종료")

def detect_faces_dnn(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    key = cv2.waitKey(1) & 0xFF    
    face_boxes = detect_faces_dnn(frame)

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

    if face_boxes:
        # 가장 큰 박스만 사용
        face_boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        x, y, w, h = face_boxes[0]
        center_x = x + w // 2
        center_y = y + h // 2
        area = w * h

        print(f"Center: ({center_x}, {center_y}), Area: {area}")
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Super Smooth Face Tracker (Fast)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
