import cv2
import numpy as np
import threading  # 파이썬 비동기 처리, 시리얼 통신을 영상과 동시에 돌림
import queue
import serial
import os  # 경로 조작에 사용됨
import re  # 정규표현식 사용을 위함, 파일 이름에서 숫자를 추출할 때 사용
import time  # 연속 촬영때 사용, 시간 측정 및 대기 기능

# 바탕화면 경로 설정
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

# ===== 파일 자동 이름 생성 =====
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

# ===== 가독성 좋은 텍스트 드로잉 =====
def draw_text(img, text, org, font_scale=1.0, thickness=2):
    # 하얀 외곽선 + 검정 본문
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness, cv2.LINE_AA)

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
        values = [motor_cmds.get(f"motor_{i}", 0) for i in range(1, 7)]
        message = ','.join(map(str, values)) + '\n'
        ser.write(message.encode('utf-8'))
        # print(f"[Serial] Sent: {message.strip()}")

    ser.close()
    print("시리얼 종료")

# ===== DNN 로드 =====
prototxt_path = r"C:\face_models\deploy.prototxt"
model_path = r"C:\face_models\res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

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

# ===== 로봇팔 제어 =====
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

# ===== 스레드/큐 시작 =====
q = queue.Queue()
serial_thread = threading.Thread(target=serial_worker, args=(q,), daemon=True)
serial_thread.start()

# ===== 카메라 =====
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("카메라 열기 실패")
    # 시리얼 종료 신호
    q.put(None)
    exit()

# ===== 상태 변수 =====
recording = False
out = None
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

# --- 연속촬영 상태 ---
photo_shooting = False     # 연속촬영 진행 중?
photo_count = 0            # 총 촬영 장수
photo_taken = 0            # 이미 찍은 장수
photo_interval = 3.0       # 인터벌 (초) —— 요구사항 3초
next_shot_at = None        # 다음 촬영 시각 (epoch)

# 안내
print("실행 중: 's'=녹화 시작, 'e'=녹화 종료, '1~9'=연속 사진촬영(3초 간격), 'q'=종료")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 캡처 실패")
            break

        now = time.time()

        # ===== 얼굴 인식 + 가장 큰 얼굴 추적 =====
        face_boxes = detect_faces_dnn(frame)
        if face_boxes:
            face_boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
            x, y, w, h = face_boxes[0]
            center_x = x + w // 2
            center_y = y + h // 2
            area = w * h

            # 박스/중심 표시
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.circle(frame, (center_x, center_y), 3, (0, 200, 0), -1)

            # 로봇팔 제어 명령 전송
            cmds = compute_motor_angles(center_x, center_y, area, frame.shape)
            cmds = clip_motor_angles(cmds, (-90, 90))
            q.put(cmds)

        # ===== 녹화 상태 =====
        if recording and out is not None:
            out.write(frame)
            draw_text(frame, "REC ●", (10, frame.shape[0]-15), 0.8, 2)

        # ===== 연속촬영 로직 =====
        if photo_shooting and next_shot_at is not None:
            remain_sec = max(0.0, next_shot_at - now)
            remain_ceil = int(np.ceil(remain_sec))

            # 좌상단 카운트다운
            if remain_ceil >= 3:
                cd = "3"
            elif remain_ceil == 2:
                cd = "2"
            elif remain_ceil == 1:
                cd = "1 cheese~!"
            else:
                cd = "cheese~!"
            draw_text(frame, cd, (20, 40), 1.2, 2)

            # 우상단 남은 장 수
            shots_left = max(0, photo_count - photo_taken)
            txt = f"Shots left: {shots_left}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            draw_text(frame, txt, (frame.shape[1] - tw - 20, 40), 0.8, 2)

            # 촬영 시점 도달 -> 저장
            if now >= next_shot_at:
                filename = get_new_picture_filename()
                cv2.imwrite(filename, frame)  # 오버레이 없이 원본이 필요하면 frame 복사본을 사용
                photo_taken += 1
                print(f"{photo_taken}/{photo_count} 저장됨: {os.path.basename(filename)}")

                if photo_taken >= photo_count:
                    # 종료
                    photo_shooting = False
                    next_shot_at = None
                    print("연속 사진 촬영 완료!")
                else:
                    next_shot_at = now + photo_interval  # 다음 장

        # ===== 화면 표시 =====
        cv2.imshow("Super Smooth Face Tracker (Fast)", frame)

        # ===== 키 입력 =====
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # 녹화 시작
        if key == ord('s') and not recording:
            output_path = get_new_filename()
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))
            if not out.isOpened():
                print("VideoWriter 열기 실패")
                out = None
            else:
                recording = True
                print(f"녹화 시작! 저장 파일명: {os.path.basename(output_path)}")

        # 녹화 종료
        if key == ord('e') and recording:
            recording = False
            if out is not None:
                out.release()
                out = None
            print("녹화 종료! 영상이 저장되었습니다.")

        # 1~4 키: 연속촬영 시작 (요구사항대로 1~4만 허용)
        # 1~9 키: 연속촬영 시작 (1~9장)
        if (ord('1') <= key <= ord('9')) and not photo_shooting:
            photo_count = key - ord('0')   # 입력 숫자만큼 촬영
            photo_taken = 0
            photo_shooting = True
            next_shot_at = now + photo_interval   # 기본 3초
            print(f"{photo_count}장의 사진 연속 촬영 시작! (간격 {photo_interval:.0f}초)")

finally:
    # 정리
    if recording and out is not None:
        out.release()
    cap.release()
    cv2.destroyAllWindows()
    # 시리얼 종료 신호
    q.put(None)
