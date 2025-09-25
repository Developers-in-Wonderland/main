import cv2
import numpy as np
import threading
import queue
import serial
import os
import re
import time

# DNN 기본 코드 (흔들림 보정 x), 로봇팔 연결도 안해놓음 하려면 COM3 -> COM5로 바꾸기

# ====== 저장 경로 (바탕화면) ======
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

# ====== 파일 자동 이름 생성 ======
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

# ====== 보기 좋은 텍스트 드로잉 ======
def draw_text(img, text, org, font_scale=0.8, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness, cv2.LINE_AA)

############ 시리얼 전송 스레드 ############
def serial_worker(q, port='COM5', baud=115200):
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

    ser.close()
    print("시리얼 종료")

# ====== DNN 얼굴검출 ======
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

# ====== 로봇팔 제어 ======
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

# ====== 스레드/큐 시작 ======
q = queue.Queue()
serial_thread = threading.Thread(target=serial_worker, args=(q,), daemon=True)
serial_thread.start()

# ====== 카메라 ======
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
  # 노트북 카메라
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
if not cap.isOpened():
    print("카메라 열기 실패")
    q.put(None)
    raise SystemExit

# ====== 상태 변수 ======
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
recording = False
out = None

photo_shooting = False
photo_count = 0
photo_taken = 0
photo_interval = 3.0
next_shot_at = None

# ====== 콘솔 로그(터미널) 설정 ======
LOG_INTERVAL = 0.3  # 초
last_face_log_ts = 0.0

print("키 안내: s=녹화 시작, e=녹화 종료, 1~9=연속사진(3초 간격), q=종료")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 캡처 실패")
            break

        now = time.time()

        # ---- 표시용 복사본 (오버레이는 여기만 그린다) ----
        display = frame.copy()

        # ===== 얼굴 검출 + 가장 큰 얼굴 선택 =====
        face_boxes = detect_faces_dnn(frame)
        if face_boxes:
            face_boxes.sort(key=lambda b: b[2]*b[3], reverse=True)
            x, y, w, h = face_boxes[0]
            center_x = x + w // 2
            center_y = y + h // 2
            area = w * h

            # 화면 오버레이는 display에만 그림
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.circle(display, (center_x, center_y), 3, (0, 200, 0), -1)
            t1_y = y - 26 if y - 26 > 10 else y + h + 22
            t2_y = t1_y + 22
            draw_text(display, f"Center: ({center_x}, {center_y})", (x, t1_y), 0.7, 2)
            draw_text(display, f"Area: {area}", (x, t2_y), 0.7, 2)

            # 콘솔 로그 (얼굴 있을 때만, 0.3s 간격)
            if now - last_face_log_ts >= LOG_INTERVAL:
                print(f"[FACE] center=({center_x}, {center_y}), area={area}")
                last_face_log_ts = now

            # 로봇팔 제어
            cmds = compute_motor_angles(center_x, center_y, area, frame.shape)
            cmds = clip_motor_angles(cmds, (-90, 90))
            q.put(cmds)

        # ===== 녹화 상태 (표시용 프레임으로 저장: 오버레이가 비디오에 보이게) =====
        if recording and out is not None:
            out.write(display)
            draw_text(display, "REC ●", (10, display.shape[0]-15), 0.8, 2)

        # ===== 연속촬영 로직 =====
        if photo_shooting and next_shot_at is not None:
            remain_sec = max(0.0, next_shot_at - now)
            remain_ceil = int(np.ceil(remain_sec))

            # 카운트다운(좌상단) — display에만
            if remain_ceil >= 3:
                cd = "3"
            elif remain_ceil == 2:
                cd = "2"
            elif remain_ceil == 1:
                cd = "1 cheese~!"
            else:
                cd = "cheese~!"
            draw_text(display, cd, (20, 40), 1.2, 2)

            # 남은 장수(우상단)
            shots_left = max(0, photo_count - photo_taken)
            txt = f"Shots left: {shots_left}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            draw_text(display, txt, (display.shape[1] - tw - 20, 40), 0.8, 2)

            # 촬영 시점
            if now >= next_shot_at:
                filename = get_new_picture_filename()
                # ★ 사진 저장은 '원본 frame'으로 — 오버레이 없음 ★
                cv2.imwrite(filename, frame)
                photo_taken += 1
                print(f"{photo_taken}/{photo_count} 저장됨: {os.path.basename(filename)}")

                if photo_taken >= photo_count:
                    photo_shooting = False
                    next_shot_at = None
                    print("연속 사진 촬영 완료!")
                else:
                    next_shot_at = now + photo_interval

        # ===== 화면 표시 (오버레이 포함본) =====
        cv2.imshow("Face Tracker (Overlay shown, Photos saved clean)", display)

        # ===== 키 입력 =====
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # 녹화 시작 (비디오는 오버레이 포함 저장)
        if key == ord('s') and not recording:
            output_path = get_new_filename()
            h, w = display.shape[:2]
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

        # 연속촬영 시작 (1~9)
        if (ord('1') <= key <= ord('9')) and not photo_shooting:
            photo_count = key - ord('0')
            photo_taken = 0
            photo_shooting = True
            next_shot_at = now + photo_interval  # 3초 뒤 첫 장
            print(f"{photo_count}장의 사진 연속 촬영 시작! (간격 {photo_interval:.0f}초)")

finally:
    if recording and out is not None:
        out.release()
    cap.release()
    cv2.destroyAllWindows()
    q.put(None)
