import cv2 #openCV 얼굴인식, 영상처리. 프레임 캡쳐 등
import serial #아두이노 시리얼 통신을 위한 모듈
import time # 시간지연, 아두이노 초기화 대기에 사용
import numpy as np # 수학 계산, clip()/각도 제한 처리용
import threading # 파이썬 비동기 처리, 시리얼 통신을 영상과 동시에 돌림
import queue # 두 스레드 간 데이터 주고 받을 떄 사용용

############ 시리얼 전송 스레드 ###############
#시리얼 전송을 담당할 백그라운드 쓰레드 함수
# q를 통해 얼굴 중심 정보 계산한 모터값 받음 -> 시리얼 포트를 통해 아두이노에 전달달
def serial_worker(q, port='COM3', baud=115200):
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)  # 아두이노 리셋 대기
        #아두이노는 시리얼 포트가 열릴 때 자동리셋되므로 2초간 대기
        print("시리얼 연결 완료")
    except Exception as e:
        print(f"시리얼 연결 실패: {e}")
        return

    while True:
        motor_cmds = q.get()
        if motor_cmds is None:
            break
        # q에서 각도 값을 하나 꺼내고, none일 시 종료
        values = [motor_cmds[f"motor_{i}"] for i in range(1, 7)]
        message = ','.join(map(str, values)) + '\n'
        ser.write(message.encode('utf-8'))
        # 모터 1~6의 값을 문자열로 묶어서 전송한다
        # ex) '10,-5,3,0,1,2\n'
        print(f"[Serial] Sent: {message.strip()}")

    ser.close()
    print("시리얼 종료")

# 얼굴 인식기 / 이미 기존 코드가 따로 있고 합칠 예정정
frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

########각도 계산 ########
# 얼굴 위치에 따른 모터 제어 각도 계산
# 얼굴의 중심좌표, 넓이, 영상 크기를 입령 받아 로봇팔의 6개의 모터가
# 어떻게 움직일지를 계산한다.
def compute_motor_angles(center_x, center_y, area, frame_shape, desired_area=42000):
    frame_h, frame_w = frame_shape[:2]
    dx = center_x - (frame_w // 2)
    dy = center_y - (frame_h // 2)
    dz = desired_area - area
    # 중심 좌표가 영상 중심과 얼마나 차이 나는지 계산한다
    # dz = 거리를 넓이(area)로 추정한 오차이다.
    
	# 각 오차에 비례한 각도 변화량 / 오차가 클수록 세게 조정, 작아지면 천천히 조정정
    # 계수는 민감도 / 저장장
    return {
        "motor_1": dx * 0.1,
        "motor_2": dy * 0.1,
        "motor_3": dy * 0.05,
        "motor_4": dz * 0.0005,
        "motor_5": dy * 0.02,
        "motor_6": dx * 0.05
    }

##### 모터 각도 제한 함수 #####
# 값이 너무 크거나 작을 시, 모터 고장위험 있음 -> 제한
# np.clip()로 -90 ~ +90 범위로 고정
def clip_motor_angles(motor_cmds, limits=(-90, 90)):
    return {k: int(np.clip(v, limits[0], limits[1])) for k, v in motor_cmds.items()}

# 시리얼 큐 및 스레드 시작
q = queue.Queue()
# serial_worker()를 별도의 스레드로 실행해
# 영상처리와 독립적으로 아두이노에 값을 계속해서 전송
serial_thread = threading.Thread(target=serial_worker, args=(q,), daemon=True)
serial_thread.start()

# 카메라 시작 
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 카메라 해상도 낮춰 속도 개선
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("카메라 열기 실패")
    exit()

print("실행 중... (q 키로 종료)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 읽기 실패")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_frontal = frontal_cascade.detectMultiScale(gray, 1.1, 8, minSize=(60, 60))
    faces_profile = profile_cascade.detectMultiScale(gray, 1.1, 8, minSize=(60, 60))

    # 우측 얼굴 보정
    flipped = cv2.flip(gray, 1)
    faces_right = profile_cascade.detectMultiScale(flipped, 1.1, 8, minSize=(60, 60))
    for (x, y, w, h) in faces_right:
        x_corr = frame.shape[1] - x - w
        faces_profile = list(faces_profile)
        faces_profile.append((x_corr, y, w, h))

    # 모든 얼굴 통합
    all_faces = list(faces_frontal) + list(faces_profile)

    if all_faces:
        all_faces.sort(key=lambda r: r[2] * r[3], reverse=True)
        x, y, w, h = all_faces[0]
        cx, cy = x + w // 2, y + h // 2
        area = w * h

        # 시각화
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"({cx},{cy})", (cx - 50, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, f"Area: {area}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 각도 계산 + 큐 전송
        # 백그라운드 스레드가 아두이노로 값 전송송
        angles = compute_motor_angles(cx, cy, area, frame.shape)
        clipped = clip_motor_angles(angles)
        q.put(clipped)
    else:
        print("얼굴 없음")

    cv2.imshow('Live Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료 처리
cap.release()
cv2.destroyAllWindows()
q.put(None)
serial_thread.join()
