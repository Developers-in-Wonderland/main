import cv2 # openCV 얼굴인식, 영상처리. 프레임 캡쳐 등
import serial # 아두이노 시리얼 통신을 위한 모듈
import time # 시간지연, 아두이노 초기화 대기에 사용
import numpy as np # 수학 계산, clip()/각도 제한 처리용
import threading # 파이썬 비동기 처리, 시리얼 통신을 영상과 동시에 돌림
import queue # 두 스레드 간 데이터 주고 받을 떄 사용용
import os # 경로 조작에 사용됨
import re # 정규표현식 사용을 위함, 파일 이름에서 숫자를 추출할 때 사용

# 바탕화면 경로 설정
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

move_ready = threading.Event()
move_ready.set()  # 처음엔 동작 준비 상태

# 예측 움직임 관련 변수
last_known_angles = None
last_face_detected_time = 0
PREDICTION_COOLDOWN_S = 0.8 # 얼굴 놓친 후 예측 움직임 시작까지 대기 시간 (조금 늘림)
PREDICTION_ACTIVE_DURATION_S = 2.0 # 예측 움직임 시도 시간 (조금 늘림)
prediction_start_time = 0
is_predicting_movement = False
DECAY_RATE = 0.65 # 감쇠율 (조금 조정)
prediction_decay_count = 0
MAX_PREDICTION_REPEAT = 3 # 예측 반복 횟수

# --- 스캔 모드 관련 변수 추가 ---
is_scanning = False
SCAN_WAIT_DURATION_S = 4.0  # 예측 실패 후 스캔 시작까지 대기 시간 (조금 줄임)
scan_start_time = 0         # 스캔 대기 시작 시간 또는 실제 스캔 시작 시간
scan_direction = 1          # 1: 오른쪽, -1: 왼쪽 (또는 시계방향/반시계방향)
SCAN_STEP_MOTOR1 = 1        # 한 스텝당 motor_1 변화량 (좌우 회전 담당 모터)
SCAN_MAX_STEPS_ONE_DIRECTION = 20 # 한 방향으로 최대 스캔 스텝 수 (조금 늘림)
scan_current_step_count = 0 # 현재 방향으로 진행한 스텝 수
SCAN_DELAY_MS = 120         # 스캔 시 움직임 딜레이 (조금 늘려 더 천천히)
RETURN_TO_DEFAULT_DELAY_MS = 250 # 기본 위치 복귀 시 딜레이 (조금 늘림)
# --- 스캔 모드 변수 끝 ---

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
def serial_worker(q, port='COM5', baud=115200): # COM 포트 확인 필요
    ser = None # ser 변수 초기화
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2) # 아두이노 리셋 대기
        print(f"시리얼 연결 완료: {port} @ {baud}bps")
    except serial.SerialException as e:
        print(f"시리얼 연결 실패 ({port}): {e}")
        print("사용 가능한 COM 포트를 확인하거나, 아두이노 연결 상태를 점검하세요.")
        return # 시리얼 연결 실패 시 스레드 종료

    while True:
        try:
            motor_cmds = q.get()
            if motor_cmds is None: # 종료 신호
                print("[Serial] 종료 신호 수신")
                break

            # 큐에 쌓인 이전 명령 버리기 (최신 것만 실행)
            while not q.empty():
                # print("[Serial Queue] 이전 명령 버리는 중...") # 디버깅용
                latest = q.get_nowait() # non-blocking get
                if latest is not None:
                    motor_cmds = latest
                else: # None을 만나면 루프 탈출 (종료 신호 처리)
                    break
            if motor_cmds is None: # latest가 None이었을 경우
                 print("[Serial] 최종 명령이 None이므로 종료")
                 break


            values = [motor_cmds[f"motor_{i}"] for i in range(1, 8)]
            message = ','.join(map(str, values)) + '\n'
            ser.write(message.encode('utf-8'))
            print(f"[Serial] Sent: {message.strip()}")

            delay_ms = motor_cmds["motor_7"]

            move_ready.clear()  # 이동 중 상태 설정
            time.sleep(delay_ms / 1000.0) # 명령 실행 시간 대기
            move_ready.set()    # 이동 끝, 다시 추적 가능하게

        except queue.Empty: # get_nowait에서 발생 가능 (이론상 발생 안 함)
            continue
        except serial.SerialException as e:
            print(f"[Serial Error] 시리얼 통신 오류: {e}")
            # 필요시 재연결 로직 추가 가능
            break # 오류 발생 시 스레드 종료
        except Exception as e:
            print(f"[Serial Worker Error] 예기치 않은 오류: {e}")
            break


    if ser and ser.is_open:
        ser.close()
    print("시리얼 스레드 종료됨.")

# 얼굴 인식 분류기 로드
frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml') # 필요시 사용

# 모터 제어 로직 함수
def compute_delay(dx, dy, min_delay=20, max_delay=60): # 딜레이 범위 약간 조정
    distance = np.sqrt(dx**2 + dy**2)
    normalized = min(distance / 320, 1.0)  # 화면 절반 정도 거리에서 최대 정규화 (640/2)
    delay = int(max_delay - (max_delay - min_delay) * normalized)
    return delay

def compute_motor_angles(center_x, center_y, area, frame_shape, desired_area=35000, deadzone_xy=20, deadzone_area=12000):
    frame_h, frame_w = frame_shape[:2]
    dx = center_x - (frame_w // 2)
    dy = center_y - (frame_h // 2)
    dz = desired_area - area # 양수: 가까워져야 함 (면적 작음), 음수: 멀어져야 함 (면적 큼)

    ddx = 0 if abs(dx) <= deadzone_xy else (-1 if dx > 0 else 1)  # dx > 0: 얼굴이 오른쪽에 -> 왼쪽으로 이동 (-1)
    ddy = 0 if abs(dy) <= deadzone_xy else (-1 if dy > 0 else 1)  # dy > 0: 얼굴이 아래쪽에 -> 위로 이동 (-1)
    ddz = 0 if abs(dz) <= deadzone_area else (1 if dz > 0 else -1) # dz > 0: 면적 작음 -> 가까이 (1)

    delay = compute_delay(dx, dy)

    # 모터 연결 및 운동학에 따라 이 값들을 조정해야 합니다.
    # 예시 값이며, 실제 로봇에 맞게 계수를 변경하세요.
    return {
        "motor_1": ddx,      # 예: 팬(좌우) 모터, ddx와 동일 방향 가정
        "motor_2": 0, # 예: 틸트(상하) 관련 모터1
        "motor_3": ddy,      # 예: 틸트(상하) 관련 모터2
        "motor_4": 3 * ddz, # 예: 틸트와 줌(거리) 결합 모터
        "motor_5": -2 * ddz, # 예: 줌(거리) 관련 모터1
        "motor_6": ddz, # 예: 줌(거리) 관련 모터2
        "motor_7": delay
    }

# 모터 고정(Freeze) 관련
motor_freeze_time = {"x": 0, "y": 0, "z": 0}
FREEZE_DURATION_S = 0.6 # 고정 지속 시간 약간 줄임
def should_freeze(axis, now):
    return now - motor_freeze_time[axis] < FREEZE_DURATION_S

def update_freeze_timer(ddx, ddy, ddz, now):
    if ddx == 0: motor_freeze_time["x"] = now
    if ddy == 0: motor_freeze_time["y"] = now
    if ddz == 0: motor_freeze_time["z"] = now

def clip_motor_angles(motor_cmds, limits=(-80, 80)): # 클리핑 범위 약간 줄임
    # motor_7 (delay)은 클리핑 대상에서 제외하거나 다른 범위 적용 가능
    clipped = {}
    for k, v in motor_cmds.items():
        if k == "motor_7":
            clipped[k] = int(np.clip(v, 10, 500)) # delay는 양수, 적절한 범위
        else:
            clipped[k] = int(np.clip(v, limits[0], limits[1]))
    return clipped

# --- 메인 실행 부분 ---
q = queue.Queue(maxsize=10) # 큐 크기 제한 (선택적)
serial_thread = threading.Thread(target=serial_worker, args=(q,), daemon=True)
serial_thread.start()

# 카메라 초기화
cap = cv2.VideoCapture(1,cv2.CAP_DSHOW) # 카메라 인덱스 0번 사용, 환경에 따라 변경
if not cap.isOpened():
    print("카메라를 열 수 없습니다. 카메라 연결 상태를 확인하세요.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30) # 실제 프레임률은 카메라 성능에 따라 다름

# 녹화 관련 변수
recording = False
out = None
fourcc = cv2.VideoWriter_fourcc(*'XVID') # 코덱 변경 시도 (MJPG도 좋음)

# 사진 촬영 관련 변수
photo_shooting = False
photo_count = 0
photo_taken = 0
photo_interval = 3 # 사진 촬영 간격 줄임
countdown_start_time = 0

print("프로그램 실행 중... 's'=녹화, 'e'=녹화종료, 숫자=사진촬영, 'q'=종료")

# --- 메인 루프 ---
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다. 카메라 연결이 끊겼을 수 있습니다.")
            # 재시도 로직 또는 종료
            time.sleep(0.5)
            cap.release()
            cap = cv2.VideoCapture(0) # 간단한 재시도
            if not cap.isOpened():
                print("카메라 재연결 실패. 종료합니다.")
                break
            ret, frame = cap.read()
            if not ret:
                print("프레임 읽기 최종 실패. 종료합니다.")
                break


        now = time.time()
        key = cv2.waitKey(1) & 0xFF

        # 'q' 키로 종료
        if key == ord('q'):
            print("사용자 요청으로 프로그램 종료 중...")
            break

        # 녹화 로직
        if key == ord('s') and not recording and not photo_shooting:
            output_path = get_new_filename()
            height, width = frame.shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
            if out.isOpened():
                recording = True
                print(f"녹화 시작: {os.path.basename(output_path)}")
            else:
                print(f"VideoWriter 열기 실패: {output_path}")
        elif key == ord('e') and recording:
            recording = False
            if out:
                out.release()
                out = None
            print("녹화 종료!")

        if recording and out:
            out.write(frame)
            cv2.putText(frame, "REC", (frame.shape[1] - 70, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 사진 촬영 로직
        if ord('0') < key <= ord('9') and not photo_shooting and not recording:
            photo_count = key - ord('0')
            photo_taken = 0
            countdown_start_time = now
            photo_shooting = True
            print(f"{photo_count}장의 사진 연속 촬영 시작 (간격: {photo_interval}초)")

        if photo_shooting:
            elapsed_photo_time = now - countdown_start_time
            seconds_left_for_photo = photo_interval - int(elapsed_photo_time)
            clean_frame_for_photo = frame.copy() # 원본 프레임 복사

            # 카운트다운 표시
            if seconds_left_for_photo > 0:
                cv2.putText(frame, str(seconds_left_for_photo), (frame.shape[1] // 2 - 30, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 3)
            elif seconds_left_for_photo == 0 and elapsed_photo_time < photo_interval + 0.5: # 0.5초간 "촬영!" 표시
                cv2.putText(frame, "Shoot!", (frame.shape[1] // 2 - 100, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)

            # 남은 촬영 컷 수 표시
            shots_left_text = f"{photo_count - photo_taken} shots left"
            cv2.putText(frame, shots_left_text, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if elapsed_photo_time >= photo_interval:
                picture_filename = get_new_picture_filename()
                cv2.imwrite(picture_filename, clean_frame_for_photo) # 텍스트 없는 프레임 저장
                photo_taken += 1
                print(f"사진 저장됨 ({photo_taken}/{photo_count}): {os.path.basename(picture_filename)}")

                if photo_taken >= photo_count:
                    photo_shooting = False
                    print("연속 사진 촬영 완료!")
                else:
                    countdown_start_time = now # 다음 사진 촬영 위해 타이머 리셋

        # 얼굴 인식 및 추적 로직 (사진 촬영 중에는 추적 안 함)
        if not photo_shooting:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 얼굴 인식 (정면만 사용, 필요시 측면 추가)
            faces = frontal_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=7, minSize=(70, 70))

            if len(faces) > 0: # 얼굴 감지됨
                if is_predicting_movement or is_scanning: # 예측/스캔 중이었다면 중단
                    print("[상태] 얼굴 재감지. 예측/스캔 중단.")
                is_predicting_movement = False
                is_scanning = False
                prediction_decay_count = 0
                scan_current_step_count = 0
                scan_start_time = 0  # 스캔 대기 시간 초기화
                # 얼굴 감지 시 큐 비우기 (중요)
                if not q.empty():
                    with q.mutex:
                        q.queue.clear()
                    print("[Queue] 얼굴 감지, 이전 명령 큐 비움.")

                # 가장 큰 얼굴 선택
                (x, y, w, h) = max(faces, key=lambda item: item[2] * item[3])
                center_x, center_y = x + w // 2, y + h // 2
                face_area = w * h

                # print(f"[얼굴 추적] 중심:({center_x},{center_y}), 넓이:{face_area}") # 디버깅용

                # 모터 각도 계산
                current_angles = compute_motor_angles(center_x, center_y, face_area, frame.shape)

                # 화면에 얼굴 영역 표시 (녹화 중이 아닐 때만)
                if not recording:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                print(f"[얼굴 추적] 중심 좌표: ({center_x}, {center_y}) | 넓이: {face_area}")

                # Freeze 로직 적용 위한 ddx, ddy, ddz 계산
                # compute_motor_angles의 deadzone 값과 일치시켜야 함
                deadzone_xy_val = 40
                deadzone_area_val = 12000
                desired_area_val = 35000

                dx_val = center_x - (frame.shape[1] // 2)
                dy_val = center_y - (frame.shape[0] // 2)
                dz_val = desired_area_val - face_area

                ddx_freeze = 0 if abs(dx_val) <= deadzone_xy_val else (-1 if dx_val > 0 else 1)
                ddy_freeze = 0 if abs(dy_val) <= deadzone_xy_val else (-1 if dy_val > 0 else 1)
                ddz_freeze = 0 if abs(dz_val) <= deadzone_area_val else (1 if dz_val > 0 else -1)

                update_freeze_timer(ddx_freeze, ddy_freeze, ddz_freeze, now)

                # 실제 모터 명령에 Freeze 적용
                if should_freeze("x", now): current_angles["motor_1"] = 0
                if should_freeze("y", now):
                    current_angles["motor_2"] = 0
                    current_angles["motor_3"] = 0
                    current_angles["motor_4"] = 0 if should_freeze("z", now) else (2 * ddz_freeze) # motor_4의 Z축 기여분만 남김
                if should_freeze("z", now):
                    current_angles["motor_5"] = 0
                    current_angles["motor_6"] = 0
                    if not should_freeze("y", now): # Y축이 안 얼었을 때 Z축 고정이 motor_4에 영향
                        current_angles["motor_4"] = -0.5 * ddy_freeze # motor_4의 Y축 기여분만 남김

                clipped_angles = clip_motor_angles(current_angles)
                last_known_angles = clipped_angles.copy() # 마지막 유효 각도 저장
                last_face_detected_time = now

                if move_ready.is_set() and not q.full():
                    q.put(clipped_angles)
                    # print(f"[Serial Track] Sent: {clipped_angles}") # 디버깅용

            else: # 얼굴 없음 (is_scanning, is_predicting_movement 또는 스캔 대기 상태)
                current_status_text = "얼굴 없음"
                if is_scanning: # 현재 스캔 중
                    current_status_text = "스캔 중..."
                    if move_ready.is_set():
                        scan_current_step_count += 1
                        if scan_current_step_count > SCAN_MAX_STEPS_ONE_DIRECTION:
                            scan_direction *= -1 # 방향 전환
                            scan_current_step_count = 0 # 현재 방향 스텝 카운트 리셋
                            print(f"[스캔] 방향 전환: {'->' if scan_direction == 1 else '<-'} (최대 {SCAN_MAX_STEPS_ONE_DIRECTION} 스텝)")
                        
                        scan_command = {f"motor_{i}": 0 for i in range(1, 7)}
                        scan_command["motor_1"] = SCAN_STEP_MOTOR1 * scan_direction
                        scan_command["motor_7"] = SCAN_DELAY_MS
                        
                        if not q.full():
                            q.put(scan_command)
                            # print(f"[Serial Scan] Sent: {scan_command}") # 디버깅용

                elif is_predicting_movement: # 현재 예측 중
                    current_status_text = "예측 중..."
                    elapsed_prediction_time = now - prediction_start_time
                    if elapsed_prediction_time > PREDICTION_ACTIVE_DURATION_S:
                        print(f"[예측 종료] 시간 초과 ({PREDICTION_ACTIVE_DURATION_S:.1f}초).")
                        is_predicting_movement = False
                        last_known_angles = None # 예측 완전 실패
                        scan_start_time = now # 스캔 대기 시작
                        # 예측 실패 시 로봇 정지
                        stop_command = {f"motor_{i}": 0 for i in range(1, 7)}
                        stop_command["motor_7"] = RETURN_TO_DEFAULT_DELAY_MS
                        if not q.full(): q.put(stop_command)

                    elif move_ready.is_set() and last_known_angles:
                        if prediction_decay_count < MAX_PREDICTION_REPEAT:
                            decayed_predict_cmd = last_known_angles.copy()
                            for i in range(1, 7): # motor_1 ~ motor_6 감쇠
                                motor_key = f"motor_{i}"
                                decayed_predict_cmd[motor_key] = int(decayed_predict_cmd[motor_key] * (DECAY_RATE ** prediction_decay_count))
                            decayed_predict_cmd["motor_7"] = 60 # 예측 시 딜레이 (조금 늘림)
                            
                            if not q.full():
                                q.put(decayed_predict_cmd)
                                # print(f"[Serial Predict] Sent (Decay {prediction_decay_count+1}): {decayed_predict_cmd}")
                            prediction_decay_count += 1
                        else: # 최대 예측 반복 도달
                            print(f"[예측 종료] 최대 반복 ({MAX_PREDICTION_REPEAT}회) 도달.")
                            is_predicting_movement = False
                            last_known_angles = None
                            scan_start_time = now # 스캔 대기 시작
                            stop_command = {f"motor_{i}": 0 for i in range(1, 7)}
                            stop_command["motor_7"] = RETURN_TO_DEFAULT_DELAY_MS
                            if not q.full(): q.put(stop_command)
                
                # 예측/스캔 중 아니고, 얼굴 없어진 지 일정 시간 지났을 때 예측 시작
                elif last_known_angles and move_ready.is_set() and not is_scanning and not is_predicting_movement:
                    if (now - last_face_detected_time) > PREDICTION_COOLDOWN_S:
                        print(f"[예측 시작] {PREDICTION_COOLDOWN_S:.1f}초 동안 얼굴 못찾음. 마지막 방향으로 이동 시도.")
                        is_predicting_movement = True
                        prediction_start_time = now
                        prediction_decay_count = 0 # 예측 시작 시 decay count 초기화
                        
                        initial_predict_cmd = last_known_angles.copy()
                        initial_predict_cmd["motor_7"] = 70 # 첫 예측 딜레이 (조금 늘림)
                        if not q.full():
                            q.put(initial_predict_cmd)
                            # print(f"[Serial Predict] Sent (Initial): {initial_predict_cmd}")
                
                # 예측 완전 실패 후, 스캔 대기 시간 경과 시 스캔 시작
                elif last_known_angles is None and not is_scanning and not is_predicting_movement:
                    current_status_text = "스캔 대기 중..."
                    if (now - scan_start_time) > SCAN_WAIT_DURATION_S:
                        print(f"[스캔 준비] {SCAN_WAIT_DURATION_S:.1f}초 대기 후 스캔 시작.")
                        is_scanning = True
                        scan_current_step_count = 0
                        scan_direction = 1 # 초기 스캔 방향
                        
                        # 기본 위치로 로봇팔 이동 명령
                        # 실제 로봇의 기본(중앙) 자세에 맞게 motor_1~6 값 설정 필요
                        default_position_cmd = {
                            "motor_1": 0, "motor_2": 0, "motor_3": 0,
                            "motor_4": 0, "motor_5": 0, "motor_6": 0,
                            "motor_7": RETURN_TO_DEFAULT_DELAY_MS
                        }
                        # 큐 비우고 기본 위치 명령 전송
                        if not q.empty():
                            with q.mutex: q.queue.clear()
                        if not q.full():
                            q.put(default_position_cmd)
                            print(f"[Robo] 기본 위치로 이동 명령: {default_position_cmd}")
                # else:
                    # print(f"[Debug] No face, LKA: {last_known_angles is not None}, Scanning: {is_scanning}, Predicting: {is_predicting_movement}")


                # '얼굴 없음' 상태 텍스트 표시 (녹화 중 아닐 때)
                if not recording:
                    status_color = (0,0,0)
                    if is_scanning: status_color = (255,100,0) # 파란색 계열
                    elif is_predicting_movement: status_color = (0,165,255) # 주황색
                    else: status_color = (0,0,255) # 빨간색 (일반적인 얼굴 없음)

                    cv2.putText(frame, current_status_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # 현재 상태 화면에 표시 (스캔, 예측)
        if is_scanning:
            scan_dir_text = 'R' if scan_direction == 1 else 'L'
            cv2.putText(frame, f"Scanning... Dir: {scan_dir_text} ({scan_current_step_count}/{SCAN_MAX_STEPS_ONE_DIRECTION})",
                        (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
        elif is_predicting_movement:
            cv2.putText(frame, f"Predicting... (Try {prediction_decay_count}/{MAX_PREDICTION_REPEAT})",
                        (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)


        cv2.imshow("Face Tracking Robot Arm", frame)

except KeyboardInterrupt: # Ctrl+C로 종료 시
    print("Ctrl+C 입력. 프로그램 강제 종료 중...")
finally:
    # --- 종료 처리 ---
    print("메인 루프 종료. 리소스 정리 시작...")
    if cap.isOpened():
        cap.release()
        print("카메라 리소스 해제됨.")
    if out:
        out.release()
        print("녹화 파일 리소스 해제됨.")
    cv2.destroyAllWindows()
    print("OpenCV 윈도우 닫힘.")

    if serial_thread.is_alive():
        print("시리얼 스레드 종료 신호 전송 및 대기...")
        q.put(None) # 시리얼 스레드 종료 신호
        serial_thread.join(timeout=3) # 최대 3초 대기
        if serial_thread.is_alive():
            print("시리얼 스레드가 시간 내에 종료되지 않았습니다.")
        else:
            print("시리얼 스레드 정상 종료됨.")
    else:
        print("시리얼 스레드가 이미 종료되었거나 시작되지 않았습니다.")

    print("프로그램이 완전히 종료되었습니다.")
