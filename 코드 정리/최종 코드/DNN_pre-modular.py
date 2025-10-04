# ─────────────────────────────────────────────────
# DNN 얼굴인식 + 중앙 추종(교정된 모터매핑) + 영상녹화 + 연속촬영
#  - 프리뷰만 오버레이 / 저장본은 깨끗하게
#  - 비선형 맵핑(tanh) + 속도제한 + 플립가드 + X/Y 반전 토글
#  - 프레임마다 터미널 로깅
# ─────────────────────────────────────────────────
import cv2, serial, time, numpy as np, threading, queue, os, re, sys

# ===============================
# 0) 경로/환경
# ===============================
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
os.makedirs(desktop_path, exist_ok=True)
try: sys.stderr = open(os.devnull, 'w')
except: pass

# ===============================
# 1) 시스템 설정
# ===============================
SERIAL_PORT = 'COM5'          # ⚠️ 실제 포트
SERIAL_BAUD  = 115200
FRAME_W, FRAME_H, FPS = 1920, 1080, 30
VIDEO_FPS, VIDEO_CODEC = 20.0, 'MJPG'

DESIRED_AREA = 42000          # 목표 얼굴 면적(거리 제어)
EMA_ALPHA    = 0.6            # 추적 스무딩(0.4~0.8 사이 튜닝)
DETECT_EVERY = 2              # DNN 간헐 실행(2=2프레임마다)

# 스레드/큐
stop_all_threads = threading.Event()
motor_q   = queue.Queue(maxsize=10)
move_ready = threading.Event(); move_ready.set()

# 좌/우, 상/하 반전 토글(실행 중 x,y 키로 전환)
INVERT_X = 0   # 좌우 축(베이스 pan)
INVERT_Y = 0   # 상하 축(팔/카메라 tilt)

# ===============================
# 2) 파일명 자동 증가
# ===============================
def _next_inc(base, ext):
    pat = re.compile(rf"{re.escape(base)}_(\d+)\.{re.escape(ext)}")
    nums = [int(m.group(1)) for f in os.listdir(desktop_path) if (m := pat.match(f))]
    return os.path.join(desktop_path, f"{base}_{max(nums, default=0)+1}.{ext}")
def get_new_filename():      return _next_inc("output", "avi")
def get_new_pic_filename():  return _next_inc("picture", "jpg")

# ===============================
# 3) DNN 얼굴검출
# ===============================
prototxt = r"C:\face_models\deploy.prototxt"
model    = r"C:\face_models\res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def detect_faces(frame, conf=0.5):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,(300,300),(104,177,123))
    net.setInput(blob)
    det = net.forward()
    boxes = []
    for i in range(det.shape[2]):
        if det[0,0,i,2] > conf:
            x1,y1,x2,y2 = (det[0,0,i,3:7]*np.array([w,h,w,h])).astype(int)
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(w-1,x2), min(h-1,y2)
            if x2>x1 and y2>y1: boxes.append((x1,y1,x2-x1,y2-y1))
    return boxes

# ===============================
# 4) 모터 제어(교정된 매핑)
#     1: 베이스 좌우(PAN)  ← dx
#     2,3,5,6: 상하(TILT)   ← dy
#     4: 거리(Z)           ← dz
# ===============================
last_step = {"x":0,"y":0,"z":0}
last_sign_ts = {"x":0.0,"y":0.0}
FLIP_GUARD_S = 0.18
MAX_DELTA_STEP = 12

def _rate_limit(prev, new, lmt=MAX_DELTA_STEP):
    return int(max(min(new, prev + lmt), prev - lmt))

def _tanh_map(err, scale, max_step):
    return int(np.clip(np.tanh(err/scale)*max_step, -max_step, max_step))

def compute_motor_angles(cx, cy, area, shape,
                         dead_xy=12, dead_area=9000,
                         max_step_pan=60, max_step_tilt=60, max_step_z=60,
                         scale_x=160, scale_y=120, scale_z=30000):
    """
    반환 dict:
      motor_1..6: 각 축 스텝, motor_7: 다음 명령까지 delay(ms)
    """
    h, w = shape[:2]
    # 부호 토글 반영
    dx = (cx - w//2) * (-1 if INVERT_X else 1)    # +면 화면 오른쪽 → 베이스 오른쪽 회전
    dy = (cy - h//2) * (-1 if INVERT_Y else 1)    # +면 화면 아래   → 팔/카메라 아래로
    dz = DESIRED_AREA - area

    now = time.time()
    if abs(dx) <= dead_xy:   dx = 0
    if abs(dy) <= dead_xy:   dy = 0
    if abs(dz) <= dead_area: dz = 0

    # 비선형 맵핑(작게 부드럽게, 크게 빠르게)
    sx = _tanh_map(dx, scale_x, max_step_pan)   # pan(좌우)   → motor_1
    sy = _tanh_map(dy, scale_y, max_step_tilt)  # tilt(상하)  → motor_2,3,5,6
    sz = _tanh_map(dz, scale_z, max_step_z)     # 거리        → motor_4

    # 플립가드(짧은 시간 내 방향 급반전 방지)
    def flip_guard(axis, val):
        if val == 0: return 0
        prev = last_step[axis]
        if np.sign(val) != np.sign(prev) and (now - last_sign_ts[axis] < FLIP_GUARD_S):
            return 0
        if np.sign(val) != np.sign(prev):
            last_sign_ts[axis] = now
        return val

    sx = flip_guard("x", sx)
    sy = flip_guard("y", sy)

    # 속도 제한(프레임간 변화량 제한)
    sx = _rate_limit(last_step["x"], sx)
    sy = _rate_limit(last_step["y"], sy)
    sz = _rate_limit(last_step["z"], sz)
    last_step.update({"x":sx, "y":sy, "z":sz})

    # 에러가 클수록 더 짧은 지연(더 자주 명령)
    err_mag = abs(dx) + abs(dy)
    delay_ms = max(12, 40 - min(int(err_mag/12), 26))

    # ⚠️ 매핑 교정: 1=PAN(dx), 2/3/5/6=TILT(dy), 4=Z(dz)
    return {
        "motor_1": sx,          # 베이스 좌우
        "motor_2": sy,          # 팔 상하1
        "motor_3": int(sy*0.6), # 팔 상하2(조금 덜)
        "motor_4": sz,          # 전후(Z)
        "motor_5": int(sy*0.4), # 미세 틸트
        "motor_6": int(sy*0.8), # 카메라 틸트
        "motor_7": delay_ms
    }

def draw_text(img, text, org, fs=0.8, th=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), th+2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0,0,0), th, cv2.LINE_AA)

# ===============================
# 5) 시리얼 워커(가장 최신 명령만)
# ===============================
def serial_worker():
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
        time.sleep(2); print("시리얼 연결 완료")
    except Exception as e:
        print("시리얼 실패:", e); return

    while not stop_all_threads.is_set():
        try:
            cmd = motor_q.get(timeout=0.1)
            if cmd is None: break
            while not motor_q.empty():
                nxt = motor_q.get_nowait()
                if nxt is not None: cmd = nxt
            vals = [cmd.get(f"motor_{i}", 0) for i in range(1,8)]
            ser.write((','.join(map(str, vals))+'\n').encode())
            time.sleep(cmd.get("motor_7", 20)/1000.0)
            move_ready.set()
        except queue.Empty:
            continue
        except Exception as e:
            print("Serial error:", e); break
    try: ser.close()
    except: pass
    print("시리얼 종료")

# ===============================
# 6) 메인 루프
# ===============================
def main():
    serial_thread = threading.Thread(target=serial_worker, daemon=True)
    serial_thread.start()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    if not cap.isOpened():
        print("카메라 열기 실패"); return

    recording, out = False, None
    photo_mode, photo_count, shots_done, next_shot = False,0,0,None
    prev_cx = prev_cy = None
    frame_id = 0
    last_faces = []

    print("키: s=녹화, e=녹화종료, 1~9=연속사진, q=종료, x=좌우반전, y=상하반전")

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            now = time.time()
            display = frame.copy()

            # DNN(간헐)
            frame_id += 1
            run_dnn = (frame_id % DETECT_EVERY == 0)
            faces = detect_faces(frame, 0.5) if run_dnn else last_faces
            if run_dnn: last_faces = faces

            if faces:
                faces.sort(key=lambda b:b[2]*b[3], reverse=True)
                x,y,w,h = faces[0]
                cx, cy = x+w//2, y+h//2
                if prev_cx is None: smx,smy = cx,cy
                else:
                    smx = int(EMA_ALPHA*prev_cx + (1-EMA_ALPHA)*cx)
                    smy = int(EMA_ALPHA*prev_cy + (1-EMA_ALPHA)*cy)
                prev_cx, prev_cy = smx, smy
                area = w*h

                # 프리뷰 오버레이
                cv2.rectangle(display,(x,y),(x+w,y+h),(0,200,0),2)
                cv2.circle(display,(smx,smy),4,(0,200,0),-1)
                draw_text(display, f"Center:({smx},{smy}) Area:{area}", (x, max(20, y-10)))

                # === 모터 제어 ===
                if move_ready.is_set() and not motor_q.full():
                    ang = compute_motor_angles(smx, smy, area, frame.shape)

                    # ------- 프레임 로그(터미널) -------
                    dx = (smx - frame.shape[1]//2) * (-1 if INVERT_X else 1)
                    dy = (smy - frame.shape[0]//2) * (-1 if INVERT_Y else 1)
                    dz = DESIRED_AREA - area
                    print(f"[FRAME] cx,cy=({cx},{cy}) sm=({smx},{smy}) "
                          f"dx={dx:4d} dy={dy:4d} dz={dz:6d}  "
                          f"M1={ang['motor_1']:3d} M2={ang['motor_2']:3d} "
                          f"M3={ang['motor_3']:3d} M4={ang['motor_4']:3d} "
                          f"M5={ang['motor_5']:3d} M6={ang['motor_6']:3d} d={ang['motor_7']:2d}ms")

                    motor_q.put(ang)
            else:
                draw_text(display, "No face", (10, 40))
                if move_ready.is_set() and not motor_q.full():
                    stop_cmd = {f"motor_{i}":0 for i in range(1,7)}; stop_cmd["motor_7"]=20
                    print("[FRAME] no-face → STOP")
                    motor_q.put(stop_cmd)

            # 연속 촬영
            if photo_mode and next_shot is not None:
                remain = next_shot - now
                if remain <= 0:
                    cv2.imwrite(get_new_pic_filename(), frame)
                    shots_done += 1
                    print(f"{shots_done}/{photo_count} 저장")
                    if shots_done >= photo_count:
                        photo_mode = False; next_shot = None; print("연속 사진 완료")
                    else: next_shot = now + 3.0
                else:
                    draw_text(display, str(int(np.ceil(remain))), (20,70), 1.6, 3)

            # 프리뷰
            cv2.imshow("DNN Face Tracker (corrected mapping)", display)

            # 키 입력
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('x'):
                globals()['INVERT_X'] ^= 1; print("INVERT_X =", INVERT_X)
            if key == ord('y'):
                globals()['INVERT_Y'] ^= 1; print("INVERT_Y =", INVERT_Y)

            if key == ord('s') and not recording:
                out = cv2.VideoWriter(get_new_filename(),
                        cv2.VideoWriter_fourcc(*VIDEO_CODEC), VIDEO_FPS, (frame.shape[1], frame.shape[0]))
                if out.isOpened(): recording=True; print("녹화 시작")
                else: out=None; print("VideoWriter 열기 실패")
            if key == ord('e') and recording:
                recording=False; out.release(); out=None; print("녹화 종료")
            if (ord('1') <= key <= ord('9')) and not photo_mode:
                photo_count = key - ord('0'); shots_done = 0
                photo_mode = True; next_shot = now + 3.0
                print(f"{photo_count}장 연속촬영 시작")

            if recording and out is not None:
                out.write(frame)

    finally:
        stop_all_threads.set()
        if recording and out: out.release()
        if cap.isOpened(): cap.release()
        cv2.destroyAllWindows()
        motor_q.put(None)
        print("정리 완료")

if __name__ == "__main__":
    main()
