import cv2
import time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

registered_face = None  # (x, y, w, h)
registered_hist = None  # 등록된 얼굴의 히스토그램
register_time = 5

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

start_time = time.time()

def get_face_histogram(hsv_frame, x, y, w, h):
    roi = hsv_frame[y:y+h, x:x+w]
    hist = cv2.calcHist([roi], [0, 1], None, [30, 32], [0, 180, 0, 256])  # 수정된 bin 수
    cv2.normalize(hist, hist)
    return hist

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    h, w = frame.shape[:2]
    screen_center = (w // 2, h // 2)

    if registered_face is None:
        if time.time() - start_time < register_time and len(faces) > 0:
            min_dist = float("inf")
            best_face = None
            for (x, y, fw, fh) in faces:
                cx, cy = x + fw // 2, y + fh // 2
                dist = (cx - screen_center[0])**2 + (cy - screen_center[1])**2
                if dist < min_dist:
                    min_dist = dist
                    best_face = (x, y, fw, fh)
            if best_face:
                registered_face = best_face
                registered_hist = get_face_histogram(hsv, *best_face)
                cv2.putText(frame, "Face registered!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Look at the camera to register...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    else:
        rx, ry, rw, rh = registered_face
        r_center = (rx + rw // 2, ry + rh // 2)
        r_area = rw * rh

        best_match = None
        best_score = float("inf")

        for (x, y, fw, fh) in faces:
            cx, cy = x + fw // 2, y + fh // 2
            area = fw * fh
            d_pos = (cx - r_center[0])**2 + (cy - r_center[1])**2
            d_area = abs(area - r_area)

            hist = get_face_histogram(hsv, x, y, fw, fh)
            hist_diff = cv2.compareHist(registered_hist, hist, cv2.HISTCMP_BHATTACHARYYA)

            score = d_pos + d_area * 0.5 + hist_diff * 3000  # 가중치 조정 반영
            if score < best_score:
                best_score = score
                best_match = (x, y, fw, fh)
                best_hist = hist

        if best_match:
            x, y, fw, fh = best_match
            registered_face = best_match
            registered_hist = best_hist
            center_x = x + fw // 2
            center_y = y + fh // 2
            face_area = fw * fh

            # 시각화 및 출력
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"Target (x={center_x}, y={center_y}, area={face_area})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Face Tracker - HSV Hist (Tuned)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
