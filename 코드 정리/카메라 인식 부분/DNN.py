import cv2
import face_recognition
import time
import numpy as np

# OpenCV DNN용 모델 로드
prototxt_path = cv2.data.haarcascades.replace("haarcascade_frontalface_default.xml", "deploy.prototxt")
model_path = cv2.data.haarcascades.replace("haarcascade_frontalface_default.xml", "res10_300x300_ssd_iter_140000.caffemodel")

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

registered_encoding = None
register_time = 5
start_time = time.time()

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
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_boxes = detect_faces_dnn(frame)

    if registered_encoding is None:
        if time.time() - start_time < register_time and len(face_boxes) > 0:
            x, y, w, h = face_boxes[0]  # 첫 얼굴만 등록
            face_roi = rgb_frame[y:y+h, x:x+w]
            encodings = face_recognition.face_encodings(face_roi)
            if len(encodings) > 0:
                registered_encoding = encodings[0]
                cv2.putText(frame, "Face registered!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Look at camera to register...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        for (x, y, w, h) in face_boxes:
            face_roi = rgb_frame[y:y+h, x:x+w]
            encodings = face_recognition.face_encodings(face_roi)
            if len(encodings) > 0:
                dist = face_recognition.face_distance([registered_encoding], encodings[0])[0]
                if dist < 0.5:
                    center_x = x + w // 2
                    center_y = y + h // 2
                    area = w * h
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Target ({dist:.2f})", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    # 여기에 center_x, center_y, area 값 → 아두이노로 시리얼 전송 가능

    cv2.imshow("DNN Face Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
