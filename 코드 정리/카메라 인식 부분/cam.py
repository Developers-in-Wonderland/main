#opencv와 연결
import cv2

# Haar Cascade 얼굴 검출기 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 아직 캠이 오지 않았으므로 기초 작업은 노트북의 기본 카메라(웹캠) 연결해 확인 (카메라 ID: 0)
cap = cv2.VideoCapture(0)

# 카메라가 열리지 않을 시 (오류 상황의 경우)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()  # 카메라에서 프레임 읽기
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 흑백 변환 (속도 향상을 위해)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # 얼굴 중심 좌표 계산
        center_x = x + w // 2
        center_y = y + h // 2
        area = w * h  # 얼굴 넓이 계산

        # 얼굴을 사각형으로 표시
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 얼굴 중심 좌표 표시
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"({center_x}, {center_y})", (center_x - 50, center_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 얼굴 넓이 출력
        cv2.putText(frame, f"Area: {area}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('Live Camera', frame)  # 화면에 프레임 표시

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # 카메라 연결 해제
cv2.destroyAllWindows()  # 창 닫기
