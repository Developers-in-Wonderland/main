import cv2 # openCV 라이브러리 불러오기기

# Haar Cascade 로드
# 정면 얼굴과 측면 얼굴 인식을 위함
# ++
frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# 카메라 연결
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# 영상 프레임을 반복해서 처리하는 루프 시작
while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break

    # 얼굴 인식을 위해 흑백으로 변환    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 정면 얼굴 인식 (정확도 향상을 위해 파라미터 튜닝)
    faces_frontal = frontal_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,        # 탐색 스케일 / 더 정밀하게 얼굴 크기 스케일 탐색
        minNeighbors=8,         # 최소 8개의 이웃이 있어야 진짜 얼굴로 판단 / 높은 신뢰도의 얼굴만 통과
        minSize=(60, 60)        # 너무 작은 얼굴은 무시 가로/세로 60픽셀 이하인 얼굴
    )

    # 좌측 측면 얼굴 인식
    faces_profile_left = profile_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(60, 60)
    )

    # 우측 측면 얼굴 인식을 위해 좌우 반전
    # 좌우 반전
    flipped_gray = cv2.flip(gray, 1)
    faces_profile_right = profile_cascade.detectMultiScale(
        flipped_gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(60, 60)
    )

    # 우측 얼굴 좌표 보정 후 좌측과 합치기
     # 우측 얼굴은 좌우 반전되었기 때문에 x 좌표를 원래대로 보정
    for (x, y, w, h) in faces_profile_right:
        x_corrected = frame.shape[1] - x - w
        faces_profile_left = list(faces_profile_left)
        faces_profile_left.append((x_corrected, y, w, h))

    # 전체 얼굴 후보 통합 (정면 + 측면면)
    all_faces = list(faces_frontal) + list(faces_profile_left)

    # 가장 큰 얼굴 하나만 사용
    # 하나의 얼굴만 표시하기 위해
    if all_faces:
        all_faces.sort(key=lambda rect: rect[2] * rect[3], reverse=True)
        # 가장 큰 얼굴 추출
        x, y, w, h = all_faces[0]

        # 얼굴 중심 좌표 및 넓이 계산
        center_x = x + w // 2
        center_y = y + h // 2
        area = w * h

        # 사각형 및 정보 표시
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 얼굴 중심을 빨간 점으로 표시
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        # 중심 좌표 텍스트 출력
        cv2.putText(frame, f"({center_x}, {center_y})", (center_x - 50, center_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # 얼굴 넓이 텍스트 출력
        cv2.putText(frame, f"Area: {area}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 화면 출력
    cv2.imshow('Live Camera', frame)

    # 'q' 키로 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 정리
cap.release()
cv2.destroyAllWindows()
