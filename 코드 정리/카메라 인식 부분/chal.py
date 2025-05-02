import cv2 # openCV의 라이브러리, 영상처리용
import os # 경로 조작에 사용됨
import re # 정규표현식 사용을 위함, 파일 이름에서 숫자를 추출할 때 사용
import time # 연속 촬영때 사용, 시간 측정 및 대기 기능능

# 바탕화면 경로 설정
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

# 영상 파일 자동 이름 생성
# ex) output1, output2 이런식으로
def get_new_filename(base_name="output", ext="avi"):
    #바탕화면에 있는 모든 파일 목록을 불러온 뒤
    existing_files = os.listdir(desktop_path)
    #파일 이름 중 output_숫자.avi형식에 맞는 것만 찾기 위한 '정규표현식'정의
    pattern = re.compile(rf"{re.escape(base_name)}_(\d+)\.{re.escape(ext)}")
    #output_1.avi, output_2.avi 등 숫자만 추출해서 리스트에 저장
    #매칭 동시에 변수에 저장
    numbers = [int(match.group(1)) for file in existing_files if (match := pattern.match(file))]
    #가장 큰 번호에 +1, 새로운 파일 번호 저장
    next_number = max(numbers, default=0) + 1
    #최종적으로 저장한 전체 경로 반환 
    return os.path.join(desktop_path, f"{base_name}_{next_number}.{ext}")

# 사진 파일 자동 이름 생성
# 위 영상저장 방식과 동일
# ex) picture1, picture2 이런식으로
def get_new_picture_filename(base_name="picture", ext="jpg"):
    #바탕화면에 있는 모든 파일 목록을 불러온 뒤
    existing_files = os.listdir(desktop_path)
    #파일 이름 중 picture_숫자.avi형식에 맞는 것만 찾기 위한 '정규표현식'정의
    pattern = re.compile(rf"{re.escape(base_name)}_(\d+)\.{re.escape(ext)}")
    #picture_1.avi, picture_2.avi 등 숫자만 추출해서 리스트에 저장
    #매칭 동시에 변수에 저장
    numbers = [int(match.group(1)) for file in existing_files if (match := pattern.match(file))]
    #가장 큰 번호에 +1, 새로운 파일 번호 저장
    next_number = max(numbers, default=0) + 1
    #최종적으로 저장한 전체 경로 반환 
    return os.path.join(desktop_path, f"{base_name}_{next_number}.{ext}")

# 카메라 연결
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라 열기 실패")
    exit()

# 변수 초기화
#녹화상태 저장
recording = False
# 영상 저장용 객체
out = None
# 영상 압축 코덱 지정, MJPG가 범용적이고 호환성 좋으므로 MJPG 형식으로 비디오 저장장
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# 연속 사진 촬영 중인지 여부를 저장
photo_shooting = False
# 총 몇 장을 촬영할 것인지 count
photo_count = 0
# 지금까지 몇 장을 찍었는 지 저장
photo_taken = 0
# 사진을 찍는 간격 설정 / 5초
photo_interval = 5
# 카운트 다운을 시작한 시간(기준이 되는 시각)
countdown_start_time = 0

# 조작 설명
print("실행 중: 's'=녹화 시작, 'e'=녹화 종료, 숫자=연속 사진촬영, 'q'=종료")

# 메인 루프
# 카메라에서 한 프레임씩 읽어온다.
while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 읽기 실패")
        break
    # 현재 시각을 저장    
    now = time.time()
    # 사용자의 키보드 입력 1ms대기 후 받는다
    key = cv2.waitKey(1) & 0xFF

    # 녹화 종료 (항상 체크)
    if key == ord('e') and recording:
        print("녹화 종료! 영상이 저장되었습니다.")
        # 녹화상태 종료
        recording = False
        # 비디오 파일 저장 종료
        out.release()
        out = None

    # 's'눌렀을 때 녹화 시작
    # 이미 촬영 중이거나 연속촬영 중에는 무시
    if key == ord('s') and not recording and not photo_shooting:
        # 영상 파일명 자동 생성
        output_path = get_new_filename()
        print(f"녹화 시작! 저장 파일명: {os.path.basename(output_path)}")
        # 프레임의 높이와 너비
        height, width = frame.shape[:2]
        # 새로운 비디오 파일 생성, 저장
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
        if not out.isOpened():
            print("VideoWriter 열기 실패")
            break
        # 녹화 상태 시작으로 지정
        recording = True

    # 영상 녹화 중이면 저장
    if recording:
        out.write(frame)
        # 녹화중 표시 화면에 띄우기기
        cv2.putText(frame, "Recording...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 숫자키로 연속 촬영 시작
    # 숫자 키가 눌렸고 아직 연속촬영 상태가 아니었다면 시작!
    if ord('0') < key <= ord('9') and not photo_shooting:
        # 입력된 숫자 키를 정수로 반환, 찍은 총 활영 장 수로
        photo_count = key - ord('0')
        # 지금까지 찍은 사진 수 0으로 초기화
        photo_taken = 0
        # 카운트 다운한 시간을 현재 시각으로 기준
        countdown_start_time = now
        # 사진 촬영 모드로 전환
        photo_shooting = True
        print(f"{photo_count}장의 사진 연속 촬영 시작!")

    # 연속 촬영 중
    if photo_shooting:
        # 카운트 다운 경과 시간 계산
        elapsed = now - countdown_start_time
        # 남은 시간 계산
        seconds_left = photo_interval - int(elapsed)
        # 글자가 없는 상태 프레임을 저장장
        clean_frame = frame.copy()  # 저장용 클린 프레임

        # 좌측 상단: 카운트다운 표시 / 5, 4, 3, 2, 1...
        if seconds_left > 0:
            cv2.putText(frame, f"{seconds_left}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
        elif seconds_left == 0 and elapsed < photo_interval + 1:
            # 0초일 때 chesse~! 출력력
            cv2.putText(frame, "Cheese~!", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)

        # 우측 상단: 남은 촬영 수 표시
        shots_left = photo_count - photo_taken
        # 남은 촬영 장 수 출력
        cv2.putText(frame, f"{shots_left}", (frame.shape[1] - 60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3)

        # 촬영 타이밍 도달 시
        if elapsed >= photo_interval:
            # 사진 파일명을 자동 생성
            filename = get_new_picture_filename()
            # 글자가 없는 프레임을 파일로 저장
            cv2.imwrite(filename, clean_frame)
            # 촬영된 사진 수 1 증가시킴
            photo_taken += 1
            print(f"{photo_taken}번째 저장됨: {os.path.basename(filename)}")

            # 모든 촬영이 끝났는 지 확인인
            if photo_taken >= photo_count:
                photo_shooting = False
                print("연속 사진 촬영 완료!")
            else:
                countdown_start_time = now  # 다음 사진 타이머 리셋

    # 화면 출력
    cv2.imshow('Live Camera', frame)

    # 종료
    if key == ord('q'):
        print("프로그램 종료")
        break

# 자원 정리
cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
