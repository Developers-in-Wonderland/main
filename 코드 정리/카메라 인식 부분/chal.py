import cv2
import os
import re
# opencv / os(경로 및 파일 관련 작업용) / re(파일 이름에서 숫자 뽑아내는데 사용) import

# 사용자의 바탕화면으로로 경로 설정
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

# 영상 저장 경로 자동 이름 증가를 위한 함수
# ex) output1, output2 이런식으로
def get_new_filename(base_name="output", ext="avi"):
    #바탕화면에 있는 모든 파일 목록을 불러온 뒤
    existing_files = os.listdir(desktop_path)
    #파일 이름 중 outpur_숫자.avi형식에 맞는 것만 찾기 위한 '정규표현식'정의
    pattern = re.compile(rf"{re.escape(base_name)}_(\d+)\.{re.escape(ext)}")
    
	#output_1.avi, output_2.avi 등 숫자만 추출해서 리스트에 저장
    #매칭 동시에 변수에 저장
    numbers = [int(match.group(1)) for file in existing_files if (match := pattern.match(file))]
    #가장 큰 번호에 +1, 새로운 파일 번호 저장
    next_number = max(numbers, default=0) + 1
    #최종적으로 저장한 전체 경로 반환 
    #ex)C:/Users/민주홍/Desktop/output_3.avi
    return os.path.join(desktop_path, f"{base_name}_{next_number}.{ext}")

# 카메라 연결결
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라 열기 실패")
    exit()

# 녹화 상태 저장
recording = False
# 영상 저장용 객체
out = None
# 영상 압축 코덱 지정, MJPG가 범용적이고 호환성 좋다
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

print("실행 중: 's' = 녹화 시작, 'e' = 녹화 종료, 'q' = 프로그램 종료")

#카메라가 꺼지기 전까지 무한루프
while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 읽기 실패")
        break

	# 키 입력 대기, &0xFF는 윈도우용 안전 처리로, 눌린 키 코드를 읽어온다다	
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and not recording:
        #자동으로 새로운 파일 이름 생성
        output_path = get_new_filename()
        print(f"녹화 시작! 저장 파일명: {os.path.basename(output_path)}")
        height, width = frame.shape[:2]
        #영상 저장을 위한 Videowriter 객체 생성성
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

        if not out.isOpened():
            print("VideoWriter 열기 실패")
            break
        recording = True

    if recording:
        out.write(frame)
        #녹화중임을 알 수 있게 화면 상단에 현 상태 보여줌줌
        cv2.putText(frame, "Recording...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if key == ord('e') and recording:
        print("녹화 종료! 영상이 저장되었습니다.")
        recording = False
        out.release()
        out = None

    cv2.imshow('Live Camera', frame)

    if key == ord('q'):
        print("프로그램 종료")
        break

#모든 자원 정리
cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
