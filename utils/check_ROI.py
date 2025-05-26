import cv2

# 1) 입력 비디오 경로와 ROI (x1, y1, x2, y2) 지정
video_path = '/home/vip1/Desktop/SSDC/neubility/gd/BTS_graduate_project/Training/video/E01_001.mp4'
roi = (1150, 300, 1600, 700)

# 2) 비디오 열고 첫 프레임 읽기
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError(f"Cannot read first frame from {video_path}")

# 3) 박스 그리기
x1, y1, x2, y2 = roi
boxed = frame.copy()
cv2.rectangle(boxed, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=3)

# 4) 파일로 저장
cv2.imwrite('first_frame_with_roi.png', boxed)
