import cv2
import matplotlib.pyplot as plt

# 1) 입력 비디오 경로와 ROI (x1, y1, x2, y2) 지정
video_path = '/home/jaehyeon/Desktop/졸작/BTS_graduate_project/Dataset/Video/E02_001.mp4'
roi = (550, 100, 800, 250)

# 2) 비디오 열고 첫 프레임 읽기
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 2000) 
ret, frame = cap.read()
cap.release()
if not ret:
    raise RuntimeError(f"Cannot read first frame from {video_path}")

# 3) ROI 박스 그리기
x1, y1, x2, y2 = roi
boxed = frame.copy()
cv2.rectangle(boxed,
              (x1, y1),
              (x2, y2),
              color=(0, 0, 255),    # 빨간색 BGR
              thickness=3)
# 4) Matplotlib으로 시각화
plt.figure(figsize=(10, 6))
# OpenCV는 BGR, Matplotlib은 RGB이므로 변환
plt.imshow(cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
