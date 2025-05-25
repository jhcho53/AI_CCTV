import cv2
import torch
import numpy as np
from model import GRUAnomalyDetector

# 설정
video_path = '/home/vip1/Desktop/SSDC/neubility/gd/BTS_graduate_project/Training/video/E01_001.mp4'
MODEL_PATH = 'anomaly_single_balanced.pth'
SEQ_LEN     = 8
STRIDE      = 1
THRESHOLD   = 0.5  # 이상 확률 임계값

# 1) 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = GRUAnomalyDetector().to(device)
state  = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.eval()

# 2) 비디오 열기
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 3) 프레임 버퍼링 & 시퀀스별 확률 계산
anomaly_probs = np.zeros(total_frames)  # 프레임 단위 score
buffer = []

# Read all frames and preprocess crop & resize as in your Dataset
fps = cap.get(cv2.CAP_PROP_FPS)
# ROI 설정 (예시)
x1, y1, x2, y2 = 1150, 300, 1600, 700

for idx in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break

    # Crop → Resize → To Tensor
    crop  = frame[y1:y2, x1:x2]
    patch = cv2.resize(crop, (256, 256))
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(patch).permute(2,0,1).float().unsqueeze(0) / 255.0  # (1,3,256,256)
    buffer.append(tensor)

    # Once we have SEQ_LEN in buffer, run inference
    if len(buffer) == SEQ_LEN:
        seq = torch.cat(buffer, dim=0).unsqueeze(0).to(device)  # (1,8,3,256,256)
        with torch.no_grad():
            prob = model(seq).item()  # scalar
        # Assign this score to the **last** frame of the window
        anomaly_probs[idx] = prob

        # slide
        buffer.pop(0)

cap.release()

# 4) 연속된 이상 프레임 구간 추출
segments = []
in_anomaly = False
start_idx  = None

for i, p in enumerate(anomaly_probs):
    if p >= THRESHOLD and not in_anomaly:
        in_anomaly = True
        start_idx  = i
    elif (p < THRESHOLD or i == total_frames-1) and in_anomaly:
        end_idx    = i if p < THRESHOLD else i
        segments.append((start_idx, end_idx))
        in_anomaly = False

# 5) 결과 출력
print("Detected anomaly segments (frame indices):")
for s, e in segments:
    t_s = s / fps
    t_e = e / fps
    print(f"  Frames {s} → {e}  (Time {t_s:.2f}s → {t_e:.2f}s)")
