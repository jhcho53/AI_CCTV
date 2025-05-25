import os
import glob
import json
import cv2
import torch
from torch.utils.data import Dataset

class VideoAnomalyDataset(Dataset):
    def __init__(self, label_dir, seq_len=8, fps=8, transform=None):
        """
        label_dir: 경로/.../Training/label/
        seq_len:  시퀀스 길이 (프레임 수)
        fps:      비디오 프레임 레이트
        """
        self.seq_len   = seq_len
        self.fps       = fps
        self.transform = transform
        self.samples   = []

        # 모든 JSON 파일을 순회
        json_paths = glob.glob(os.path.join(label_dir, '*.json'))
        for ann_path in json_paths:
            # JSON 읽기
            with open(ann_path, 'r') as f:
                item = json.load(f)

            # video 폴더 경로 계산
            label_dir_parent = os.path.dirname(label_dir)      # .../Training
            video_dir        = os.path.join(label_dir_parent, 'video')
            base_name        = os.path.splitext(os.path.basename(ann_path))[0]
            video_path       = os.path.join(video_dir, base_name + '.mp4')

            # 프레임 총개수
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # 이상 구간(초→프레임 인덱스)
            anomaly_frames = []
            for s_sec, e_sec in item['annotations']['event_frame']:
                s_idx = int(s_sec * fps)
                e_idx = int(e_sec * fps)
                anomaly_frames.append((s_idx, e_idx))

            # 슬라이딩 윈도우별 샘플 생성
            for i in range(frame_count - seq_len + 1):
                label = 0
                for (s, e) in anomaly_frames:
                    if not (i + seq_len - 1 < s or i > e):
                        label = 1
                        break
                self.samples.append((video_path, i, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, start_idx, label = self.samples[idx]
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

        frames = []
        for _ in range(self.seq_len):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (128, 128))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames.append(t)
        cap.release()

        seq = torch.stack(frames, dim=0)  # (T, C, H, W)
        if self.transform:
            seq = self.transform(seq)
        return seq, torch.tensor(label, dtype=torch.float)
