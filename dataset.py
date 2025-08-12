import os
import glob
import json
import cv2
import torch
from torch.utils.data import Dataset
import os, json, cv2, numpy as np, torch
from torch.utils.data import Dataset

def _as_roi(roi):
    # roi가 [x1,y1,x2,y2]이거나 {"0":860,"1":170,"2":1200,"3":450} 같은 dict일 수도 있음
    if isinstance(roi, dict):
        vals = []
        for k in ["0", "1", "2", "3", 0, 1, 2, 3]:
            if k in roi:
                vals.append(int(roi[k]))
        roi = vals[:4]
    if isinstance(roi, (list, tuple)) and len(roi) >= 4:
        x1, y1, x2, y2 = map(int, roi[:4])
        return (x1, y1, x2, y2)
    return None

def _clamp_roi(roi, w, h):
    x1, y1, x2, y2 = roi
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return (x1, y1, x2, y2)

def _extract_segments_from_event_frame(event_frame, fps, event_length, seq_len, total_frames):
    """event_frame이 [s,e] 이면 그대로, s만 있으면 event_length(초)로 길이 보강"""
    segs = []
    if isinstance(event_frame, list):
        # [s,e] or [[s,e], [s2,e2] ...] or [s1,e1,s2,e2,...]
        if all(isinstance(x, int) for x in event_frame):
            if len(event_frame) == 2:
                s, e = sorted(event_frame)
                segs.append((s, e))
            elif len(event_frame) > 2:
                it = iter(event_frame)
                for s, e in zip(it, it):
                    s, e = int(s), int(e)
                    if s > e: s, e = e, s
                    segs.append((s, e))
        elif all(isinstance(x, (list, tuple)) and len(x) >= 2 for x in event_frame):
            for pair in event_frame:
                s, e = int(pair[0]), int(pair[1])
                if s > e: s, e = e, s
                segs.append((s, e))
    elif isinstance(event_frame, int):
        s = event_frame
        if event_length and fps and fps > 0:
            e = s + int(round(float(event_length) * float(fps)))
        else:
            e = s + seq_len - 1
        segs.append((s, e))

    # 유효 범위로 클램프
    out = []
    for s, e in segs:
        s = max(0, s)
        e = min(total_frames - 1, e)
        if s <= e:
            out.append((s, e))
    return out

class MultiVideoAnomalyDataset(Dataset):
    """
    label_dir의 각 JSON을 읽고:
      - metadata.file_name 에서 비디오 경로 생성
      - annotations.event_frame -> 이상 구간(segments)로 변환
      - roi -> [x1,y1,x2,y2]
    시퀀스를 슬라이딩으로 만들고, 구간과 겹치면 label=1.0, 아니면 0.0
    __getitem__은 [T,C,H,W] float(0~1), label(float32) 반환
    """
    def __init__(self, label_dir, seq_len=8, default_roi=(1150,300,1600,700),
                 transform=None, stride=1, resize=(112, 112)):
        self.seq_len = seq_len
        self.default_roi = tuple(default_roi)
        self.transform = transform
        self.resize = resize
        self.samples = []  # (video_path, start_frame, label, roi)

        json_files = [f for f in os.listdir(label_dir) if f.lower().endswith('.json')]
        json_files.sort()

        for jf in json_files:
            jpath = os.path.join(label_dir, jf)
            with open(jpath, 'r', encoding='utf-8') as f:
                meta = json.load(f)

            # 1) 비디오 경로
            md = meta.get("metadata", {})
            file_name = md.get("file_name") or meta.get("file_name")
            if not file_name:
                raise ValueError(f"[{jf}] metadata.file_name 이 없습니다.")
            video_path = file_name if os.path.isabs(file_name) else os.path.join(os.path.dirname(jpath), file_name)
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"[{jf}] 비디오 파일을 찾을 수 없습니다: {video_path}")

            # 2) 비디오 기본 정보
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"비디오를 열 수 없습니다: {video_path}")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else None
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            # 3) ROI
            roi = _as_roi(meta.get("roi"))
            if roi is None:
                roi = self.default_roi
            roi = _clamp_roi(roi, width, height)

            # 4) 이상 구간(segments) 생성
            ann = meta.get("annotations", {})
            event_frame = ann.get("event_frame")
            event_length = ann.get("event_length")
            segments = _extract_segments_from_event_frame(event_frame, fps, event_length,
                                                          self.seq_len, total_frames)

            # 5) 시퀀스 샘플 생성 (슬라이딩 윈도우)
            if total_frames <= 0:
                continue
            last_start = max(0, total_frames - self.seq_len)
            for st in range(0, last_start + 1, stride):
                ed = st + self.seq_len - 1
                # 구간과 겹치면 이상(1.0)
                is_abnormal = any(not (ed < a or st > b) for (a, b) in segments) if segments else False
                lab = 1.0 if is_abnormal else 0.0
                self.samples.append((video_path, st, lab, roi))

        # 프레임 읽을 때 다시 클램프할 필요가 없도록 캐시만 준비
        self._size_cache = {}

    def __len__(self):
        return len(self.samples)

    def _read_seq(self, video_path, start, roi, count):
        x1, y1, x2, y2 = roi
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(count):
            ok, frame = cap.read()
            if not ok:
                break
            crop = frame[y1:y2, x1:x2]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            if self.resize is not None:
                crop = cv2.resize(crop, self.resize, interpolation=cv2.INTER_LINEAR)
            frames.append(crop)
        cap.release()

        # 끝에서 모자라면 마지막 프레임 반복
        if len(frames) == 0:
            h, w = (self.resize[1], self.resize[0]) if self.resize else (y2 - y1, x2 - x1)
            frames = [np.zeros((h, w, 3), dtype=np.uint8)]
        while len(frames) < count:
            frames.append(frames[-1])

        arr = np.stack(frames, axis=0)          # [T,H,W,3]
        arr = arr.transpose(0, 3, 1, 2)         # [T,3,H,W]
        tensor = torch.from_numpy(arr).float() / 255.0
        return tensor

    def __getitem__(self, idx):
        video_path, start, label, roi = self.samples[idx]
        seq = self._read_seq(video_path, start, roi, self.seq_len)
        label = torch.tensor(label, dtype=torch.float32)
        return seq, label




import os
import json
import cv2
import torch
from torch.utils.data import Dataset

class SingleVideoAnomalyDataset(Dataset):
    """
    Map-style Dataset for a single video anomaly detection using one JSON annotation with ROI cropping.

    Args:
        ann_path (str): Path to the JSON annotation file (e.g., '/.../label/E01_001.json').
        seq_len  (int): Number of consecutive frames per sequence.
        fps      (int): Frames per second rate (unused for frame-index labels).
        roi      (tuple): (x1, y1, x2, y2) Region of interest to crop in each frame.
        transform (callable, optional): Transform applied to each returned sequence tensor.
    """
    def __init__(self, ann_path, seq_len=8, fps=8,
                 roi=(1150, 300, 1600, 700), transform=None):
        super().__init__()
        self.seq_len = seq_len
        self.fps = fps  # fps는 event_frame이 초단위가 아닐 경우에만 사용
        self.transform = transform
        self.roi = roi  # (x1, y1, x2, y2)
        self.samples = []  # List of (video_path, start_frame_idx, label)

        # Load JSON annotation
        with open(ann_path, 'r') as f:
            item = json.load(f)

        # Derive video path from annotation path
        label_dir   = os.path.dirname(ann_path)
        project_dir = os.path.dirname(label_dir)
        video_dir   = os.path.join(project_dir, 'video')
        base_name   = os.path.splitext(os.path.basename(ann_path))[0]
        video_path  = os.path.join(video_dir, base_name + '.mp4')

        # Get total frame count
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # event_frame 필드는 프레임 인덱스 범위를 직접 제공
        anomaly_frames = []
        for start_idx, end_idx in item['annotations']['event_frame']:
            s = int(start_idx)
            e = int(end_idx)
            anomaly_frames.append((s, e))

        # Generate sliding window samples
        for start in range(0, frame_count - seq_len + 1):
            end = start + seq_len - 1
            label = 0.0
            for s, e in anomaly_frames:
                if not (end < s or start > e):
                    label = 1.0
                    break
            self.samples.append((video_path, start, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, start_idx, label = self.samples[idx]
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

        frames = []
        x1, y1, x2, y2 = self.roi
        for _ in range(self.seq_len):
            ret, frame = cap.read()
            if not ret:
                break
            # Crop ROI
            crop = frame[y1:y2, x1:x2]
            # Resize to 256×256
            patch = cv2.resize(crop, (256, 256))
            # Convert BGR→RGB and to tensor
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
            frames.append(tensor)
        cap.release()

        seq = torch.stack(frames, dim=0)  # (T, C, 256, 256)
        if self.transform:
            seq = self.transform(seq)
        return seq, torch.tensor(label, dtype=torch.float)
