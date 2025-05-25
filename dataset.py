import os
import glob
import json
import cv2
import torch
from torch.utils.data import IterableDataset

class VideoAnomalyIterableDataset(IterableDataset):
    def __init__(self, label_dir, seq_len=8, fps=8, transform=None):
        """
        IterableDataset for video anomaly detection.

        label_dir: Path to .../Training/label/ containing JSON annotation files.
        seq_len: Number of consecutive frames per sequence.
        fps: Sampling rate (frames per second) for converting seconds to frame indices.
        transform: Optional tensor-level transform applied to each sequence.
        """
        super().__init__()
        self.label_dir = label_dir
        self.seq_len = seq_len
        self.fps = fps
        self.transform = transform
        # List all JSON annotation files
        self.json_paths = glob.glob(os.path.join(label_dir, '*.json'))

    def __iter__(self):
        # Iterate over each annotation file lazily
        for ann_path in self.json_paths:
            # Load annotation JSON
            with open(ann_path, 'r') as f:
                item = json.load(f)
            # Determine corresponding video path
            base_dir = os.path.dirname(self.label_dir)
            video_dir = os.path.join(base_dir, 'video')
            base_name = os.path.splitext(os.path.basename(ann_path))[0]
            video_path = os.path.join(video_dir, base_name + '.mp4')

            # Open video to get frame count
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Convert anomaly time segments to frame indices
            anomaly_frames = []
            for start_sec, end_sec in item['annotations']['event_frame']:
                start_idx = int(start_sec * self.fps)
                end_idx = int(end_sec   * self.fps)
                anomaly_frames.append((start_idx, end_idx))

            # Generate sequences with non-overlapping stride = seq_len
            for start in range(0, frame_count - self.seq_len + 1, self.seq_len):
                end = start + self.seq_len - 1
                # Determine label: 1 if any frame overlaps anomaly
                label = 0
                for s, e in anomaly_frames:
                    if not (end < s or start > e):
                        label = 1
                        break
                # Read frames for this sequence
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                frames = []
                for _ in range(self.seq_len):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame, (128, 128))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                    frames.append(tensor)
                # Yield only complete sequences
                if len(frames) == self.seq_len:
                    seq = torch.stack(frames, dim=0)  # Shape: (T, C, H, W)
                    if self.transform:
                        seq = self.transform(seq)
                    yield seq, torch.tensor(label, dtype=torch.float)
            cap.release()