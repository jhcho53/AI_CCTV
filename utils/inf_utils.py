import cv2
import torch
import numpy as np
from model.model import GRUAnomalyDetector
from utils.video_vlm import run_video_inference
import time

def load_model(model_path, device=None):
    """
    Load the GRU anomaly detector model.
    Returns (model, device).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GRUAnomalyDetector().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device

def detect_anomaly_probs(
    video_path, model, device,
    seq_len=8, roi=(0,0,256,256)
):
    """
    Process the entire video and return per-frame anomaly probabilities.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    x1, y1, x2, y2 = roi
    probs = np.zeros(total, dtype=float)
    buffer = []

    for idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        crop = frame[y1:y2, x1:x2]
        patch = cv2.resize(crop, (256,256))
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(patch).permute(2,0,1).float().unsqueeze(0) / 255.0
        buffer.append(tensor)
        if len(buffer) == seq_len:
            seq = torch.cat(buffer, dim=0).unsqueeze(0).to(device)
            with torch.no_grad(): prob = model(seq).item()
            probs[idx] = prob
            buffer.pop(0)

    cap.release()
    return probs, fps

def extract_segments(probs, threshold=0.5):
    """
    Given per-frame probabilities, extract contiguous anomaly segments.
    Returns list of (start_frame, end_frame).
    """
    segments = []
    in_anom = False
    start = 0
    for i, p in enumerate(probs):
        if p >= threshold and not in_anom:
            in_anom = True
            start = i
        elif (p < threshold or i == len(probs)-1) and in_anom:
            end = i
            segments.append((start, end))
            in_anom = False
    return segments

def save_cropped_segments(
    video_path, segments, output_pattern,
    roi, fps, fourcc_str='mp4v'
):
    """
    Save each segment's cropped ROI as a separate MP4 file.
    output_pattern: a format string like 'segment_{idx}.mp4'.
    Returns list of saved file paths.
    """
    x1, y1, x2, y2 = roi
    w, h = x2 - x1, y2 - y1
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)

    out_paths = []
    for idx, (s,e) in enumerate(segments):
        start_time = s / fps
        end_time   = e / fps
        out_path = output_pattern.format(idx=idx)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w,h))
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        for f in range(s, e+1):
            ret, frame = cap.read()
            if not ret: break
            crop = frame[y1:y2, x1:x2]
            writer.write(crop)
        writer.release()
        out_paths.append(out_path)
        print(f'Saved {out_path} [{s}–{e}]')
        print(f'Saved {out_path} [{s}–{e}] ({start_time:.2f}s–{end_time:.2f}s)')
    cap.release()
    return out_paths

def run_inference(
    video_path,
    model_path='anomaly_single_balanced.pth',
    seq_len=8,
    roi=(1150,300,1600,700),
    threshold=0.5,
    # VLM 인자 추가
    vlm_model=None,
    vlm_tokenizer=None,
    generation_config=None,
    vlm_segments=8,
    vlm_max_num=1
):
    # 1) 이상 탐지
    model, device = load_model(model_path)
    probs, fps = detect_anomaly_probs(video_path, model, device, seq_len, roi)
    segments = extract_segments(probs, threshold)
    pattern = 'anomaly_segment_{idx}.mp4'
    out_paths = save_cropped_segments(video_path, segments, pattern, roi, fps)

    # 2) VLM 추론
    results = {}
    for seg_path in out_paths:
        qa = run_video_inference(
            model=vlm_model,
            tokenizer=vlm_tokenizer,
            video_path=seg_path,
            generation_config=generation_config,
            num_segments=vlm_segments,
            max_num=vlm_max_num
        )
        results[seg_path] = qa
    return results
