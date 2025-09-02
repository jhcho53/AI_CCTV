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

def _merge_segments(segments, gap_tolerance=200):
    """
    인접/겹침 세그먼트를 병합합니다.
    gap_tolerance(프레임) 이내로 떨어진 구간은 하나로 합칩니다.
    (예) [10,20] 다음 [22,30], gap=1 → 병합 / gap=2 → 병합 / gap=3 → 미병합
    """
    if not segments:
        return []
    segments = sorted(segments)  # 안전용 정렬
    merged = [segments[0]]
    for s, e in segments[1:]:
        ps, pe = merged[-1]
        # gap = 다음 시작 - 이전 끝 - 1, inclusive 프레임 기준
        if s <= pe + gap_tolerance + 1:  # 겹치거나(gap<0) 인접/근접(gap<=gap_tolerance)
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def extract_segments(probs, threshold=0.5, merge_gap=200, min_len=1):
    """
    프레임별 확률에서 연속된 이상 구간을 추출하고,
    작은 끊김(gap)들은 merge_gap 프레임 허용치 이내면 병합합니다.
    반환: [(start_frame, end_frame)]  (둘 다 inclusive)
    """
    segments = []
    in_anom = False
    start = 0

    for i, p in enumerate(probs):
        if p >= threshold:
            if not in_anom:
                in_anom = True
                start = i
        else:
            if in_anom:
                end = i - 1  # 마지막으로 threshold 이상이었던 프레임
                if end >= start:
                    segments.append((start, end))
                in_anom = False

    # 마지막이 이상 상태로 끝난 경우 마무리
    if in_anom:
        segments.append((start, len(probs) - 1))

    # 최소 길이 필터링(옵션)
    if min_len > 1:
        segments = [(s, e) for (s, e) in segments if (e - s + 1) >= min_len]

    # 근접 구간 병합
    if merge_gap is not None and merge_gap >= 0:
        segments = _merge_segments(segments, gap_tolerance=merge_gap)

    return segments

def save_cropped_segments(
    video_path, segments, output_pattern,
    roi, fps, fourcc_str='mp4v'
):
    """
    각 세그먼트의 ROI를 잘라 개별 MP4로 저장합니다.
    segments는 (start_frame, end_frame) inclusive.
    """
    import os
    import cv2

    x1, y1, x2, y2 = map(int, roi)
    w, h = x2 - x1, y2 - y1
    assert w > 0 and h > 0, f"Invalid ROI size: {(w, h)}"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    out_paths = []

    for idx, (s, e) in enumerate(segments):
        if e < s:
            continue
        out_path = output_pattern.format(idx=idx)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create writer: {out_path}")

        # 시크 후 쓰기
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        written = 0
        for f in range(s, e + 1):
            ret, frame = cap.read()
            if not ret:
                break
            crop = frame[y1:y2, x1:x2]
            writer.write(crop)
            written += 1

        writer.release()

        if written > 0:
            start_time = s / fps
            end_time = e / fps
            out_paths.append(out_path)
            print(f"Saved {out_path} [{s}–{e}] ({start_time:.2f}s–{end_time:.2f}s)")
        else:
            # 쓰인 프레임이 없으면 파일 제거(선택)
            try:
                os.remove(out_path)
            except Exception:
                pass

    cap.release()
    return out_paths

def run_inference(
    video_path,
    model_path='anomaly_single_balanced.pth',
    seq_len=8,
    roi=None,                      # 기본: ROI 없이(전체 프레임)
    threshold=0.1,
    # VLM 인자
    vlm_model=None,
    vlm_tokenizer=None,
    generation_config=None,
    vlm_segments=8,
    vlm_max_num=1
):
    """
    roi 기본값을 None으로 두고, None이면 전체 프레임을 사용합니다.
    detect_anomaly_probs / save_cropped_segments 에는 유효한 ROI 튜플을 전달합니다.
    """

    # 유효 ROI 결정
    if roi is None:
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                roi_eff = (0, 0, w, h)  # 전체 프레임
            else:
                # 열기 실패 시, 하위 함수가 None을 허용한다면 None 전달
                roi_eff = None
        except Exception:
            roi_eff = None
    else:
        roi_eff = roi

    # 1) 이상 탐지
    model, device = load_model(model_path)
    probs, fps = detect_anomaly_probs(video_path, model, device, seq_len, roi_eff)
    segments = extract_segments(probs, threshold)
    pattern = 'anomaly_segment_{idx}.mp4'
    out_paths = save_cropped_segments(video_path, segments, pattern, roi_eff, fps)

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