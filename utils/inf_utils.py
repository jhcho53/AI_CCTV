import os
import cv2
import torch
import numpy as np
import tempfile
from typing import List, Tuple, Dict, Optional

from model.model import GRUAnomalyDetector

# VLM: prefer frame-direct inference (run_frames_inference); fallback to run_video_inference
try:
    from utils.video_vlm import run_video_inference, run_frames_inference  # type: ignore
except Exception:
    from utils.video_vlm import run_video_inference  # type: ignore
    run_frames_inference = None  # frame-direct input not available


# =========================
# 1) Model loading (cached)
# =========================

_MODEL_CACHE: Dict[str, Tuple[torch.nn.Module, torch.device]] = {}


def load_model(model_path: str, device: Optional[torch.device] = None) -> Tuple[torch.nn.Module, torch.device]:
    """
    Load the GRU anomaly detector model with caching.

    Args:
        model_path: Path to the model weights (.pth).
        device: torch.device to place the model on. If None, picks CUDA if available, else CPU.

    Returns:
        (model, device)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_key = f"{os.path.abspath(model_path)}::{device.type}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model = GRUAnomalyDetector().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    _MODEL_CACHE[cache_key] = (model, device)
    return model, device


# =========================
# 2) Common utilities
# =========================

def _ensure_roi_from_first_frame(frames: List[np.ndarray], roi: Optional[Tuple[int, int, int, int]]):
    """
    Return ROI as a tuple. If roi is None, use the full extent of the first frame.
    """
    if roi is not None:
        return tuple(map(int, roi))
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    return (0, 0, w, h)


def _ensure_roi_from_video(video_path: str, roi: Optional[Tuple[int, int, int, int]]):
    """
    Return ROI as a tuple. If roi is None, probe the video size and use the full frame.
    """
    if roi is not None:
        return tuple(map(int, roi))
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        if w > 0 and h > 0:
            return (0, 0, w, h)
    return None


def _preprocess_patch_bgr(frame_bgr: np.ndarray, roi: Tuple[int, int, int, int]) -> torch.Tensor:
    """
    Crop frame to ROI (clamped to image boundaries), resize to 256x256, convert to RGB tensor in [0,1].

    Returns:
        Tensor of shape (1, 3, 256, 256)
    """
    H, W = frame_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, roi)
    # Clamp ROI to frame bounds
    x1 = max(0, min(W, x1))
    y1 = max(0, min(H, y1))
    x2 = max(0, min(W, x2))
    y2 = max(0, min(H, y2))
    if x2 <= x1 or y2 <= y1:
        crop = frame_bgr  # fallback: invalid ROI → use full frame
    else:
        crop = frame_bgr[y1:y2, x1:x2]
    patch = cv2.resize(crop, (256, 256))
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(patch).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return tensor  # (1, 3, 256, 256)


# =========================
# 3) Anomaly probabilities
# =========================

def detect_anomaly_probs_from_frames(
    frames: List[np.ndarray],
    model: torch.nn.Module,
    device: torch.device,
    seq_len: int = 8,
    roi: Optional[Tuple[int, int, int, int]] = (0, 0, 256, 256),
) -> np.ndarray:
    """
    Compute per-frame anomaly probabilities from a list of frames.
    Processing mirrors detect_anomaly_probs(video_path, ...).

    Returns:
        probs: np.ndarray of shape (len(frames),) with probabilities in [0,1].
    """
    total = len(frames)
    roi_eff = _ensure_roi_from_first_frame(frames, roi)
    if roi_eff is None or total == 0:
        return np.zeros(total, dtype=float)

    probs = np.zeros(total, dtype=float)
    buffer: List[torch.Tensor] = []

    for idx, frame in enumerate(frames):
        tensor = _preprocess_patch_bgr(frame, roi_eff)  # (1,3,256,256)
        buffer.append(tensor)
        if len(buffer) == seq_len:
            seq = torch.cat(buffer, dim=0).unsqueeze(0).to(device)  # (1, seq, 3, 256, 256)
            with torch.no_grad():
                prob = model(seq).item()
            probs[idx] = prob
            buffer.pop(0)

    return probs


def detect_anomaly_probs(
    video_path: str,
    model: torch.nn.Module,
    device: torch.device,
    seq_len: int = 8,
    roi: Optional[Tuple[int, int, int, int]] = (0, 0, 256, 256),
) -> Tuple[np.ndarray, float]:
    """
    Compute per-frame anomaly probabilities by reading a video file directly.

    Returns:
        (probs, fps)
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-3:
        fps = 30.0

    if roi is None:
        # Full-frame ROI
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        roi_eff = (0, 0, w, h) if w > 0 and h > 0 else None
    else:
        roi_eff = tuple(map(int, roi))

    probs = np.zeros(total, dtype=float)
    buffer: List[torch.Tensor] = []

    for idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if roi_eff is None:
            probs[idx] = 0.0
            continue
        tensor = _preprocess_patch_bgr(frame, roi_eff)  # (1,3,256,256)
        buffer.append(tensor)
        if len(buffer) == seq_len:
            seq = torch.cat(buffer, dim=0).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = model(seq).item()
            probs[idx] = prob
            buffer.pop(0)

    cap.release()
    return probs, fps


# =========================
# 4) Segment extraction/merge
# =========================

def _merge_segments(segments: List[Tuple[int, int]], gap_tolerance: int = 200) -> List[Tuple[int, int]]:
    """
    Merge adjacent/overlapping segments. If the gap between segments is within
    gap_tolerance (in frames), merge them into one.
    """
    if not segments:
        return []
    segments = sorted(segments)
    merged = [segments[0]]
    for s, e in segments[1:]:
        ps, pe = merged[-1]
        if s <= pe + gap_tolerance + 1:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def extract_segments(
    probs: np.ndarray,
    threshold: float = 0.5,
    merge_gap: int = 200,
    min_len: int = 1,
) -> List[Tuple[int, int]]:
    """
    Extract contiguous anomalous ranges from per-frame probabilities and merge them.

    Args:
        probs: per-frame probabilities.
        threshold: frames >= threshold are considered anomalous.
        merge_gap: maximum allowed gap (frames) between segments to merge them.
        min_len: minimum segment length (frames) to keep.

    Returns:
        List of segments as [(start_frame, end_frame)] inclusive.
    """
    segments: List[Tuple[int, int]] = []
    in_anom = False
    start = 0

    for i, p in enumerate(probs):
        if p >= threshold:
            if not in_anom:
                in_anom = True
                start = i
        else:
            if in_anom:
                end = i - 1
                if end >= start:
                    segments.append((start, end))
                in_anom = False

    if in_anom:
        segments.append((start, len(probs) - 1))

    if min_len > 1:
        segments = [(s, e) for (s, e) in segments if (e - s + 1) >= min_len]

    if merge_gap is not None and merge_gap >= 0:
        segments = _merge_segments(segments, gap_tolerance=merge_gap)

    return segments


# =========================
# 5) Segment cropping (file/frames)
# =========================

def save_cropped_segments(
    video_path: str,
    segments: List[Tuple[int, int]],
    output_pattern: str,
    roi: Tuple[int, int, int, int],
    fps: float,
    fourcc_str: str = "mp4v",
) -> List[str]:
    """
    Save each segment (inclusive frame range) into a cropped MP4 file.

    Args:
        output_pattern: e.g., 'anomaly_segment_{idx}.mp4'
    """
    x1, y1, x2, y2 = map(int, roi)
    w, h = x2 - x1, y2 - y1
    assert w > 0 and h > 0, f"Invalid ROI size: {(w, h)}"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    out_paths: List[str] = []

    for idx, (s, e) in enumerate(segments):
        if e < s:
            continue
        out_path = output_pattern.format(idx=idx)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create writer: {out_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        written = 0
        for _f in range(s, e + 1):
            ret, frame = cap.read()
            if not ret:
                break
            crop = frame[y1:y2, x1:x2]
            writer.write(crop)
            written += 1
        writer.release()

        if written > 0:
            start_time = s / max(fps, 1e-6)
            end_time = e / max(fps, 1e-6)
            out_paths.append(out_path)
            print(f"Saved {out_path} [{s}–{e}] ({start_time:.2f}s–{end_time:.2f}s)")
        else:
            try:
                os.remove(out_path)
            except Exception:
                pass

    cap.release()
    return out_paths


def collect_cropped_segments_frames(
    frames: List[np.ndarray],
    segments: List[Tuple[int, int]],
    roi: Tuple[int, int, int, int],
) -> List[List[np.ndarray]]:
    """
    Return cropped frame lists per segment without writing files.
    """
    x1, y1, x2, y2 = map(int, roi)
    w, h = x2 - x1, y2 - y1
    assert w > 0 and h > 0, f"Invalid ROI size: {(w, h)}"

    clips: List[List[np.ndarray]] = []
    for s, e in segments:
        if e < s or s < 0 or e >= len(frames):
            continue
        clip = []
        for i in range(s, e + 1):
            f = frames[i]
            crop = f[y1:y2, x1:x2]
            clip.append(crop.copy())
        if clip:
            clips.append(clip)
    return clips


# =========================
# 6) VLM: frames-first, fallback to video
# =========================

def _write_frames_to_temp_video(frames: List[np.ndarray], fps: float) -> str:
    """
    Write frames to a temporary MP4 (fallback) and return the path.
    """
    if not frames:
        raise ValueError("No frames to write.")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    path = tmp.name
    tmp.close()
    writer = cv2.VideoWriter(path, fourcc, max(fps, 1.0), (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return path


def run_vlm_on_frames(
    vlm_model,
    vlm_tokenizer,
    clip_frames: List[np.ndarray],
    generation_config=None,
    num_segments: int = 8,
    max_num: int = 1,
    fps_for_fallback: float = 30.0,
):
    """
    Run VLM on frames if supported; otherwise create a tiny temporary video and fallback to run_video_inference.
    """
    generation_config = generation_config or {"max_new_tokens": 512, "do_sample": True}

    # 1) Frame-direct path
    if run_frames_inference is not None:
        return run_frames_inference(
            model=vlm_model,
            tokenizer=vlm_tokenizer,
            frames=clip_frames,
            generation_config=generation_config,
            num_segments=num_segments,
            max_num=max_num,
        )

    # 2) Video fallback
    tmp_path = _write_frames_to_temp_video(clip_frames, fps_for_fallback)
    try:
        qa = run_video_inference(
            model=vlm_model,
            tokenizer=vlm_tokenizer,
            video_path=tmp_path,
            generation_config=generation_config,
            num_segments=num_segments,
            max_num=max_num,
        )
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return qa


# =========================
# 7) End-to-end: frames / file
# =========================

def run_inference_on_frames(
    frames: List[np.ndarray],
    model_or_tuple,  # (model, device) or a model path (str)
    seq_len: int = 8,
    roi: Optional[Tuple[int, int, int, int]] = None,
    threshold: float = 0.1,
    # VLM
    vlm_model=None,
    vlm_tokenizer=None,
    generation_config=None,
    vlm_segments: int = 8,
    vlm_max_num: int = 1,
    tag: str = "stream_clip",
    fps_for_vlm_fallback: float = 30.0,
    merge_gap: int = 200,
    min_len: int = 1,
) -> Dict[str, List[Tuple[str, str]]]:
    """
    End-to-end on in-memory frames: anomaly detection → segment extraction → VLM.

    Returns:
        { tag: [(Q, A), ...] }
    """
    if isinstance(model_or_tuple, str):
        model, device = load_model(model_or_tuple)
    else:
        model, device = model_or_tuple

    roi_eff = _ensure_roi_from_first_frame(frames, roi)
    probs = detect_anomaly_probs_from_frames(
        frames=frames,
        model=model,
        device=device,
        seq_len=seq_len,
        roi=roi_eff,
    )
    segments = extract_segments(probs, threshold=threshold, merge_gap=merge_gap, min_len=min_len)
    clips = collect_cropped_segments_frames(frames, segments, roi_eff)

    results: Dict[str, List[Tuple[str, str]]] = {}
    qa_all: List[Tuple[str, str]] = []
    for clip in clips:
        qa = run_vlm_on_frames(
            vlm_model=vlm_model,
            vlm_tokenizer=vlm_tokenizer,
            clip_frames=clip,
            generation_config=generation_config,
            num_segments=vlm_segments,
            max_num=vlm_max_num,
            fps_for_fallback=fps_for_vlm_fallback,
        )
        qa_all.extend(qa)

    results[tag] = qa_all
    return results


def run_inference(
    video_path: str,
    model_path: str = "anomaly_single_balanced.pth",
    seq_len: int = 8,
    roi: Optional[Tuple[int, int, int, int]] = None,  # None → full frame
    threshold: float = 0.1,
    # VLM
    vlm_model=None,
    vlm_tokenizer=None,
    generation_config=None,
    vlm_segments: int = 8,
    vlm_max_num: int = 1,
    merge_gap: int = 200,
    min_len: int = 1,
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Backward-compatible file-path entry: returns {segment_path: [(Q, A), ...]}.

    Pipeline:
      1) Determine ROI (full-frame if None)
      2) Load model (cached)
      3) Compute anomaly probabilities + extract segments
      4) Save cropped segment videos
      5) Run VLM per saved segment
    """
    # 1) ROI
    roi_eff = _ensure_roi_from_video(video_path, roi)

    # 2) Model (cached)
    model, device = load_model(model_path)

    # 3) Anomaly prob + segments
    probs, fps = detect_anomaly_probs(video_path, model, device, seq_len, roi_eff)
    segments = extract_segments(probs, threshold=threshold, merge_gap=merge_gap, min_len=min_len)

    # 4) Save cropped segment files
    pattern = "anomaly_segment_{idx}.mp4"
    out_paths = save_cropped_segments(video_path, segments, pattern, roi_eff, fps)

    # 5) VLM per segment (file-based)
    results: Dict[str, List[Tuple[str, str]]] = {}
    for seg_path in out_paths:
        qa = run_video_inference(
            model=vlm_model,
            tokenizer=vlm_tokenizer,
            video_path=seg_path,
            generation_config=generation_config,
            num_segments=vlm_segments,
            max_num=vlm_max_num,
        )
        results[seg_path] = qa

    return results
