import os
import cv2
import torch
import numpy as np
import time
import tempfile
from typing import List, Tuple, Dict, Optional

from model.model import GRUAnomalyDetector

# VLM: 프레임 직접 추론(run_frames_inference)이 있으면 사용하고,
# 없으면 run_video_inference로 폴백(필요 시 임시 파일 생성).
try:
    from utils.video_vlm import run_video_inference, run_frames_inference  # type: ignore
except Exception:
    from utils.video_vlm import run_video_inference  # type: ignore
    run_frames_inference = None  # 프레임 입력 미지원 환경


# =========================
# 1) 모델 로드 (캐싱)
# =========================

_MODEL_CACHE: Dict[str, Tuple[torch.nn.Module, torch.device]] = {}

def load_model(model_path: str, device: Optional[torch.device] = None):
    """
    Load the GRU anomaly detector model with caching.
    Returns (model, device).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
# 2) 공통 유틸
# =========================

def _ensure_roi_from_first_frame(frames: List[np.ndarray], roi):
    """
    roi가 None이면 첫 프레임 전체 영역을 ROI로 반환.
    """
    if roi is not None:
        return tuple(map(int, roi))
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    return (0, 0, w, h)

def _ensure_roi_from_video(video_path: str, roi):
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

def _preprocess_patch_bgr(frame_bgr: np.ndarray, roi: Tuple[int, int, int, int]):
    x1, y1, x2, y2 = roi
    crop = frame_bgr[y1:y2, x1:x2]
    patch = cv2.resize(crop, (256, 256))
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(patch).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return tensor  # (1, 3, 256, 256)

def _decode_video_to_frames(video_path: str) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-3:
        fps = 30.0  # 합리적 기본값
    ok = True
    while ok:
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return frames, fps


# =========================
# 3) 이상 확률 계산
# =========================

def detect_anomaly_probs_from_frames(
    frames: List[np.ndarray],
    model: torch.nn.Module,
    device: torch.device,
    seq_len: int = 8,
    roi: Optional[Tuple[int, int, int, int]] = (0, 0, 256, 256)
):
    """
    프레임 리스트를 받아 per-frame anomaly 확률을 반환.
    로직/전처리는 detect_anomaly_probs(video_path, ...)와 동일하게 유지.
    """
    total = len(frames)
    roi_eff = _ensure_roi_from_first_frame(frames, roi)
    if roi_eff is None:
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
    roi: Tuple[int, int, int, int] = (0, 0, 256, 256)
):
    """
    기존 인터페이스 유지: 비디오를 직접 열어 per-frame anomaly 확률 계산.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-3:
        fps = 30.0

    if roi is None:
        # 전체 프레임 ROI
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
# 4) 세그먼트 추출/병합
# =========================

def _merge_segments(segments: List[Tuple[int, int]], gap_tolerance: int = 200):
    """
    인접/겹침 세그먼트를 병합합니다.
    gap_tolerance(프레임) 이내로 떨어진 구간은 하나로 합칩니다.
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
    min_len: int = 1
):
    """
    프레임별 확률에서 연속된 이상 구간을 추출하고 병합합니다.
    반환: [(start_frame, end_frame)]  (inclusive)
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
# 5) 세그먼트 자르기 (파일/프레임)
# =========================

def save_cropped_segments(
    video_path: str,
    segments: List[Tuple[int, int]],
    output_pattern: str,
    roi: Tuple[int, int, int, int],
    fps: float,
    fourcc_str: str = 'mp4v'
):
    """
    기존 파일 저장 버전 (호환성 유지).
    segments는 (start_frame, end_frame) inclusive.
    """
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
    roi: Tuple[int, int, int, int]
) -> List[List[np.ndarray]]:
    """
    파일 저장 없이, 세그먼트별로 ROI를 잘라 프레임 리스트로 반환.
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
# 6) VLM: 프레임 기반 우선, 불가 시 폴백
# =========================

def _write_frames_to_temp_video(frames: List[np.ndarray], fps: float) -> str:
    """
    프레임 리스트를 임시 MP4로 저장(폴백용).
    """
    if not frames:
        raise ValueError("No frames to write.")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
    fps_for_fallback: float = 30.0
):
    """
    가능하면 프레임 직접 VLM 추론을 수행하고,
    그렇지 않으면 작은 임시 비디오를 생성하여 run_video_inference로 폴백.
    """
    generation_config = generation_config or {"max_new_tokens": 512, "do_sample": True}

    # 1) 프레임 직접 지원 시
    if run_frames_inference is not None:
        return run_frames_inference(
            model=vlm_model,
            tokenizer=vlm_tokenizer,
            frames=clip_frames,
            generation_config=generation_config,
            num_segments=num_segments,
            max_num=max_num
        )

    # 2) 폴백: 임시 파일
    tmp_path = _write_frames_to_temp_video(clip_frames, fps_for_fallback)
    try:
        qa = run_video_inference(
            model=vlm_model,
            tokenizer=vlm_tokenizer,
            video_path=tmp_path,
            generation_config=generation_config,
            num_segments=num_segments,
            max_num=max_num
        )
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return qa


# =========================
# 7) 엔드투엔드: 프레임/파일
# =========================

def run_inference_on_frames(
    frames: List[np.ndarray],
    model_or_tuple,                      # (model, device) 또는 모델 경로(str)
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
    min_len: int = 1
) -> Dict[str, List[Tuple[str, str]]]:
    """
    스트리밍/메모리 상 프레임 리스트를 직접 받아 이상탐지+VLM 수행.
    반환: { tag: [(Q, A), ...] }
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
        roi=roi_eff
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
            fps_for_fallback=fps_for_vlm_fallback
        )
        qa_all.extend(qa)

    results[tag] = qa_all
    return results


def run_inference(
    video_path: str,
    model_path: str = 'anomaly_single_balanced.pth',
    seq_len: int = 8,
    roi: Optional[Tuple[int, int, int, int]] = None,  # None이면 전체 프레임
    threshold: float = 0.1,
    # VLM 인자
    vlm_model=None,
    vlm_tokenizer=None,
    generation_config=None,
    vlm_segments: int = 8,
    vlm_max_num: int = 1,
    merge_gap: int = 200,
    min_len: int = 1
) -> Dict[str, List[Tuple[str, str]]]:
    """
    기존 시그니처/반환형 유지: 파일 경로 입력 → {segment_path: [(Q,A), ...]}
    내부는 모델 캐싱 + 프레임 기반 공용 로직을 사용.
    """
    # 1) ROI 결정
    roi_eff = _ensure_roi_from_video(video_path, roi)

    # 2) 모델 로드(캐싱)
    model, device = load_model(model_path)

    # 3) 이상 확률 + 세그먼트
    probs, fps = detect_anomaly_probs(video_path, model, device, seq_len, roi_eff)
    segments = extract_segments(probs, threshold=threshold, merge_gap=merge_gap, min_len=min_len)

    # 4) 세그먼트 ROI 잘라 파일 저장(기존과 동일 동작)
    pattern = 'anomaly_segment_{idx}.mp4'
    out_paths = save_cropped_segments(video_path, segments, pattern, roi_eff, fps)

    # 5) VLM 추론 (파일 기반: 기존과 동일)
    results: Dict[str, List[Tuple[str, str]]] = {}
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
