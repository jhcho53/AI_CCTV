#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch VLM Evaluation over a directory of short video sequences (OpenCV-only, no ROS).
- Hard-disable TensorFlow import by default (robust stub with __spec__/__path__)
  to avoid CUDA/XLA conflicts. Can be disabled with ALLOW_TF_IMPORT=1.
- Avoid decord/pyav; sample frames using OpenCV only.
- Always use frames_direct (run_frames_inference) to feed frames directly to InternVL.
- 2-stage gating:
  1) Fast Yes/No with minimal tokens/frames
  2) If Yes -> concise description + optional saving (copy or skim) + JSONL logging

Requires:
  - utils/video_vlm.py with:
      init_model(...)
      run_frames_inference(...)   # must exist
"""

# ---------- robust TensorFlow stub (before ANY other import) ----------
import os, sys, types, importlib.machinery
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

def _stub_package(name: str):
    """Create a package-like stub with __spec__ and __path__ so find_spec() doesn't error."""
    if name in sys.modules:
        m = sys.modules[name]
        if getattr(m, "__spec__", None) is None:
            m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        if not hasattr(m, "__path__"):
            m.__path__ = []  # type: ignore
        return
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    m.__path__ = []  # type: ignore
    sys.modules[name] = m

# By default, block TF to avoid CUDA/XLA conflicts. Can be disabled via env.
if os.environ.get("ALLOW_TF_IMPORT", "0").lower() not in ("1", "true", "yes", "y"):
    for mod in [
        "tensorflow",
        "tensorflow.python",
        "tensorflow.compat",
        "tensorflow.compat.v1",
        "tensorflow.compat.v2",
        "tf_keras",
        "keras",  # some stacks try keras first
    ]:
        _stub_package(mod)
# ----------------------------------------------------------------------

import cv2
import json
import time
import glob
import shutil
import argparse
import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime

try:
    cv2.setNumThreads(1)
except Exception:
    pass

from utils.video_vlm import init_model
try:
    # must exist; we won't use run_video_inference to avoid decoders
    from utils.video_vlm import run_frames_inference
except Exception:
    print("[ERROR] run_frames_inference not found in utils/video_vlm.py")
    print("        Please add it (see previously provided InternVL implementation).")
    sys.exit(1)

# ----------------- helpers -----------------
YES_TOKENS = {"yes", "y", "true", "1", "yeah", "yep", "affirmative"}

def text_norm(s: str) -> str:
    import re
    return re.sub(r'[\W_]+', ' ', (s or '')).strip().lower()

def is_yes(s: str) -> bool:
    t = text_norm(s)
    return t in YES_TOKENS or t.startswith("yes")

YES_TOKENS = {"yes", "y", "true", "1", "yeah", "yep", "affirmative"}

def text_norm(s: str) -> str:
    import re
    return re.sub(r'[\W_]+', ' ', (s or '')).strip().lower()

def is_yes(s: str) -> bool:
    t = text_norm(s)
    return t in YES_TOKENS or t.startswith("yes")

def is_fall_positive(qa_pairs):
    """
    변경된 규칙:
    - 첫 번째 답변이 Yes(계열)면 Positive
    """
    if not qa_pairs:
        return False
    return is_yes(qa_pairs[0][1])


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def unique_path(base_dir: str, base_name: str) -> str:
    p = os.path.join(base_dir, base_name)
    if not os.path.exists(p):
        return p
    root, ext = os.path.splitext(base_name)
    i = 1
    while True:
        cand = os.path.join(base_dir, f"{root}_{i}{ext}")
        if not os.path.exists(cand):
            return cand
        i += 1

def parse_roi(val: str) -> Optional[Tuple[int,int,int,int]]:
    val = (val or "").strip()
    if not val:
        return None
    parts = [x for x in val.replace(",", " ").split() if x]
    if len(parts) != 4:
        raise ValueError("ROI must be 'x1,y1,x2,y2'")
    x1, y1, x2, y2 = map(int, parts)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI must satisfy x2>x1 and y2>y1")
    return (x1, y1, x2, y2)

def uniform_indices(total: int, n: int) -> List[int]:
    if total <= 0:
        return []
    n = max(1, int(n))
    if n == 1:
        return [total - 1]  # last frame
    return np.linspace(0, total - 1, n).astype(int).tolist()

def crop_downscale(frame_bgr: np.ndarray, roi: Optional[Tuple[int,int,int,int]], target_w: int) -> np.ndarray:
    f = frame_bgr
    if roi is not None:
        x1, y1, x2, y2 = roi
        f = f[y1:y2, x1:x2]
    if target_w and target_w > 0 and f.shape[1] > target_w:
        h, w = f.shape[:2]
        scale = target_w / float(w)
        f = cv2.resize(f, (target_w, max(1, int(h*scale))), interpolation=cv2.INTER_AREA)
    return f

def list_videos(input_dir: str, exts: List[str], recursive: bool) -> List[str]:
    vids = []
    for e in exts:
        e = e if e.startswith(".") else "." + e
        pattern = os.path.join(input_dir, "**", f"*{e}") if recursive else os.path.join(input_dir, f"*{e}")
        vids.extend(glob.glob(pattern, recursive=recursive))
    vids = [v for v in vids if os.path.isfile(v)]
    vids.sort()
    return vids

# -------- OpenCV-only frame access (no decord/pyav) --------
def get_meta_opencv(path: str) -> Tuple[float, int]:
    """Return (fps, total_frames). If count is 0, do a counting pass."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if total <= 0:
        # fallback: count frames (short clips -> OK)
        cap = cv2.VideoCapture(path)
        total = 0
        ok = True
        while ok:
            ok, _ = cap.read()
            if ok:
                total += 1
        cap.release()
    if fps <= 1e-6:
        fps = 25.0
    return fps, total

def sample_frames_opencv(path: str, indices: List[int], roi: Optional[Tuple[int,int,int,int]], target_w: int) -> List[np.ndarray]:
    frames = []
    if not indices:
        return frames
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    for i in indices:
        idx = i if total <= 0 else max(0, min(i, total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            # try sequential read fallback
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
        frames.append(crop_downscale(bgr, roi, target_w))
    cap.release()
    return frames

def write_frames_to_temp_video(frames_bgr: List[np.ndarray], fps: float, prefer_container: str = "avi") -> str:
    """Only used when save_mode='skim'; writes a thin clip to /dev/shm."""
    if not frames_bgr:
        raise ValueError("No frames to write.")
    h, w = frames_bgr[0].shape[:2]
    tmpdir = "/dev/shm" if os.path.isdir("/dev/shm") else None
    if prefer_container == "avi":
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        suffix = ".avi"
        alt_fourcc = cv2.VideoWriter_fourcc(*"mp4v"); alt_suffix = ".mp4"
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        suffix = ".mp4"
        alt_fourcc = cv2.VideoWriter_fourcc(*"MJPG"); alt_suffix = ".avi"
    import tempfile
    tmp = tempfile.NamedTemporaryFile(dir=tmpdir, suffix=suffix, delete=False)
    path = tmp.name; tmp.close()
    vw = cv2.VideoWriter(path, fourcc, max(float(fps or 10.0), 1.0), (w, h))
    if not vw.isOpened():
        os.remove(path)
        tmp = tempfile.NamedTemporaryFile(dir=tmpdir, suffix=alt_suffix, delete=False)
        path = tmp.name; tmp.close()
        vw = cv2.VideoWriter(path, alt_fourcc, max(float(fps or 10.0), 1.0), (w, h))
        if not vw.isOpened():
            os.remove(path)
            raise RuntimeError("VideoWriter open failed for both containers.")
    for f in frames_bgr:
        vw.write(f)
    vw.release()
    return path
# -----------------------------------------------------------

# ----------------- core batch logic -----------------
def process_video(video_path: str, model, tokenizer, args) -> dict:
    t0 = time.monotonic()

    # meta + sampling indices
    fps, total = get_meta_opencv(video_path)
    gate_idxs = uniform_indices(total, args.gate_segments)
    desc_idxs = uniform_indices(total, args.desc_segments)

    qa_gate: List[Tuple[str, str]] = []
    qa_desc: List[Tuple[str, str]] = []
    saved_path = ""

    # ---------- Stage 1: gate ----------
    t_gate0 = time.monotonic()
    try:
        gate_frames = sample_frames_opencv(video_path, gate_idxs, args.roi, args.target_width)
        if not gate_frames:
            raise RuntimeError("No frames sampled for Stage1.")
        qa_gate = run_frames_inference(
            model=model, tokenizer=tokenizer,
            frames=gate_frames,
            generation_config={"max_new_tokens": args.gate_max_new_tokens,
                               "do_sample": args.gate_do_sample},
            num_segments=len(gate_frames),
            max_num=args.max_patches
        )
    except Exception as e:
        print(f"[ERROR] Gate inference failed for {os.path.basename(video_path)}: {e}")
        return {
            "video": video_path, "fall": False, "saved_path": "",
            "fps_est": float(fps), "qa_gate": [], "qa_desc": [],
            "elapsed_gate": time.monotonic()-t_gate0, "elapsed_desc": 0.0
        }
    t_gate1 = time.monotonic()
    fall_yes = is_fall_positive(qa_gate)

    # ---------- Stage 2: description (+ save) ----------
    t_desc0 = time.monotonic()
    if fall_yes:
        try:
            desc_frames = sample_frames_opencv(video_path, desc_idxs, args.roi, args.target_width)
            if not desc_frames:
                raise RuntimeError("No frames sampled for Stage2.")
            qa_desc = run_frames_inference(
                model=model, tokenizer=tokenizer,
                frames=desc_frames,
                generation_config={"max_new_tokens": args.desc_max_new_tokens,
                                   "do_sample": args.desc_do_sample},
                num_segments=len(desc_frames),
                max_num=args.max_patches
            )
            # save positive
            scenes_dir = os.path.join(args.alerts_dir, "scenes"); ensure_dir(scenes_dir)
            if args.save_mode == "copy":
                base = f"fall_{now_str()}_{os.path.basename(video_path)}"
                saved_path = unique_path(scenes_dir, base)
                shutil.copy2(video_path, saved_path)
            elif args.save_mode == "skim":
                path = write_frames_to_temp_video(
                    desc_frames, fps=min(fps, args.save_skim_fps), prefer_container=args.save_container
                )
                base = f"fall_{now_str()}{os.path.splitext(path)[1]}"
                dest = unique_path(scenes_dir, base)
                try:
                    shutil.move(path, dest)
                    saved_path = dest
                except Exception:
                    saved_path = path
            else:
                saved_path = ""
        except Exception as e:
            print(f"[ERROR] Desc/save failed for {os.path.basename(video_path)}: {e}")
    t_desc1 = time.monotonic()

    # ---------- Logging positives ----------
    if fall_yes:
        logs_dir = os.path.join(args.alerts_dir, "logs"); ensure_dir(logs_dir)
        rec = {
            "timestamp": datetime.now().isoformat(timespec='seconds'),
            "video": video_path,
            "saved_path": saved_path,
            "roi": args.roi,
            "fps_est": float(fps),
            "gate_segments": args.gate_segments,
            "gate_max_new_tokens": args.gate_max_new_tokens,
            "desc_segments": args.desc_segments,
            "desc_max_new_tokens": args.desc_max_new_tokens,
            "save_mode": args.save_mode,
            "save_container": args.save_container,
            "qa": (qa_gate + [("-----","-----")] + qa_desc)
        }
        jsonl = os.path.join(logs_dir, "vlm_results.jsonl")
        try:
            with open(jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[WARN] JSONL append failed: {e}")

    # ---------- stdout summary ----------
    if fall_yes:
        print(f"\n=== FALL detected in {os.path.basename(video_path)} ===")
        for q, a in (qa_gate + [("-----","-----")] + qa_desc):
            print(f"Q: {q}\nA: {a}\n")
        if saved_path:
            print(f"[SAVED] {saved_path}")
    else:
        if args.verbose:
            print(f"[INFO] No fall: {os.path.basename(video_path)}")

    return {
        "video": video_path,
        "fall": fall_yes,
        "saved_path": saved_path,
        "fps_est": float(fps),
        "qa_gate": qa_gate,
        "qa_desc": qa_desc,
        "elapsed_gate": t_gate1 - t_gate0,
        "elapsed_desc": (t_desc1 - t_desc0) if fall_yes else 0.0
    }

# ----------------- CLI -----------------
def build_argparser():
    p = argparse.ArgumentParser(description="Batch VLM evaluation over a video folder (OpenCV-only frames_direct).")
    # IO
    p.add_argument("--input_dir", required=True, help="Directory containing video files")
    p.add_argument("--exts", default=".mp4,.avi,.mov,.mkv", help="Comma-separated extensions")
    p.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    p.add_argument("--limit", type=int, default=0, help="Process only first N videos (0=all)")
    p.add_argument("--alerts_dir", default="alerts", help="Base dir for scenes/ and logs/")
    p.add_argument("--save_mode", default="copy", choices=["copy","skim","none"],
                   help="On positive: copy original, or save skim clip, or none")
    p.add_argument("--save_container", default="avi", choices=["avi","mp4"],
                   help="Container for skim clip")
    p.add_argument("--save_skim_fps", type=float, default=8.0, help="FPS for skim clip")
    # Model
    p.add_argument("--model_id", default=None, help="HF model id/path for init_model(path=...)")
    p.add_argument("--prewarm", action="store_true", help="Do a warm-up call")
    p.add_argument("--no_prewarm", dest="prewarm", action="store_false")
    p.set_defaults(prewarm=True)
    # Sampling / preprocessing
    p.add_argument("--roi", default="", help="ROI as 'x1,y1,x2,y2' (optional)")
    p.add_argument("--target_width", type=int, default=480, help="Downscale width after ROI")
    p.add_argument("--max_patches", type=int, default=1, help="InternVL tiling patches per frame (smaller=faster)")
    # Gating (Stage 1)
    p.add_argument("--gate_segments", type=int, default=1, help="Frames for gate (1=last frame only)")
    p.add_argument("--gate_max_new_tokens", type=int, default=2, help="Tokens for gate")
    p.add_argument("--gate_do_sample", action="store_true", help="Sampling for gate")
    # Description (Stage 2)
    p.add_argument("--desc_segments", type=int, default=3, help="Frames for description")
    p.add_argument("--desc_max_new_tokens", type=int, default=64, help="Tokens for description")
    p.add_argument("--desc_do_sample", action="store_true", help="Sampling for description")
    # Misc
    p.add_argument("--verbose", action="store_true")
    return p

def main():
    args = build_argparser().parse_args()

    # Prepare IO
    ensure_dir(args.alerts_dir)
    ensure_dir(os.path.join(args.alerts_dir, "scenes"))
    ensure_dir(os.path.join(args.alerts_dir, "logs"))

    # Parse ROI
    try:
        args.roi = parse_roi(args.roi)
    except Exception as e:
        print(f"[ERROR] Invalid ROI: {e}")
        sys.exit(1)

    # List videos
    exts = [e.strip() for e in args.exts.split(",") if e.strip()]
    videos = list_videos(args.input_dir, exts, args.recursive)
    if args.limit and args.limit > 0:
        videos = videos[:args.limit]
    if not videos:
        print("[WARN] No videos found.")
        sys.exit(0)

    # Load VLM
    print("[INFO] Loading VLM...")
    t0 = time.monotonic()
    if args.model_id:
        model, tokenizer = init_model(path=args.model_id)
    else:
        model, tokenizer = init_model()
    print(f"[INFO] VLM loaded in {time.monotonic()-t0:.2f}s (frames_direct=True)")

    # Warm-up (optional)
    if args.prewarm:
        print("[INFO] Warming up VLM...")
        try:
            dummy = np.zeros((224,224,3), dtype=np.uint8)
            _ = run_frames_inference(
                model=model, tokenizer=tokenizer,
                frames=[dummy],
                generation_config={"max_new_tokens": 8, "do_sample": False},
                num_segments=1, max_num=1
            )
        except Exception as e:
            print(f"[WARN] Warm-up skipped: {e}")
        print("[INFO] Warm-up done.")

    # Process
    n_total = len(videos)
    n_pos = 0
    total_gate = 0.0
    total_desc = 0.0

    for k, vp in enumerate(videos, 1):
        print(f"\n[{k}/{n_total}] {os.path.basename(vp)}")
        res = process_video(vp, model, tokenizer, args)
        if res["fall"]:
            n_pos += 1
        total_gate += res.get("elapsed_gate", 0.0)
        total_desc += res.get("elapsed_desc", 0.0)

    print("\n=== Summary ===")
    print(f"Processed: {n_total}")
    print(f"Positives: {n_pos}")
    if n_total > 0:
        print(f"Avg gate time: {total_gate / n_total:.2f}s")
    if n_pos > 0:
        print(f"Avg desc time (positives): {total_desc / n_pos:.2f}s")

if __name__ == "__main__":
    main()
