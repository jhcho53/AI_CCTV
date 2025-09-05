#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 Image subscriber → 5s chunk VLM inference (no anomaly model), multi-camera.

Legacy video_vlm API compatibility (NO prompt support):
- utils.video_vlm.run_frames_inference / run_video_inference do NOT accept a `prompt`.
- Those functions internally ask two fixed questions:
  (1) fall (Yes/No), then (2) scene description.
- This node gates by checking the FIRST answer (must be Yes/No).

Pipeline (memory-efficient):
- Collect raw ROS2 images into 5-second chunks (per camera).
- At ingest time, apply ROI crop (clamped) + downscale, then store ONLY JPEG bytes (not full frames).
- Stage 1 (gating): decode only a few sampled JPEGs and run VLM with tiny token budget.
- If first answer is Yes → Stage 2 (description): decode another few frames and run VLM with a larger token budget.
- If fall is positive and saving is enabled, stream JPEGs to a video file (no large RAM usage).
- If frames-direct inference is unavailable, fall back to creating a small temp video under /dev/shm.
- Warm-up timings and per-chunk performance are logged.

Multi-camera:
- camera_count: number of cameras (default 1)
- image_topic_tmpl: input topic template including "{idx}" placeholder, e.g., "/camera{idx}/image_raw"
- For camera_count==1, the legacy "image_topic" is respected.
"""

import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import json
import time
import tempfile
import shutil
import re
import numpy as np
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
from typing import List, Tuple, Optional, Any, Deque, Dict

try:
    cv2.setNumThreads(1)
except Exception:
    pass

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from utils.video_vlm import init_model, run_video_inference
try:
    from utils.video_vlm import run_frames_inference
    HAVE_RUN_FRAMES = True
except Exception:
    run_frames_inference = None
    HAVE_RUN_FRAMES = False


# ---------- Text utilities ----------
YES_TOKENS = {"yes", "y", "true", "1", "yeah", "yep", "affirmative"}

def text_norm(s: str) -> str:
    """Lowercase and collapse non-alnum to spaces for robust Yes matching."""
    return re.sub(r'[\W_]+', ' ', (s or '')).strip().lower()

def is_yes(s: str) -> bool:
    """Heuristic Yes detector."""
    t = text_norm(s)
    return t in YES_TOKENS or t.startswith("yes")

def is_fall_positive(qa_pairs: List[Tuple[str, str]]) -> bool:
    """
    Decide fall-positive ONLY by the first answer (Yes-family).
    The question text/keywords are not used.
    """
    if not qa_pairs:
        return False
    first_answer = qa_pairs[0][1]
    return is_yes(first_answer)


# ---------- Filesystem / path helpers ----------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def unique_path(base_dir: str, base_name: str) -> str:
    """Return a unique path under base_dir using base_name; suffix with _N if exists."""
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


# ---------- ROI / sampling ----------
def parse_roi_param(val: Any) -> Optional[Tuple[int, int, int, int]]:
    """
    Parse ROI from parameter value.
    Accepts: None/""/[] → None; [x1,y1,x2,y2]; "x1,y1,x2,y2".
    """
    if val in (None, "", []):
        return None
    if isinstance(val, (list, tuple)) and len(val) == 4:
        x1, y1, x2, y2 = map(int, val)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("ROI must satisfy x2>x1 and y2>y1")
        return (x1, y1, x2, y2)
    if isinstance(val, str):
        parts = [p for p in re.split(r"[,\s]+", val.strip()) if p]
        if len(parts) != 4:
            raise ValueError("ROI string must contain 4 ints")
        x1, y1, x2, y2 = map(int, parts)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("ROI must satisfy x2>x1 and y2>y1")
        return (x1, y1, x2, y2)
    raise ValueError("Unsupported ROI param type")

def uniform_indices(total: int, n: int) -> List[int]:
    """Pick n uniform indices in [0, total-1]."""
    if total <= 0:
        return []
    n = max(1, int(n))
    if n == 1:
        return [total - 1]  # pick the last frame for fast gating
    return np.linspace(0, total - 1, n).astype(int).tolist()


# ---------- JPEG buffer I/O (memory-efficient) ----------
def crop_downscale(frame_bgr: np.ndarray, roi: Optional[Tuple[int, int, int, int]], target_width: int) -> np.ndarray:
    """
    Crop to ROI (clamped to image bounds) and downscale to target_width (keeping aspect ratio).
    """
    f = frame_bgr
    if roi is not None:
        H, W = f.shape[:2]
        x1, y1, x2, y2 = roi
        x1 = max(0, min(W, x1)); x2 = max(0, min(W, x2))
        y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
        if x2 > x1 and y2 > y1:
            f = f[y1:y2, x1:x2]
    if target_width and target_width > 0 and f.shape[1] > target_width:
        h, w = f.shape[:2]
        scale = target_width / float(w)
        f = cv2.resize(f, (target_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return f

def bgr_to_jpeg_bytes(frame_bgr: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed for JPEG.")
    return buf.tobytes()

def jpeg_bytes_to_bgr(jpg: bytes) -> np.ndarray:
    arr = np.frombuffer(jpg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cv2.imdecode failed for JPEG.")
    return img

def jpegs_to_temp_video(
    jpegs: List[bytes],
    size_wh: Tuple[int, int],
    fps: float,
    prefer_container: str = "avi",
) -> str:
    """
    Stream-decode JPEG bytes and write to a temp video under /dev/shm when available.
    Returns the temp file path.
    """
    if not jpegs:
        raise ValueError("No JPEGs to write.")
    w, h = size_wh
    tmpdir = "/dev/shm" if os.path.isdir("/dev/shm") else None

    def _open_writer(prefer: str):
        if prefer == "avi":
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            tmp = tempfile.NamedTemporaryFile(dir=tmpdir, suffix=".avi", delete=False)
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            tmp = tempfile.NamedTemporaryFile(dir=tmpdir, suffix=".mp4", delete=False)
        path = tmp.name; tmp.close()
        writer = cv2.VideoWriter(path, fourcc, max(float(fps or 30.0), 1.0), (w, h))
        return writer, path

    writer, path = _open_writer(prefer_container)
    if not writer.isOpened():
        os.remove(path)
        prefer_alt = "mp4" if prefer_container == "avi" else "avi"
        writer, path = _open_writer(prefer_alt)
        if not writer.isOpened():
            os.remove(path)
            raise RuntimeError("VideoWriter open failed.")

    try:
        for b in jpegs:
            f = jpeg_bytes_to_bgr(b)
            writer.write(f)
    finally:
        writer.release()
    return path

def jpegs_sample_to_video(
    jpegs: List[bytes],
    indices: List[int],
    size_wh: Tuple[int, int],
    fps: float,
    prefer_container: str = "avi",
) -> str:
    """
    Write only sampled frames to a temp video (memory-saver).
    """
    if not indices:
        raise ValueError("No indices to write.")
    w, h = size_wh
    tmpdir = "/dev/shm" if os.path.isdir("/dev/shm") else None

    def _open_writer(prefer: str):
        if prefer == "avi":
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            tmp = tempfile.NamedTemporaryFile(dir=tmpdir, suffix=".avi", delete=False)
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            tmp = tempfile.NamedTemporaryFile(dir=tmpdir, suffix=".mp4", delete=False)
        path = tmp.name; tmp.close()
        writer = cv2.VideoWriter(path, fourcc, max(float(fps or 10.0), 1.0), (w, h))
        return writer, path

    writer, path = _open_writer(prefer_container)
    if not writer.isOpened():
        os.remove(path)
        prefer_alt = "mp4" if prefer_container == "avi" else "avi"
        writer, path = _open_writer(prefer_alt)
        if not writer.isOpened():
            os.remove(path)
            raise RuntimeError("VideoWriter open failed.")

    try:
        for i in indices:
            f = jpeg_bytes_to_bgr(jpegs[i])
            writer.write(f)
    finally:
        writer.release()
    return path


# ---------- Per-camera context ----------
@dataclass
class CamCtx:
    idx: int
    topic: str
    chunk_jpegs: List[bytes] = field(default_factory=list)
    frame_size: Optional[Tuple[int, int]] = None
    chunk_start_t: Optional[float] = None
    processing: bool = False


# ---------- ROS2 node (multi-camera) ----------
class VLMGatedNode(Node):
    def __init__(self):
        super().__init__("vlm_chunk_recorder_gated_multi")

        # Inputs / general
        self.declare_parameter("image_topic", "/camera/image_raw")      # legacy (single-camera)
        self.declare_parameter("camera_count", 1)                       
        self.declare_parameter("image_topic_tmpl", "/camera{idx}/image_raw") 
        self.declare_parameter("chunk_secs", 5.0)
        self.declare_parameter("est_fps", 15.0)
        self.declare_parameter("roi", "")                  # "x1,y1,x2,y2" string (or [] with a param file)
        self.declare_parameter("alerts_dir", "alerts")
        self.declare_parameter("keep_raw_when_yes", True)  # Save clip on positive fall
        self.declare_parameter("target_width", 480)        # Downscale width
        self.declare_parameter("jpeg_quality", 80)         # JPEG quality (size vs. quality)

        # Saving
        self.declare_parameter("save_mode", "full")        # "full" | "skim"
        self.declare_parameter("save_container", "avi")    # "avi"(MJPG) | "mp4"(mp4v)
        self.declare_parameter("save_skim_fps", 8.0)       # FPS for skim mode

        # Optimization
        self.declare_parameter("use_frames_direct", True)  # Prefer frames-direct
        self.declare_parameter("prewarm", True)

        # Gating (fast Yes/No)
        self.declare_parameter("gate_segments", 1)
        self.declare_parameter("gate_max_new_tokens", 2)
        self.declare_parameter("gate_do_sample", False)

        # Description (only if Yes)
        self.declare_parameter("desc_segments", 3)
        self.declare_parameter("desc_max_new_tokens", 64)
        self.declare_parameter("desc_do_sample", False)

        # Resolve parameters
        gp = self.get_parameter
        self.single_topic = gp("image_topic").get_parameter_value().string_value
        self.camera_count = int(gp("camera_count").value)
        self.topic_tmpl   = gp("image_topic_tmpl").get_parameter_value().string_value

        self.chunk_secs  = float(gp("chunk_secs").value)
        self.est_fps     = float(gp("est_fps").value)
        self.alerts_dir  = gp("alerts_dir").get_parameter_value().string_value
        self.keep_raw    = bool(gp("keep_raw_when_yes").value)
        self.target_w    = int(gp("target_width").value)
        self.jpeg_quality= int(gp("jpeg_quality").value)

        self.save_mode   = gp("save_mode").get_parameter_value().string_value.lower()
        self.save_container = gp("save_container").get_parameter_value().string_value.lower()
        self.save_skim_fps  = float(gp("save_skim_fps").value)

        self.use_frames_direct = bool(gp("use_frames_direct").value) and HAVE_RUN_FRAMES
        self.prewarm = bool(gp("prewarm").value)

        self.gate_segments = int(gp("gate_segments").value)
        self.gate_max_tok  = int(gp("gate_max_new_tokens").value)
        self.gate_sample   = bool(gp("gate_do_sample").value)

        self.desc_segments = int(gp("desc_segments").value)
        self.desc_max_tok  = int(gp("desc_max_new_tokens").value)
        self.desc_sample   = bool(gp("desc_do_sample").value)

        try:
            self.roi = parse_roi_param(gp("roi").value)
        except Exception as e:
            raise RuntimeError(f"Invalid ROI param: {e}")

        # Paths
        self.scenes_dir = os.path.join(self.alerts_dir, "scenes")
        self.logs_dir   = os.path.join(self.alerts_dir, "logs")
        ensure_dir(self.scenes_dir); ensure_dir(self.logs_dir)
        self.jsonl_path = os.path.join(self.logs_dir, "vlm_results.jsonl")

        # VLM load + warm-up
        self.get_logger().info("Loading VLM...")
        t0 = time.monotonic()
        self.vlm_model, self.vlm_tokenizer = init_model()
        self.get_logger().info(f"VLM loaded in {time.monotonic()-t0:.2f}s | frames_direct={self.use_frames_direct}")

        if self.prewarm:
            self.get_logger().info("Warming up VLM...")
            try:
                dummy = np.zeros((224, 224, 3), dtype=np.uint8)
                if self.use_frames_direct:
                    _ = run_frames_inference(
                        model=self.vlm_model, tokenizer=self.vlm_tokenizer,
                        frames=[dummy],
                        generation_config={"max_new_tokens": 8, "do_sample": False},
                        num_segments=1, max_num=1
                    )
                else:
                    # Warm-up with a single-frame temp video
                    path = jpegs_to_temp_video([bgr_to_jpeg_bytes(dummy)], (224, 224), fps=5.0, prefer_container="avi")
                    try:
                        _ = run_video_inference(
                            model=self.vlm_model, tokenizer=self.vlm_tokenizer,
                            video_path=path,
                            generation_config={"max_new_tokens": 8, "do_sample": False},
                            num_segments=1, max_num=1
                        )
                    finally:
                        try:
                            os.remove(path)
                        except Exception:
                            pass
            except Exception as e:
                self.get_logger().warning(f"Warm-up skipped: {e}")
            self.get_logger().info("Warm-up done.")

        # State
        self.bridge = CvBridge()
        self.cams: Dict[int, CamCtx] = {}
        self.subs: List[Any] = []

        # Build subscriptions
        topics: List[str]
        if self.camera_count <= 1:
            topics = [self.single_topic]
        else:
            topics = [self.topic_tmpl.replace("{idx}", str(i)) for i in range(self.camera_count)]

        for i, topic in enumerate(topics):
            self.cams[i] = CamCtx(idx=i, topic=topic)
            self.subs.append(self.create_subscription(Image, topic, self._make_image_cb(i), 10))
            self.get_logger().info(f"[cam{i}] Subscribed: {topic}")

        # One timer for all cameras
        self.timer = self.create_timer(0.05, self.timer_cb)
        self.get_logger().info(f"chunk_secs={self.chunk_secs}s, cameras={len(self.cams)}")

    # ---------- Per-camera Image callback ----------
    def _make_image_cb(self, cam_idx: int):
        def _cb(msg: Image):
            cam = self.cams[cam_idx]
            if cam.processing:  # drop while processing
                return
            try:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except CvBridgeError as e:
                self.get_logger().warning(f"[cam{cam_idx}] CvBridgeError: {e}")
                return
            if frame is None:
                return

            f_small = crop_downscale(frame, self.roi, self.target_w)
            if cam.frame_size is None:
                cam.frame_size = (int(f_small.shape[1]), int(f_small.shape[0]))
            try:
                jpg = bgr_to_jpeg_bytes(f_small, quality=self.jpeg_quality)
                cam.chunk_jpegs.append(jpg)
            except Exception as e:
                self.get_logger().warning(f"[cam{cam_idx}] JPEG encode failed: {e}")

            if cam.chunk_start_t is None:
                cam.chunk_start_t = time.monotonic()
        return _cb

    # ---------- Timer: check all cameras ----------
    def timer_cb(self):
        for cam_idx, cam in self.cams.items():
            if cam.chunk_start_t is None or cam.processing:
                continue
            elapsed = time.monotonic() - cam.chunk_start_t
            if elapsed < self.chunk_secs:
                continue

            # close chunk
            jpegs_to_process = cam.chunk_jpegs
            size_wh = cam.frame_size
            cam.chunk_jpegs = []
            cam.frame_size = None
            chunk_elapsed = max(elapsed, 1e-3)
            cam.chunk_start_t = None
            cam.processing = True
            try:
                self.process_chunk(cam, jpegs_to_process, size_wh, chunk_elapsed)
            except Exception as e:
                self.get_logger().error(f"[cam{cam_idx}] process_chunk error: {e}")
            finally:
                cam.processing = False

    # ---------- Main processing (per camera) ----------
    def process_chunk(self, cam: CamCtx, jpegs: List[bytes], size_wh: Optional[Tuple[int, int]], elapsed_sec: float):
        if not jpegs or size_wh is None:
            return

        t0 = time.monotonic()
        n = len(jpegs)
        fps = max(1.0, float(n) / float(elapsed_sec or 1.0))
        if n < 3:
            fps = max(fps, self.est_fps)

        # ---- Stage 1: gating ----
        self.get_logger().info(f"[cam{cam.idx}] [Perf] Stage1 gate begin")
        t_gate0 = time.monotonic()
        gate_idxs = uniform_indices(n, self.gate_segments)
        gate_frames = [jpeg_bytes_to_bgr(jpegs[i]) for i in gate_idxs]

        try:
            if self.use_frames_direct:
                qa_gate = run_frames_inference(
                    model=self.vlm_model, tokenizer=self.vlm_tokenizer,
                    frames=gate_frames,
                    generation_config={"max_new_tokens": self.gate_max_tok, "do_sample": self.gate_sample},
                    num_segments=len(gate_frames), max_num=1
                )
            else:
                gate_clip = jpegs_sample_to_video(
                    jpegs, gate_idxs, size_wh, fps=min(fps, 10.0), prefer_container="avi"
                )
                try:
                    qa_gate = run_video_inference(
                        model=self.vlm_model, tokenizer=self.vlm_tokenizer,
                        video_path=gate_clip,
                        generation_config={"max_new_tokens": self.gate_max_tok, "do_sample": self.gate_sample},
                        num_segments=len(gate_idxs), max_num=1
                    )
                finally:
                    try:
                        os.remove(gate_clip)
                    except Exception:
                        pass
        except Exception as e:
            self.get_logger().error(f"[cam{cam.idx}] gate inference failed: {e}")
            return
        t_gate1 = time.monotonic()

        fall_yes = is_fall_positive(qa_gate)
        self.get_logger().info(f"[cam{cam.idx}] [Perf] Stage1 gate end (fall={fall_yes}) elapsed={t_gate1 - t_gate0:.2f}s")

        if not fall_yes:
            self.get_logger().info(f"[cam{cam.idx}] No fall: discard chunk (skip Stage 2)")
            self._log_perf(cam.idx, t0, elapsed_sec, vlm=t_gate1 - t_gate0, frames=n)
            return

        # ---- Stage 2: description ----
        self.get_logger().info(f"[cam{cam.idx}] [Perf] Stage2 desc begin")
        t_desc0 = time.monotonic()
        desc_idxs = uniform_indices(n, self.desc_segments)
        desc_frames = [jpeg_bytes_to_bgr(jpegs[i]) for i in desc_idxs]

        try:
            if self.use_frames_direct:
                qa_desc = run_frames_inference(
                    model=self.vlm_model, tokenizer=self.vlm_tokenizer,
                    frames=desc_frames,
                    generation_config={"max_new_tokens": self.desc_max_tok, "do_sample": self.desc_sample},
                    num_segments=len(desc_frames), max_num=1
                )
                saved_path = ""
            else:
                desc_clip = jpegs_sample_to_video(
                    jpegs, desc_idxs, size_wh, fps=min(fps, 10.0), prefer_container="avi"
                )
                try:
                    qa_desc = run_video_inference(
                        model=self.vlm_model, tokenizer=self.vlm_tokenizer,
                        video_path=desc_clip,
                        generation_config={"max_new_tokens": self.desc_max_tok, "do_sample": self.desc_sample},
                        num_segments=len(desc_idxs), max_num=1
                    )
                finally:
                    saved_path = ""
                    try:
                        os.remove(desc_clip)
                    except Exception:
                        pass
        except Exception as e:
            self.get_logger().error(f"[cam{cam.idx}] desc inference failed: {e}")
            return
        t_desc1 = time.monotonic()

        # ---- Save (stream JPEG→VideoWriter) ----
        saved_path = ""
        if self.keep_raw:
            try:
                if self.save_mode == "skim":
                    tmp_path = jpegs_sample_to_video(
                        jpegs, desc_idxs, size_wh, fps=self.save_skim_fps, prefer_container=self.save_container
                    )
                else:
                    tmp_path = jpegs_to_temp_video(
                        jpegs, size_wh, fps=fps, prefer_container=self.save_container
                    )
                ext = os.path.splitext(tmp_path)[1] or (".avi" if self.save_container == "avi" else ".mp4")
                dest = unique_path(self.scenes_dir, f"fall_cam{cam.idx}_{now_str()}{ext}")
                try:
                    shutil.move(tmp_path, dest)
                except Exception:
                    dest = tmp_path
                saved_path = dest
                self.get_logger().info(f"[cam{cam.idx}] [ALERT] FALL detected → saved: {saved_path}")
            except Exception as e:
                self.get_logger().warning(f"[cam{cam.idx}] Save streaming failed: {e}")
                saved_path = ""

        # ---- Output + logging ----
        qa_all = qa_gate + [("-----", "-----")] + qa_desc
        ts = datetime.now().isoformat(timespec='seconds')
        print(f"\n=== [cam{cam.idx}] [{ts}] FALL detected ===")
        for q, a in qa_all:
            print(f"Q: {q}\nA: {a}\n")

        rec = {
            "timestamp": ts,
            "camera_idx": cam.idx,
            "source_topic": cam.topic,
            "saved_path": saved_path,
            "roi": self.roi,
            "duration_sec": round(elapsed_sec, 3),
            "fps_est": round(fps, 2),
            "gate_segments": self.gate_segments,
            "gate_max_new_tokens": self.gate_max_tok,
            "desc_segments": self.desc_segments,
            "desc_max_new_tokens": self.desc_max_tok,
            "save_mode": self.save_mode,
            "save_container": self.save_container,
            "qa": qa_all
        }
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            self.get_logger().warning(f"[cam{cam.idx}] JSONL append failed: {e}")

        self._log_perf(cam.idx, t0, elapsed_sec, vlm=(t_gate1 - t_gate0) + (t_desc1 - t_desc0), frames=n)

    def _log_perf(self, cam_idx: int, t0, collect_s, vlm, frames):
        t1 = time.monotonic()
        self.get_logger().info(
            f"[cam{cam_idx}] [Perf] collect={collect_s:.2f}s, vlm={vlm:.2f}s, total={t1 - t0:.2f}s, frames={frames}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = VLMGatedNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
