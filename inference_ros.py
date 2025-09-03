#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 Image 구독 → 5초 청크 VLM 추론 (Anomaly 없음)
- 가능하면 프레임을 직접 VLM에 전달(run_frames_inference) → 디스크 I/O 제거
- 불가 시 /dev/shm(RAM 디스크)에 임시 파일로 저장 (eMMC/SD I/O 병목 최소화)
- 시작 시 워밍업(옵션)으로 첫 추론 지연 제거
- 단계별 소요시간을 로그로 출력(수집/인코딩/추론/총합)
- 낙상 Q/A 첫 답변이 Yes/True일 때만 청크 저장/로깅
"""

import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import json
import time
import shutil
import tempfile
import re
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional, Any

try:
    cv2.setNumThreads(1)
except Exception:
    pass

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from utils.video_vlm import init_model, run_video_inference
# 프레임 직접 추론 지원 여부 확인
try:
    from utils.video_vlm import run_frames_inference
    HAVE_RUN_FRAMES = True
except Exception:
    run_frames_inference = None
    HAVE_RUN_FRAMES = False

YES_TOKENS = {"yes", "y", "true", "1", "yeah", "yep", "affirmative"}
def text_norm(s: str) -> str:
    return re.sub(r'[\W_]+', ' ', (s or '')).strip().lower()
def is_yes(s: str) -> bool:
    t = text_norm(s)
    return t in YES_TOKENS or t.startswith("yes")
def is_fall_positive(qa_pairs: List[Tuple[str, str]]) -> bool:
    if not qa_pairs: return False
    q0, a0 = qa_pairs[0]
    qn = f" {text_norm(q0)} "
    if (" fall " in qn) or (" fallen " in qn) or (" fall down " in qn):
        return is_yes(a0)
    return False

def ensure_dir(path: str): os.makedirs(path, exist_ok=True)
def now_str(): return datetime.now().strftime("%Y%m%d_%H%M%S")
def unique_path(base_dir: str, base_name: str) -> str:
    p = os.path.join(base_dir, base_name)
    if not os.path.exists(p): return p
    root, ext = os.path.splitext(base_name); i = 1
    while True:
        cand = os.path.join(base_dir, f"{root}_{i}{ext}")
        if not os.path.exists(cand): return cand
        i += 1

def parse_roi_param(val: Any) -> Optional[Tuple[int,int,int,int]]:
    if val in (None, "", []): return None
    if isinstance(val, (list,tuple)) and len(val)==4:
        x1,y1,x2,y2 = map(int,val); 
        if x2<=x1 or y2<=y1: raise ValueError("ROI must satisfy x2>x1,y2>y1")
        return (x1,y1,x2,y2)
    if isinstance(val,str):
        parts=[p for p in re.split(r"[,\s]+",val.strip()) if p]
        if len(parts)!=4: raise ValueError("ROI string must be 4 ints")
        x1,y1,x2,y2 = map(int,parts)
        if x2<=x1 or y2<=y1: raise ValueError("ROI must satisfy x2>x1,y2>y1")
        return (x1,y1,x2,y2)
    raise ValueError("Unsupported ROI format")

def crop_and_downscale(frames: List[np.ndarray],
                       roi: Optional[Tuple[int,int,int,int]],
                       target_width: Optional[int]) -> List[np.ndarray]:
    out=[]
    for f in frames:
        if roi is not None:
            x1,y1,x2,y2 = roi
            f = f[y1:y2, x1:x2]
        if target_width and target_width>0 and f.shape[1] > target_width:
            h, w = f.shape[:2]
            scale = target_width / float(w)
            new_size = (target_width, max(1,int(h*scale)))
            f = cv2.resize(f, new_size, interpolation=cv2.INTER_AREA)
        out.append(f)
    return out

def write_frames_to_temp_video(frames: List[np.ndarray], fps: float,
                               prefer_container="avi") -> str:
    if not frames: raise ValueError("No frames to write.")
    h, w = frames[0].shape[:2]
    # RAM 디스크 우선
    tmpdir = "/dev/shm" if os.path.isdir("/dev/shm") else None
    if prefer_container=="avi":
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        tmp = tempfile.NamedTemporaryFile(dir=tmpdir, suffix=".avi", delete=False)
        path = tmp.name; tmp.close()
        writer = cv2.VideoWriter(path, fourcc, max(float(fps or 30.0),1.0), (w,h))
        if not writer.isOpened():
            os.remove(path)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            tmp = tempfile.NamedTemporaryFile(dir=tmpdir, suffix=".mp4", delete=False)
            path = tmp.name; tmp.close()
            writer = cv2.VideoWriter(path, fourcc, max(float(fps or 30.0),1.0), (w,h))
            if not writer.isOpened():
                os.remove(path); raise RuntimeError("VideoWriter open failed.")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        tmp = tempfile.NamedTemporaryFile(dir=tmpdir, suffix=".mp4", delete=False)
        path = tmp.name; tmp.close()
        writer = cv2.VideoWriter(path, fourcc, max(float(fps or 30.0),1.0), (w,h))
        if not writer.isOpened():
            os.remove(path)
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            tmp = tempfile.NamedTemporaryFile(dir=tmpdir, suffix=".avi", delete=False)
            path = tmp.name; tmp.close()
            writer = cv2.VideoWriter(path, fourcc, max(float(fps or 30.0),1.0), (w,h))
            if not writer.isOpened():
                os.remove(path); raise RuntimeError("VideoWriter open failed.")
    for f in frames: writer.write(f)
    writer.release()
    return path


class VLMChunkNode(Node):
    def __init__(self):
        super().__init__("vlm_chunk_recorder_fast")

        # 파라미터
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("chunk_secs", 5.0)
        self.declare_parameter("est_fps", 15.0)
        self.declare_parameter("roi", "")
        self.declare_parameter("alerts_dir", "alerts")
        self.declare_parameter("keep_raw", False)

        # 최적화 파라미터
        self.declare_parameter("use_frames_direct", True)  # run_frames_inference 사용
        self.declare_parameter("target_width", 640)        # 프레임 다운스케일 폭
        self.declare_parameter("prewarm", True)            # 시작 시 워밍업 한 번

        # VLM
        self.declare_parameter("vlm_segments", 4)
        self.declare_parameter("vlm_max", 1)
        self.declare_parameter("max_new_tokens", 128)
        self.declare_parameter("do_sample", False)

        # 로드
        self.image_topic   = self.get_parameter("image_topic").get_parameter_value().string_value
        self.chunk_secs    = float(self.get_parameter("chunk_secs").value)
        self.est_fps       = float(self.get_parameter("est_fps").value)
        roi_param          = self.get_parameter("roi").value
        self.alerts_dir    = self.get_parameter("alerts_dir").get_parameter_value().string_value
        self.keep_raw      = bool(self.get_parameter("keep_raw").value)

        self.use_frames_direct = bool(self.get_parameter("use_frames_direct").value) and HAVE_RUN_FRAMES
        self.target_width  = int(self.get_parameter("target_width").value)
        self.prewarm       = bool(self.get_parameter("prewarm").value)

        self.vlm_segments  = int(self.get_parameter("vlm_segments").value)
        self.vlm_max       = int(self.get_parameter("vlm_max").value)
        self.max_new_tokens= int(self.get_parameter("max_new_tokens").value)
        self.do_sample     = bool(self.get_parameter("do_sample").value)

        try:
            self.roi = parse_roi_param(roi_param)
        except Exception as e:
            raise RuntimeError(f"Invalid ROI param: {e}")

        # 경로
        self.scenes_dir = os.path.join(self.alerts_dir, "scenes")
        self.logs_dir   = os.path.join(self.alerts_dir, "logs")
        ensure_dir(self.scenes_dir); ensure_dir(self.logs_dir)
        self.jsonl_path = os.path.join(self.logs_dir, "vlm_results.jsonl")

        # VLM 로드 + 워밍업
        self.get_logger().info("Loading VLM...")
        t0 = time.monotonic()
        self.vlm_model, self.vlm_tokenizer = init_model()
        t1 = time.monotonic()
        self.get_logger().info(f"VLM loaded in {t1 - t0:.2f}s")

        if self.prewarm:
            self.get_logger().info("Warming up VLM...")
            try:
                # 아주 작은 더미 프레임 1~2장으로 워밍업
                dummy = np.zeros((224,224,3), dtype=np.uint8)
                if self.use_frames_direct:
                    _ = run_frames_inference(
                        model=self.vlm_model, tokenizer=self.vlm_tokenizer,
                        frames=[dummy, dummy],
                        generation_config={"max_new_tokens":16, "do_sample":False},
                        num_segments=1, max_num=1
                    )
                else:
                    # 임시 파일 워밍업 (RAM 디스크)
                    path = write_frames_to_temp_video([dummy, dummy], fps=5.0, prefer_container="avi")
                    try:
                        _ = run_video_inference(
                            model=self.vlm_model, tokenizer=self.vlm_tokenizer,
                            video_path=path,
                            generation_config={"max_new_tokens":16, "do_sample":False},
                            num_segments=1, max_num=1
                        )
                    finally:
                        try: os.remove(path)
                        except Exception: pass
            except Exception as e:
                self.get_logger().warn(f"Warm-up skipped due to error: {e}")
            self.get_logger().info("Warm-up done.")

        # 상태
        self.bridge = CvBridge()
        self.frames: List[np.ndarray] = []
        self.chunk_start_t: Optional[float] = None
        self.processing = False

        # ROS
        self.sub = self.create_subscription(Image, self.image_topic, self.image_cb, 10)
        self.timer = self.create_timer(0.05, self.timer_cb)  # 더 촘촘히 체크
        self.get_logger().info(f"Subscribed: {self.image_topic} | chunk_secs={self.chunk_secs}s | frames_direct={self.use_frames_direct}")

    def image_cb(self, msg: Image):
        if self.processing:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().warn(f"CvBridgeError: {e}")
            return
        if frame is None: return
        self.frames.append(frame)
        if self.chunk_start_t is None:
            self.chunk_start_t = time.monotonic()

    def timer_cb(self):
        if self.chunk_start_t is None or self.processing:
            return
        elapsed = time.monotonic() - self.chunk_start_t
        if elapsed < self.chunk_secs:
            return
        frames_to_process = self.frames
        self.frames = []
        chunk_elapsed = max(elapsed, 1e-3)
        self.chunk_start_t = None
        self.processing = True
        try:
            self.process_chunk(frames_to_process, chunk_elapsed)
        except Exception as e:
            self.get_logger().error(f"process_chunk error: {e}")
        finally:
            self.processing = False

    def process_chunk(self, frames: List[np.ndarray], elapsed_sec: float):
        if not frames: return

        t0 = time.monotonic()
        fps = max(1.0, float(len(frames)) / float(elapsed_sec or 1.0))
        if len(frames) < 3: fps = max(fps, self.est_fps)

        # ROI + 다운스케일 (I/O 및 VLM 비전백본 비용 절감)
        frames_small = crop_and_downscale(frames, self.roi, self.target_width)
        t_pre = time.monotonic()

        if self.use_frames_direct:
            # (A) 프레임 직접 추론 경로
            self.get_logger().info("[Perf] start VLM (frames_direct)")
            try:
                qa = run_frames_inference(
                    model=self.vlm_model, tokenizer=self.vlm_tokenizer,
                    frames=frames_small,
                    generation_config={"max_new_tokens": self.max_new_tokens,
                                       "do_sample": self.do_sample},
                    num_segments=self.vlm_segments, max_num=self.vlm_max
                )
                clip_path = ""  # 파일 없음
            except Exception as e:
                self.get_logger().error(f"VLM(frames) failed: {e}")
                return
            t_vlm = time.monotonic()
            enc_time = 0.0
        else:
            # (B) 임시 파일 경로 (/dev/shm)
            t_enc0 = time.monotonic()
            try:
                raw_clip = write_frames_to_temp_video(frames_small, fps, prefer_container="avi")
            except Exception as e:
                self.get_logger().error(f"Failed to write temp video: {e}")
                return
            t_enc1 = time.monotonic()
            self.get_logger().info("[Perf] start VLM (video_path)")
            try:
                qa = run_video_inference(
                    model=self.vlm_model, tokenizer=self.vlm_tokenizer,
                    video_path=raw_clip,
                    generation_config={"max_new_tokens": self.max_new_tokens,
                                       "do_sample": self.do_sample},
                    num_segments=self.vlm_segments, max_num=self.vlm_max
                )
            except Exception as e:
                self.get_logger().error(f"VLM(video) failed: {e}")
                try: os.remove(raw_clip)
                except Exception: pass
                return
            t_vlm = time.monotonic()
            enc_time = t_enc1 - t_enc0
            clip_path = raw_clip

        # 판정 및 저장/로깅 (Yes만)
        fall_pos = is_fall_positive(qa)
        ts = datetime.now().isoformat(timespec='seconds')

        if fall_pos:
            if self.use_frames_direct:
                # 프레임 직접일 때도 결과 보존을 원하면 파일로 저장(선택)
                # 여기서는 경량화를 위해 저장하지 않고 메시지만 남김
                saved_path = ""
            else:
                # 파일 경로를 scenes로 이동
                ext = os.path.splitext(clip_path)[1] or ".avi"
                dest = unique_path(self.scenes_dir, f"fall_{now_str()}{ext}")
                try:
                    shutil.move(clip_path, dest)
                    saved_path = dest
                except Exception as e:
                    self.get_logger().warn(f"Failed to move chunk: {e}")
                    saved_path = clip_path

            print(f"\n=== [{ts}] FALL: saved:{bool(saved_path)} path:{os.path.basename(saved_path) if saved_path else '(frames_direct)'} ===")
            for q,a in qa:
                print(f"Q: {q}\nA: {a}\n")

            rec = {
                "timestamp": ts,
                "source_topic": self.image_topic,
                "saved_path": saved_path,
                "roi": self.roi,
                "duration_sec": round(elapsed_sec,3),
                "fps_est": round(fps,2),
                "vlm_segments": self.vlm_segments,
                "vlm_max": self.vlm_max,
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.do_sample,
                "qa": qa
            }
            try:
                with open(self.jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as e:
                self.get_logger().warn(f"JSONL append failed: {e}")
        else:
            # No → 파일/메모리 폐기
            if not self.use_frames_direct and os.path.exists(clip_path):
                try: os.remove(clip_path)
                except Exception: pass
            self.get_logger().info("No fall: chunk discarded")

        t1 = time.monotonic()
        self.get_logger().info(
            f"[Perf] collect={elapsed_sec:.2f}s, pre={t_pre - t0:.2f}s, "
            f"encode={(enc_time):.2f}s, vlm={(t_vlm - (t_pre if self.use_frames_direct else t_enc1)):.2f}s, "
            f"total={t1 - t0:.2f}s, frames={len(frames)} (~{fps:.1f}fps)"
        )


def main(args=None):
    rclpy.init(args=args)
    node=None
    try:
        node = VLMChunkNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
