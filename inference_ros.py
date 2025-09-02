#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 Image 토픽 구독 → 5초 청크를 임시 비디오로 저장 → VLM(run_video_inference)만 실행
VLM 추론 중에는 카메라 프레임을 받지 않고(drop) 처리합니다.

의존:
  - rclpy
  - sensor_msgs.msg.Image
  - cv_bridge
  - opencv-python
  - utils.video_vlm.init_model, run_video_inference
"""

import os
# TensorFlow 비활성화 (Transformers 등에서 TF를 로드하지 않도록)
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import json
import time
import tempfile
from datetime import datetime
from typing import List, Tuple, Optional, Any
import re

# OpenCV 멀티 스레드 레이스 완화
try:
    cv2.setNumThreads(1)
except Exception:
    pass

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from utils.video_vlm import init_model, run_video_inference


# -------------------------------
# 유틸
# -------------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def parse_roi_param(val: Any) -> Optional[Tuple[int, int, int, int]]:
    """
    ROI 파라미터 파싱:
      - [] 또는 '' 또는 None -> None
      - [x1, y1, x2, y2] 또는 "x1,y1,x2,y2" 또는 "x1 y1 x2 y2"
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
            raise ValueError("ROI string must have 4 integers: 'x1,y1,x2,y2'")
        x1, y1, x2, y2 = map(int, parts)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("ROI must satisfy x2>x1 and y2>y1")
        return (x1, y1, x2, y2)
    raise ValueError("Unsupported ROI param format")

def _write_frames_to_temp_video(frames: List, fps: float,
                                roi: Optional[Tuple[int,int,int,int]] = None,
                                prefer_container: str = "avi") -> str:
    """
    프레임 리스트(BGR)를 (필요 시 ROI 크롭 후) 임시 비디오로 저장.
    MJPG(AVI) 우선, 실패 시 mp4v(MP4) 폴백.
    """
    if not frames:
        raise ValueError("No frames to write.")

    # ROI가 있으면 미리 크롭해 I/O·인코딩 비용을 줄임
    if roi is not None:
        x1, y1, x2, y2 = roi
        cropped = []
        for f in frames:
            f = f[y1:y2, x1:x2]
            cropped.append(f)
        frames = cropped

    h, w = frames[0].shape[:2]

    if prefer_container == "avi":
        suffix = ".avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        path = tmp.name; tmp.close()
        writer = cv2.VideoWriter(path, fourcc, max(float(fps or 30.0), 1.0), (w, h))
        if not writer.isOpened():
            # fallback mp4
            os.remove(path)
            suffix = ".mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            path = tmp.name; tmp.close()
            writer = cv2.VideoWriter(path, fourcc, max(float(fps or 30.0), 1.0), (w, h))
            if not writer.isOpened():
                os.remove(path)
                raise RuntimeError("Failed to open VideoWriter for both MJPG(AVI) and mp4v(MP4).")
    else:
        suffix = ".mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        path = tmp.name; tmp.close()
        writer = cv2.VideoWriter(path, fourcc, max(float(fps or 30.0), 1.0), (w, h))
        if not writer.isOpened():
            os.remove(path)
            suffix = ".avi"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            path = tmp.name; tmp.close()
            writer = cv2.VideoWriter(path, fourcc, max(float(fps or 30.0), 1.0), (w, h))
            if not writer.isOpened():
                os.remove(path)
                raise RuntimeError("Failed to open VideoWriter for both mp4v(MP4) and MJPG(AVI).")

    for f in frames:
        writer.write(f)
    writer.release()
    return path


# -------------------------------
# ROS2 노드 (VLM-only)
# -------------------------------
class VLMChunkNode(Node):
    def __init__(self):
        super().__init__("vlm_chunk_recorder")

        # ---- 파라미터 선언 ----
        self.declare_parameter("image_topic", "/v4l2_camera")   # 필요 시 /camera/image_raw 로 변경
        self.declare_parameter("chunk_secs", 5.0)               # ★ 5초 청크
        self.declare_parameter("est_fps", 15.0)                 # Jetson 환경에서 보수적 기본 FPS
        self.declare_parameter("roi", "")                       # "x1,y1,x2,y2" or []
        self.declare_parameter("alerts_dir", "alerts")
        self.declare_parameter("keep_raw", False)               # 원본 청크 파일 보관 여부

        # VLM 설정
        self.declare_parameter("vlm_segments", 6)               # 세그먼트 수 (낮출수록 빠름)
        self.declare_parameter("vlm_max", 1)                    # 프레임당 패치 수
        self.declare_parameter("max_new_tokens", 256)           # 생성 토큰수 (낮출수록 빠름)
        self.declare_parameter("do_sample", True)

        # ---- 파라미터 로드 ----
        self.image_topic   = self.get_parameter("image_topic").get_parameter_value().string_value
        self.chunk_secs    = float(self.get_parameter("chunk_secs").value)
        self.est_fps       = float(self.get_parameter("est_fps").value)
        roi_param          = self.get_parameter("roi").value
        self.alerts_dir    = self.get_parameter("alerts_dir").get_parameter_value().string_value
        self.keep_raw      = bool(self.get_parameter("keep_raw").value)

        self.vlm_segments  = int(self.get_parameter("vlm_segments").value)
        self.vlm_max       = int(self.get_parameter("vlm_max").value)
        self.max_new_tokens= int(self.get_parameter("max_new_tokens").value)
        self.do_sample     = bool(self.get_parameter("do_sample").value)

        try:
            self.roi = parse_roi_param(roi_param)
        except Exception as e:
            raise RuntimeError(f"Invalid ROI param: {e}")

        # ---- 출력 경로 ----
        self.logs_dir   = os.path.join(self.alerts_dir, "logs")
        ensure_dir(self.logs_dir)
        self.jsonl_path = os.path.join(self.logs_dir, "vlm_results.jsonl")

        # ---- VLM 로드 ----
        self.get_logger().info("Loading VLM...")
        self.vlm_model, self.vlm_tokenizer = init_model()
        self.get_logger().info("VLM loaded.")

        # ---- 상태 ----
        self.bridge = CvBridge()
        self.frames: List = []                # 현재 청크 프레임들 (BGR)
        self.chunk_start_t: Optional[float] = None
        self.processing = False               # ★ 추론 중 True → 프레임 드롭

        # ---- 구독/타이머 ----
        self.sub = self.create_subscription(Image, self.image_topic, self.image_cb, 10)
        self.timer = self.create_timer(0.1, self.timer_cb)  # 10Hz로 청크 마감 체크
        self.get_logger().info(f"Subscribed: {self.image_topic} | chunk_secs={self.chunk_secs}s")

    # ---------------------------
    # 콜백
    # ---------------------------
    def image_cb(self, msg: Image):
        # ★ 추론 중이면 프레임 DROP (메모리/CPU 절약)
        if self.processing:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().warn(f"CvBridgeError: {e}")
            return

        if frame is None:
            return

        self.frames.append(frame)
        if self.chunk_start_t is None:
            self.chunk_start_t = time.monotonic()

    def timer_cb(self):
        # 청크가 아직 시작되지 않았거나, 이미 처리 중이면 스킵
        if self.chunk_start_t is None or self.processing:
            return

        elapsed = time.monotonic() - self.chunk_start_t
        if elapsed < self.chunk_secs:
            return

        # 청크 마감: 프레임 복사/리셋 후 처리
        frames_to_process = self.frames
        self.frames = []
        chunk_elapsed = max(elapsed, 1e-3)
        self.chunk_start_t = None  # 다음 청크는 image_cb에서 시작
        self.processing = True     # ★ 이 동안은 image_cb에서 프레임 드롭

        # 동기 처리
        try:
            self.process_chunk(frames_to_process, chunk_elapsed)
        except Exception as e:
            self.get_logger().error(f"process_chunk error: {e}")
        finally:
            self.processing = False

    # ---------------------------
    # 청크 처리 (VLM만)
    # ---------------------------
    def process_chunk(self, frames: List, elapsed_sec: float):
        if not frames:
            return

        fps = max(1.0, float(len(frames)) / float(elapsed_sec or 1.0))
        # 너무 적게 들어오면 est_fps 사용
        if len(frames) < 3:
            fps = max(fps, self.est_fps)

        # 1) 프레임 → (ROI 적용) → 임시 비디오 파일
        try:
            raw_clip = _write_frames_to_temp_video(frames, fps,
                                                   roi=self.roi,
                                                   prefer_container="avi")
        except Exception as e:
            self.get_logger().error(f"Failed to write temp video: {e}")
            return

        # 2) VLM 추론만 수행
        try:
            qa = run_video_inference(
                model=self.vlm_model,
                tokenizer=self.vlm_tokenizer,
                video_path=raw_clip,
                generation_config={"max_new_tokens": self.max_new_tokens,
                                   "do_sample": self.do_sample},
                num_segments=self.vlm_segments,
                max_num=self.vlm_max
            )
        except Exception as e:
            self.get_logger().error(f"VLM inference failed on {raw_clip}: {e}")
            if not self.keep_raw and os.path.exists(raw_clip):
                try: os.remove(raw_clip)
                except Exception: pass
            return

        # 3) 결과 출력 + JSONL 로깅
        ts = datetime.now().isoformat(timespec='seconds')
        print(f"\n=== [{ts}] VLM for chunk: {os.path.basename(raw_clip)} ===")
        for q, a in qa:
            print(f"Q: {q}\nA: {a}\n")

        ensure_dir(self.logs_dir)
        record = {
            "timestamp": ts,
            "source_topic": self.image_topic,
            "chunk_path": raw_clip if self.keep_raw else "",  # keep_raw가 아니면 비워둠
            "roi": self.roi,
            "duration_sec": round(elapsed_sec, 3),
            "fps_est": round(fps, 2),
            "vlm_segments": self.vlm_segments,
            "vlm_max": self.vlm_max,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "qa": qa
        }
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            self.get_logger().warn(f"Failed to append JSONL: {e}")

        # 4) 원본 청크 파일 정리
        if not self.keep_raw and os.path.exists(raw_clip):
            try:
                os.remove(raw_clip)
            except Exception:
                pass


# -------------------------------
# 엔트리포인트
# -------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = None
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
