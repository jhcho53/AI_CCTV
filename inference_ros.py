#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 Image 토픽 구독 → 프레임 청크를 임시 비디오로 저장 → 기존 run_inference(video_path=...) 실행
낙상(첫 질문이 fall 관련이고 첫 답변이 Yes)일 때만 세그먼트 저장 + CSV 로깅

의존:
  - rclpy
  - sensor_msgs.msg.Image
  - cv_bridge
  - opencv-python
  - utils.inf_utils.run_inference
  - utils.video_vlm.init_model
"""

import os
# TensorFlow 비활성화 (Transformers 등에서 TF를 로드하지 않도록)
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import re
import csv
import json
import time
import shutil
import tempfile
from datetime import datetime
from typing import List, Tuple, Optional, Any

# OpenCV 멀티 스레드 레이스 완화
try:
    cv2.setNumThreads(1)
except Exception:
    pass

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from utils.inf_utils import run_inference
from utils.video_vlm import init_model


# -------------------------------
# 문자열 유틸/게이팅 로직
# -------------------------------
YES_TOKENS = {"yes", "y", "true", "1", "yeah", "yep", "affirmative"}
NO_TOKENS  = {"no", "n", "false", "0", "nope", "nah"}

def text_norm(s: str) -> str:
    return re.sub(r'[\W_]+', ' ', (s or '')).strip().lower()

def is_yes(s: str) -> bool:
    t = text_norm(s)
    return t in YES_TOKENS or t.startswith("yes")

def is_fall_positive(qa_pairs: List[Tuple[str, str]]) -> bool:
    """
    규칙:
    1) 첫 Q/A가 'Did someone fall/fallen/fall down ... ?' 계열일 때만 평가
    2) 그 답변이 Yes일 때만 True
    """
    if not qa_pairs:
        return False
    q0, a0 = qa_pairs[0]
    qn = text_norm(q0)
    contains_fall = (" fall " in f" {qn} ") or (" fallen " in f" {qn} ") or (" fall down " in f" {qn} ")
    if not contains_fall:
        return False
    return is_yes(a0)

def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

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

def append_csv(csv_path: str, row: dict, field_order: List[str]):
    ensure_dir(os.path.dirname(csv_path) or ".")
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        if not file_exists:
            w.writeheader()
        w.writerow(row)

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


# -------------------------------
# 비디오 파일 저장 유틸
# -------------------------------
def _write_frames_to_temp_video(frames: List, fps: float, prefer_container: str = "avi") -> str:
    """
    프레임 리스트(BGR)를 임시 비디오로 저장.
    MJPG(AVI) 우선, 실패 시 mp4v(MP4) 폴백.
    """
    if not frames:
        raise ValueError("No frames to write.")
    h, w = frames[0].shape[:2]
    # 1st try
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
# ROS2 노드
# -------------------------------
class AnomalyRecorderNode(Node):
    def __init__(self):
        super().__init__("anomaly_vlm_recorder")

        # ---- 파라미터 선언 ----
        self.declare_parameter("image_topic", "/v4l2_camera")
        self.declare_parameter("chunk_secs", 10.0)      # 청크 길이(초)
        self.declare_parameter("est_fps", 30.0)         # FPS 추정(가변 프레임일 때)
        self.declare_parameter("model_path", "anomaly_single_balanced.pth")
        self.declare_parameter("seq_len", 8)
        self.declare_parameter("threshold", 0.3)
        self.declare_parameter("vlm_segments", 8)
        self.declare_parameter("vlm_max", 1)
        self.declare_parameter("roi", "")               # "x1,y1,x2,y2" or []
        self.declare_parameter("alerts_dir", "alerts")
        self.declare_parameter("keep_raw", False)       # 원본 청크 파일 보관
        self.declare_parameter("keep_nonfall_segments", False)  # 낙상 No 세그먼트도 보관할지
        self.declare_parameter("preview", False)        # 간이 미리보기

        # ---- 파라미터 로드 ----
        self.image_topic   = self.get_parameter("image_topic").get_parameter_value().string_value
        self.chunk_secs    = float(self.get_parameter("chunk_secs").value)
        self.est_fps       = float(self.get_parameter("est_fps").value)
        self.model_path    = self.get_parameter("model_path").get_parameter_value().string_value
        self.seq_len       = int(self.get_parameter("seq_len").value)
        self.threshold     = float(self.get_parameter("threshold").value)
        self.vlm_segments  = int(self.get_parameter("vlm_segments").value)
        self.vlm_max       = int(self.get_parameter("vlm_max").value)
        roi_param          = self.get_parameter("roi").value
        self.alerts_dir    = self.get_parameter("alerts_dir").get_parameter_value().string_value
        self.keep_raw      = bool(self.get_parameter("keep_raw").value)
        self.keep_nonfall  = bool(self.get_parameter("keep_nonfall_segments").value)
        self.preview       = bool(self.get_parameter("preview").value)

        try:
            self.roi = parse_roi_param(roi_param)
        except Exception as e:
            raise RuntimeError(f"Invalid ROI param: {e}")

        # ---- 출력 경로 ----
        self.scenes_dir = os.path.join(self.alerts_dir, "scenes")
        self.logs_dir   = os.path.join(self.alerts_dir, "logs")
        ensure_dir(self.scenes_dir)
        ensure_dir(self.logs_dir)
        self.csv_log_path = os.path.join(self.logs_dir, "fall_events.csv")
        self.csv_fields = [
            "timestamp", "source_topic", "raw_clip", "segment_path", "saved_path",
            "fall_positive", "first_question", "first_answer", "qa_json",
            "roi", "seq_len", "threshold", "vlm_segments", "vlm_max_num"
        ]

        # ---- VLM 로드 ----
        self.get_logger().info("Loading VLM...")
        self.vlm_model, self.vlm_tokenizer = init_model()
        self.get_logger().info("VLM loaded.")

        # ---- 상태 ----
        self.bridge = CvBridge()
        self.frames: List = []                # 현재 청크 프레임들 (BGR)
        self.chunk_start_t: Optional[float] = None
        self.processing = False               # 청크 처리 중 플래그

        # ---- 구독/타이머 ----
        self.sub = self.create_subscription(Image, self.image_topic, self.image_cb, 10)
        self.timer = self.create_timer(0.1, self.timer_cb)  # 10Hz로 청크 마감 체크
        self.get_logger().info(f"Subscribed: {self.image_topic} | chunk_secs={self.chunk_secs}s")

    # ---------------------------
    # 콜백
    # ---------------------------
    def image_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().warn(f"CvBridgeError: {e}")
            return

        if frame is None:
            return

        # 프리뷰(옵션)
        if self.preview:
            try:
                cv2.imshow("ROS Preview (press q to close preview)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.preview = False
                    cv2.destroyAllWindows()
            except Exception:
                self.preview = False

        self.frames.append(frame)
        if self.chunk_start_t is None:
            self.chunk_start_t = time.monotonic()

    def timer_cb(self):
        # 청크가 아직 시작되지 않았거나, 처리 중이면 스킵
        if self.chunk_start_t is None or self.processing:
            return

        elapsed = time.monotonic() - self.chunk_start_t
        if elapsed < self.chunk_secs:
            return

        # 청크 마감: 프레임 복사/리셋 후 처리
        frames_to_process = self.frames
        self.frames = []
        chunk_elapsed = max(elapsed, 1e-3)
        self.chunk_start_t = time.monotonic()  # 다음 청크 시작
        self.processing = True

        # 동기 처리(간단): 콜백 안에서 바로 처리
        try:
            self.process_chunk(frames_to_process, chunk_elapsed)
        except Exception as e:
            self.get_logger().error(f"process_chunk error: {e}")
        finally:
            self.processing = False

    # ---------------------------
    # 청크 처리
    # ---------------------------
    def process_chunk(self, frames: List, elapsed_sec: float):
        if not frames:
            return

        fps = max(1.0, float(len(frames)) / float(elapsed_sec or 1.0))
        # 프레임 수가 너무 적으면 est_fps 사용
        if len(frames) < 3:
            fps = max(fps, self.est_fps)

        # 1) 프레임 → 임시 비디오 파일
        try:
            raw_clip = _write_frames_to_temp_video(frames, fps, prefer_container="avi")
        except Exception as e:
            self.get_logger().error(f"Failed to write temp video: {e}")
            return

        # 2) 파일 모드 추론
        try:
            outputs = run_inference(
                video_path=raw_clip,
                model_path=self.model_path,
                seq_len=self.seq_len,
                roi=self.roi,
                threshold=self.threshold,
                vlm_model=self.vlm_model,
                vlm_tokenizer=self.vlm_tokenizer,
                generation_config={"max_new_tokens": 512, "do_sample": True},
                vlm_segments=self.vlm_segments,
                vlm_max_num=self.vlm_max
            )
        except Exception as e:
            self.get_logger().error(f"Inference failed on {raw_clip}: {e}")
            if not self.keep_raw and os.path.exists(raw_clip):
                try: os.remove(raw_clip)
                except Exception: pass
            return

        # 3) 게이팅 + 저장/로깅
        any_saved = False
        for seg_path, qa in outputs.items():
            fall_pos = is_fall_positive(qa)
            first_q = qa[0][0] if qa else ""
            first_a = qa[0][1] if qa else ""
            if fall_pos:
                base_name = f"fall_{now_str()}_{os.path.basename(seg_path)}"
                dest_path = unique_path(self.scenes_dir, base_name)
                ensure_dir(os.path.dirname(dest_path))
                try:
                    shutil.move(seg_path, dest_path)
                    any_saved = True
                    self.get_logger().info(f"[ALERT] FALL detected → saved: {dest_path}")
                except Exception as e:
                    self.get_logger().warn(f"Failed to move segment: {e}")
                    dest_path = ""  # 로깅상 빈값

                append_csv(self.csv_log_path, {
                    "timestamp": datetime.now().isoformat(timespec='seconds'),
                    "source_topic": self.image_topic,
                    "raw_clip": raw_clip,
                    "segment_path": seg_path,
                    "saved_path": dest_path,
                    "fall_positive": True,
                    "first_question": first_q,
                    "first_answer": first_a,
                    "qa_json": json.dumps(qa, ensure_ascii=False),
                    "roi": str(self.roi),
                    "seq_len": self.seq_len,
                    "threshold": self.threshold,
                    "vlm_segments": self.vlm_segments,
                    "vlm_max_num": self.vlm_max
                }, self.csv_fields)
            else:
                # No면 저장하지 않음 → 세그먼트 파일 삭제(기본)
                if not self.keep_nonfall:
                    try:
                        os.remove(seg_path)
                    except Exception:
                        pass

        # 4) 원본 청크 파일 정리
        if not self.keep_raw and os.path.exists(raw_clip):
            try:
                os.remove(raw_clip)
            except Exception:
                pass

        if not any_saved:
            self.get_logger().info(f"No fall detected in chunk ({os.path.basename(raw_clip)}).")


# -------------------------------
# 엔트리포인트
# -------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = AnomalyRecorderNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None and node.preview:
            try: cv2.destroyAllWindows()
            except Exception: pass
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
