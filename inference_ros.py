#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 Image 구독 → 5초 청크 VLM 추론(Anomaly 없음)
- 2단계 게이팅:
   1) 빠른 Yes/No (낙상 여부)만 생성 → 토큰/세그먼트 최소화
   2) Yes일 때만 자세 설명(토큰/세그먼트 확대) + 저장/로깅
- 가능하면 프레임 직접 추론(run_frames_inference) 활용, 없으면 /dev/shm 임시파일
- 시작 워밍업으로 첫 추론 지연 제거
- 단계별 시간 프로파일 출력
"""

import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2, json, time, tempfile, shutil, re, numpy as np
from datetime import datetime
from typing import List, Tuple, Optional, Any

try: cv2.setNumThreads(1)
except Exception: pass

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


# ---------- 유틸 ----------
YES_TOKENS = {"yes", "y", "true", "1", "yeah", "yep", "affirmative"}
def text_norm(s: str) -> str:
    return re.sub(r'[\W_]+', ' ', (s or '')).strip().lower()
def is_yes(s: str) -> bool:
    t = text_norm(s)
    return t in YES_TOKENS or t.startswith("yes")

def is_fall_positive(qa_pairs: List[Tuple[str, str]]) -> bool:
    """첫 Q/A가 fall/fallen/fall down 포함 & 첫 답이 Yes/True면 True"""
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
    if isinstance(val,(list,tuple)) and len(val)==4:
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

def crop_and_downscale(frames, roi, target_width):
    out=[]
    for f in frames:
        if roi is not None:
            x1,y1,x2,y2 = roi
            f = f[y1:y2, x1:x2]
        if target_width and target_width>0 and f.shape[1] > target_width:
            h,w = f.shape[:2]
            scale = target_width/float(w)
            f = cv2.resize(f, (target_width, max(1,int(h*scale))), interpolation=cv2.INTER_AREA)
        out.append(f)
    return out

def write_frames_to_temp_video(frames, fps, prefer_container="avi"):
    if not frames: raise ValueError("No frames to write.")
    h,w = frames[0].shape[:2]
    tmpdir = "/dev/shm" if os.path.isdir("/dev/shm") else None
    # MJPG 우선, 실패 시 mp4v
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
    for f in frames: writer.write(f)
    writer.release()
    return path


# ---------- ROS2 노드 ----------
class VLMGatedNode(Node):
    def __init__(self):
        super().__init__("vlm_chunk_recorder_gated")

        # 입력/일반
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("chunk_secs", 5.0)
        self.declare_parameter("est_fps", 15.0)
        self.declare_parameter("roi", "")
        self.declare_parameter("alerts_dir", "alerts")
        self.declare_parameter("keep_raw_when_yes", True)  # Yes일 때 저장
        self.declare_parameter("target_width", 480)        # 다운스케일 폭

        # 최적화
        self.declare_parameter("use_frames_direct", True)  # run_frames_inference 사용
        self.declare_parameter("prewarm", True)

        # 게이팅(빠른 Yes/No)
        self.declare_parameter("gate_segments", 1)
        self.declare_parameter("gate_max_new_tokens", 2)
        self.declare_parameter("gate_do_sample", False)

        # 설명(Yes일 때만)
        self.declare_parameter("desc_segments", 3)
        self.declare_parameter("desc_max_new_tokens", 64)
        self.declare_parameter("desc_do_sample", False)

        # 로드
        gp = self.get_parameter
        self.image_topic = gp("image_topic").get_parameter_value().string_value
        self.chunk_secs  = float(gp("chunk_secs").value)
        self.est_fps     = float(gp("est_fps").value)
        self.alerts_dir  = gp("alerts_dir").get_parameter_value().string_value
        self.keep_raw    = bool(gp("keep_raw_when_yes").value)
        self.target_w    = int(gp("target_width").value)

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

        # 경로
        self.scenes_dir = os.path.join(self.alerts_dir, "scenes")
        self.logs_dir   = os.path.join(self.alerts_dir, "logs")
        ensure_dir(self.scenes_dir); ensure_dir(self.logs_dir)
        self.jsonl_path = os.path.join(self.logs_dir, "vlm_results.jsonl")

        # VLM 로드 + 워밍업
        self.get_logger().info("Loading VLM...")
        t0 = time.monotonic()
        self.vlm_model, self.vlm_tokenizer = init_model()
        self.get_logger().info(f"VLM loaded in {time.monotonic()-t0:.2f}s")

        if self.prewarm:
            self.get_logger().info("Warming up VLM...")
            try:
                dummy = np.zeros((224,224,3), dtype=np.uint8)
                if self.use_frames_direct:
                    _ = run_frames_inference(
                        model=self.vlm_model, tokenizer=self.vlm_tokenizer,
                        frames=[dummy, dummy],
                        generation_config={"max_new_tokens":8, "do_sample":False},
                        num_segments=1, max_num=1
                    )
                else:
                    p = write_frames_to_temp_video([dummy, dummy], fps=5.0)
                    try:
                        _ = run_video_inference(
                            model=self.vlm_model, tokenizer=self.vlm_tokenizer,
                            video_path=p,
                            generation_config={"max_new_tokens":8, "do_sample":False},
                            num_segments=1, max_num=1
                        )
                    finally:
                        try: os.remove(p)
                        except Exception: pass
            except Exception as e:
                self.get_logger().warn(f"Warm-up skipped: {e}")
            self.get_logger().info("Warm-up done.")

        # 상태
        self.bridge = CvBridge()
        self.frames: List[np.ndarray] = []
        self.chunk_start_t: Optional[float] = None
        self.processing = False

        # ROS
        self.sub = self.create_subscription(Image, self.image_topic, self.image_cb, 10)
        self.timer = self.create_timer(0.05, self.timer_cb)
        self.get_logger().info(f"Subscribed: {self.image_topic} | chunk_secs={self.chunk_secs}s | frames_direct={self.use_frames_direct}")

    def image_cb(self, msg: Image):
        if self.processing:  # 추론 중 드롭
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

        # 다운스케일 & ROI
        frames_small = crop_and_downscale(frames, self.roi, self.target_w)
        fps = max(1.0, float(len(frames)) / float(elapsed_sec or 1.0))
        if len(frames) < 3: fps = max(fps,  self.get_parameter("est_fps").value)

        # ---- 1) 게이팅: Yes/No만 ----
        self.get_logger().info("[Perf] Stage1 gate begin")
        t_gate0 = time.monotonic()
        try:
            if self.use_frames_direct:
                qa_gate = run_frames_inference(
                    model=self.vlm_model, tokenizer=self.vlm_tokenizer,
                    frames=frames_small,
                    generation_config={"max_new_tokens": self.gate_max_tok,
                                       "do_sample": self.gate_sample},
                    num_segments=self.gate_segments, max_num=1
                )
                gate_clip_path = ""
            else:
                path_gate = write_frames_to_temp_video(frames_small, fps)
                qa_gate = run_video_inference(
                    model=self.vlm_model, tokenizer=self.vlm_tokenizer,
                    video_path=path_gate,
                    generation_config={"max_new_tokens": self.gate_max_tok,
                                       "do_sample": self.gate_sample},
                    num_segments=self.gate_segments, max_num=1
                )
                gate_clip_path = path_gate
        except Exception as e:
            self.get_logger().error(f"gate inference failed: {e}")
            return
        finally:
            if not self.use_frames_direct and 'path_gate' in locals():
                try: os.remove(path_gate)
                except Exception: pass
        t_gate1 = time.monotonic()

        fall_yes = is_fall_positive(qa_gate)
        self.get_logger().info(f"[Perf] Stage1 gate end (fall={fall_yes}) elapsed={t_gate1 - t_gate0:.2f}s")

        if not fall_yes:
            self.get_logger().info("No fall: discard chunk (skip Stage2)")
            self._log_perf(t0, elapsed_sec, pre=0.0, enc=0.0, vlm=t_gate1 - t_gate0, frames=len(frames))
            return

        # ---- 2) 설명: Yes일 때만 ----
        self.get_logger().info("[Perf] Stage2 desc begin")
        t_desc0 = time.monotonic()
        # 설명은 조금 넉넉히 생성; 그래도 저사양을 고려해 제한
        try:
            if self.use_frames_direct:
                qa_desc = run_frames_inference(
                    model=self.vlm_model, tokenizer=self.vlm_tokenizer,
                    frames=frames_small,
                    generation_config={"max_new_tokens": self.desc_max_tok,
                                       "do_sample": self.desc_sample},
                    num_segments=self.desc_segments, max_num=1
                )
                saved_path = ""  # frames_direct 경로 없음
            else:
                path_desc = write_frames_to_temp_video(frames_small, fps)
                qa_desc = run_video_inference(
                    model=self.vlm_model, tokenizer=self.vlm_tokenizer,
                    video_path=path_desc,
                    generation_config={"max_new_tokens": self.desc_max_tok,
                                       "do_sample": self.desc_sample},
                    num_segments=self.desc_segments, max_num=1
                )
                if self.get_parameter("keep_raw_when_yes").value:
                    ext = os.path.splitext(path_desc)[1] or ".avi"
                    dest = unique_path(self.scenes_dir, f"fall_{now_str()}{ext}")
                    try: shutil.move(path_desc, dest)
                    except Exception:
                        dest = path_desc
                    saved_path = dest
                else:
                    saved_path = ""
                    try: os.remove(path_desc)
                    except Exception: pass
        except Exception as e:
            self.get_logger().error(f"desc inference failed: {e}")
            return
        t_desc1 = time.monotonic()

        # 출력 + 로깅
        qa_all = qa_gate + [("-----", "-----")] + qa_desc
        ts = datetime.now().isoformat(timespec='seconds')
        print(f"\n=== [{ts}] FALL detected ===")
        for q,a in qa_all:
            print(f"Q: {q}\nA: {a}\n")

        rec = {
            "timestamp": ts,
            "source_topic": self.image_topic,
            "saved_path": saved_path,
            "roi": self.roi,
            "duration_sec": round(elapsed_sec,3),
            "fps_est": round(fps,2),
            "gate_segments": self.gate_segments,
            "gate_max_new_tokens": self.gate_max_tok,
            "desc_segments": self.desc_segments,
            "desc_max_new_tokens": self.desc_max_tok,
            "qa": qa_all
        }
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            self.get_logger().warn(f"JSONL append failed: {e}")

        self._log_perf(t0, elapsed_sec, pre=0.0, enc=0.0,
                       vlm=(t_gate1 - t_gate0) + (t_desc1 - t_desc0), frames=len(frames))

    def _log_perf(self, t0, collect_s, pre, enc, vlm, frames):
        t1 = time.monotonic()
        self.get_logger().info(
            f"[Perf] collect={collect_s:.2f}s, pre={pre:.2f}s, encode={enc:.2f}s, "
            f"vlm={vlm:.2f}s, total={t1 - t0:.2f}s, frames={frames}"
        )


def main(args=None):
    rclpy.init(args=args)
    node=None
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
