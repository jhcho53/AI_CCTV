#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Video → 5초 청크 VLM 추론(낙상 게이팅 + 설명) [ROS2 Timer/Logger 사용, 카메라 토픽 구독 없음]

- 2단계 게이팅:
   1) 빠른 Yes/No(낙상 여부)만 생성 → 토큰/세그먼트 최소화
   2) Yes일 때만 자세 설명(토큰/세그먼트 확대) + 저장/로깅
- frames_direct:
   * 프레임을 메모리에 원본으로 쌓지 않고, ROI+다운스케일 후 JPEG 바이트로만 보관(메모리↓)
   * 게이팅/설명 시 필요한 프레임만 JPEG→BGR로 디코딩하여 VLM에 투입
   * Yes일 때 저장: JPEG들을 순차 디코딩하여 VideoWriter로 스트리밍 기록(메모리↓)
- frames_direct가 불가할 때는 /dev/shm 임시파일로 폴백
- 시작 워밍업, 단계별 시간 프로파일 출력
"""

import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2, json, time, tempfile, shutil, re, numpy as np, threading
from datetime import datetime
from typing import List, Tuple, Optional, Any

try:
    cv2.setNumThreads(1)
except Exception:
    pass

import rclpy
from rclpy.node import Node

# === 외부 VLM 유틸 ===
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
    import re
    return re.sub(r'[\W_]+', ' ', (s or '')).strip().lower()

def is_yes(s: str) -> bool:
    t = text_norm(s)
    return t in YES_TOKENS or t.startswith("yes")

def is_fall_positive(qa_pairs):
    """
    규칙:
    - Q/A 목록의 첫 번째 답변이 'Yes(계열)'이면 Positive
    """
    if not qa_pairs:
        return False
    first_answer = qa_pairs[0][1]
    return is_yes(first_answer)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def now_str():
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

def parse_roi_param(val: Any) -> Optional[Tuple[int,int,int,int]]:
    if val in (None, "", []):
        return None
    if isinstance(val,(list,tuple)) and len(val)==4:
        x1,y1,x2,y2 = map(int,val)
        if x2<=x1 or y2<=y1: raise ValueError("ROI must satisfy x2>x1,y2>y1")
        return (x1,y1,x2,y2)
    if isinstance(val,str):
        parts=[p for p in re.split(r"[,\s]+",val.strip()) if p]
        if len(parts)!=4: raise ValueError("ROI string must be 4 ints")
        x1,y1,x2,y2 = map(int,parts)
        if x2<=x1 or y2<=y1: raise ValueError("ROI must satisfy x2>x1,y2>y1")
        return (x1,y1,x2,y2)
    raise ValueError("Unsupported ROI param")

def uniform_indices(total: int, n: int) -> List[int]:
    if total <= 0: return []
    n = max(1, int(n))
    if n == 1: return [total - 1]  # 마지막 프레임 하나(게이팅 최적)
    return np.linspace(0, total - 1, n).astype(int).tolist()

# ---------- JPEG 버퍼 기반 저장/로드 유틸 ----------
def crop_downscale(frame_bgr: np.ndarray, roi, target_width: int) -> np.ndarray:
    f = frame_bgr
    if roi is not None:
        x1,y1,x2,y2 = roi
        f = f[y1:y2, x1:x2]
    if target_width and target_width > 0 and f.shape[1] > target_width:
        h,w = f.shape[:2]
        scale = target_width / float(w)
        f = cv2.resize(f, (target_width, max(1, int(h*scale))), interpolation=cv2.INTER_AREA)
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

def jpegs_to_temp_video(jpegs: List[bytes], size_wh: Tuple[int,int], fps: float,
                        prefer_container: str = "avi") -> str:
    """JPEG 바이트 시퀀스를 순차 디코딩해 임시 비디오(/dev/shm)에 스트리밍 기록."""
    if not jpegs: raise ValueError("No JPEGs to write.")
    w,h = size_wh
    tmpdir = "/dev/shm" if os.path.isdir("/dev/shm") else None
    # 컨테이너 우선 순서
    if prefer_container == "avi":
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
    try:
        for b in jpegs:
            f = jpeg_bytes_to_bgr(b)
            writer.write(f)
    finally:
        writer.release()
    return path

def jpegs_sample_to_video(jpegs: List[bytes], indices: List[int], size_wh: Tuple[int,int],
                          fps: float, prefer_container: str = "avi") -> str:
    """샘플 인덱스만 영상으로 저장(메모리 절약)."""
    if not indices: raise ValueError("No indices to write.")
    w,h = size_wh
    tmpdir = "/dev/shm" if os.path.isdir("/dev/shm") else None
    if prefer_container == "avi":
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        tmp = tempfile.NamedTemporaryFile(dir=tmpdir, suffix=".avi", delete=False)
        path = tmp.name; tmp.close()
        writer = cv2.VideoWriter(path, fourcc, max(float(fps or 10.0),1.0), (w,h))
        if not writer.isOpened():
            os.remove(path)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            tmp = tempfile.NamedTemporaryFile(dir=tmpdir, suffix=".mp4", delete=False)
            path = tmp.name; tmp.close()
            writer = cv2.VideoWriter(path, fourcc, max(float(fps or 10.0),1.0), (w,h))
            if not writer.isOpened():
                os.remove(path); raise RuntimeError("VideoWriter open failed.")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        tmp = tempfile.NamedTemporaryFile(dir=tmpdir, suffix=".mp4", delete=False)
        path = tmp.name; tmp.close()
        writer = cv2.VideoWriter(path, fourcc, max(float(fps or 10.0),1.0), (w,h))
        if not writer.isOpened():
            os.remove(path)
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            tmp = tempfile.NamedTemporaryFile(dir=tmpdir, suffix=".avi", delete=False)
            path = tmp.name; tmp.close()
            writer = cv2.VideoWriter(path, fourcc, max(float(fps or 10.0),1.0), (w,h))
            if not writer.isOpened():
                os.remove(path); raise RuntimeError("VideoWriter open failed.")
    try:
        for i in indices:
            f = jpeg_bytes_to_bgr(jpegs[i])
            writer.write(f)
    finally:
        writer.release()
    return path

# ---------- 로컬 비디오 전용 ROS2 노드 ----------
class VLMGatedNode(Node):
    def __init__(self):
        super().__init__("vlm_chunk_local_video")

        # 입력/일반
        self.declare_parameter("video_path", "")           # ★ 필수: 로컬 동영상 경로
        self.declare_parameter("video_loop", False)        # 파일 끝에서 루프 재생
        self.declare_parameter("video_sleep", 0.0)         # FPS 불명/오프라인 가속 시 0.0(최대속)
        self.declare_parameter("chunk_secs", 5.0)
        self.declare_parameter("est_fps", 15.0)
        self.declare_parameter("roi", "")                  # "x1,y1,x2,y2"
        self.declare_parameter("alerts_dir", "alerts")
        self.declare_parameter("keep_raw_when_yes", True)  # Yes일 때 저장
        self.declare_parameter("target_width", 480)        # 다운스케일 폭
        self.declare_parameter("jpeg_quality", 80)         # JPEG 품질(용량/화질 트레이드오프)

        # 저장 모드
        self.declare_parameter("save_mode", "full")        # "full" | "skim"
        self.declare_parameter("save_container", "avi")    # "avi"(MJPG) | "mp4"(mp4v)
        self.declare_parameter("save_skim_fps", 8.0)       # skim 모드 FPS

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
        self.video_path  = gp("video_path").get_parameter_value().string_value
        self.video_loop  = bool(gp("video_loop").value)
        self.video_sleep = float(gp("video_sleep").value)

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

        if not self.video_path:
            raise RuntimeError("video_path 파라미터를 지정해야 합니다. (-p video_path:=/path/to/video.mp4)")

        # 경로
        self.scenes_dir = os.path.join(self.alerts_dir, "scenes")
        self.logs_dir   = os.path.join(self.alerts_dir, "logs")
        ensure_dir(self.scenes_dir); ensure_dir(self.logs_dir)
        self.jsonl_path = os.path.join(self.logs_dir, "vlm_results.jsonl")

        # VLM 로드 + 워밍업
        self.get_logger().info("Loading VLM...")
        t0 = time.monotonic()
        self.vlm_model, self.vlm_tokenizer = init_model()
        self.get_logger().info(f"VLM loaded in {time.monotonic()-t0:.2f}s | frames_direct={self.use_frames_direct}")

        if self.prewarm:
            self.get_logger().info("Warming up VLM...")
            try:
                dummy = np.zeros((224,224,3), dtype=np.uint8)
                if self.use_frames_direct:
                    _ = run_frames_inference(
                        model=self.vlm_model, tokenizer=self.vlm_tokenizer,
                        frames=[dummy],
                        generation_config={"max_new_tokens":8, "do_sample":False},
                        num_segments=1, max_num=1
                    )
                else:
                    # 1프레임 임시 비디오로 워밍업
                    path = jpegs_to_temp_video([bgr_to_jpeg_bytes(dummy)], (224,224), fps=5.0, prefer_container="avi")
                    try:
                        _ = run_video_inference(
                            model=self.vlm_model, tokenizer=self.vlm_tokenizer,
                            video_path=path,
                            generation_config={"max_new_tokens":8, "do_sample":False},
                            num_segments=1, max_num=1
                        )
                    finally:
                        try: os.remove(path)
                        except Exception: pass
            except Exception as e:
                self.get_logger().warn(f"Warm-up skipped: {e}")
            self.get_logger().info("Warm-up done.")

        # 상태 (메모리 절약: JPEG 바이트 버퍼만 유지)
        self.chunk_jpegs: List[bytes] = []        # ROI+다운스케일 후 JPEG 바이트들
        self.frame_size: Optional[Tuple[int,int]] = None  # (w,h) after crop/downscale
        self.chunk_start_t: Optional[float] = None
        self.processing = False
        self._shutdown = False

        # 입력: 로컬 비디오 리더 스레드 시작
        self._video_thread = threading.Thread(target=self._video_reader_loop, daemon=True)
        self._video_thread.start()

        # 주기 타이머: 청크 종료/처리 판정
        self.timer = self.create_timer(0.05, self.timer_cb)
        self.get_logger().info(f"[Video Mode] {self.video_path} | chunk_secs={self.chunk_secs}s")

    # ---------- 프레임 수집(공용) ----------
    def ingest_frame_bgr(self, frame: np.ndarray):
        if self.processing or frame is None:
            return
        f_small = crop_downscale(frame, self.roi, self.target_w)
        if self.frame_size is None:
            self.frame_size = (int(f_small.shape[1]), int(f_small.shape[0]))
        try:
            jpg = bgr_to_jpeg_bytes(f_small, quality=self.jpeg_quality)
            self.chunk_jpegs.append(jpg)
        except Exception as e:
            self.get_logger().warn(f"JPEG encode failed: {e}")
        if self.chunk_start_t is None:
            self.chunk_start_t = time.monotonic()

    # ---------- 비디오 리더 루프 ----------
    def _video_reader_loop(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.get_logger().error(f"Open failed: {self.video_path}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = (1.0 / fps) if fps and fps > 1e-3 else float(self.video_sleep)
        while not self._shutdown:
            ok, frame = cap.read()
            if not ok:
                if self.video_loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            self.ingest_frame_bgr(frame)
            if delay > 0:
                time.sleep(delay)
        cap.release()

    # ---------- 타이머: 5초 청크 처리 ----------
    def timer_cb(self):
        if self.chunk_start_t is None or self.processing:
            return
        elapsed = time.monotonic() - self.chunk_start_t
        if elapsed < self.chunk_secs:
            return

        # 청크 마감: 버퍼 스왑(참조 이동) 후 처리
        jpegs_to_process = self.chunk_jpegs
        size_wh = self.frame_size
        self.chunk_jpegs = []
        self.frame_size = None
        chunk_elapsed = max(elapsed, 1e-3)
        self.chunk_start_t = None
        self.processing = True
        try:
            self.process_chunk_bytes(jpegs_to_process, size_wh, chunk_elapsed)
        except Exception as e:
            self.get_logger().error(f"process_chunk_bytes error: {e}")
        finally:
            self.processing = False

    # ---------- 처리(메모리 절약 경로) ----------
    def process_chunk_bytes(self, jpegs: List[bytes], size_wh: Tuple[int,int], elapsed_sec: float):
        if not jpegs or size_wh is None:
            return

        t0 = time.monotonic()
        n = len(jpegs)
        fps = max(1.0, float(n) / float(elapsed_sec or 1.0))
        if n < 3:  # 너무 적으면 est_fps 사용
            fps = max(fps, self.est_fps)

        # ---- Stage1: 게이팅(필요한 프레임만 디코딩) ----
        self.get_logger().info("[Perf] Stage1 gate begin")
        t_gate0 = time.monotonic()
        gate_idxs = uniform_indices(n, self.gate_segments)
        gate_frames = [jpeg_bytes_to_bgr(jpegs[i]) for i in gate_idxs]

        try:
            if self.use_frames_direct:
                qa_gate = run_frames_inference(
                    model=self.vlm_model, tokenizer=self.vlm_tokenizer,
                    frames=gate_frames,
                    generation_config={"max_new_tokens": self.gate_max_tok,
                                       "do_sample": self.gate_sample},
                    num_segments=len(gate_frames), max_num=1
                )
            else:
                # frames_direct 불가 → 샘플 프레임들로 임시 비디오 생성 후 run_video_inference
                gate_clip = jpegs_sample_to_video(jpegs, gate_idxs, size_wh, fps=min(fps, 10.0),
                                                  prefer_container="avi")
                try:
                    qa_gate = run_video_inference(
                        model=self.vlm_model, tokenizer=self.vlm_tokenizer,
                        video_path=gate_clip,
                        generation_config={"max_new_tokens": self.gate_max_tok,
                                           "do_sample": self.gate_sample},
                        num_segments=len(gate_idxs), max_num=1
                    )
                finally:
                    try: os.remove(gate_clip)
                    except Exception: pass
        except Exception as e:
            self.get_logger().error(f"gate inference failed: {e}")
            return
        t_gate1 = time.monotonic()

        fall_yes = is_fall_positive(qa_gate)
        self.get_logger().info(f"[Perf] Stage1 gate end (fall={fall_yes}) elapsed={t_gate1 - t_gate0:.2f}s")

        if not fall_yes:
            self.get_logger().info("No fall: discard chunk (skip Stage2)")
            self._log_perf(t0, elapsed_sec, vlm=t_gate1 - t_gate0, frames=n)
            return

        # ---- Stage2: 설명(필요한 프레임만 디코딩) ----
        self.get_logger().info("[Perf] Stage2 desc begin")
        t_desc0 = time.monotonic()
        desc_idxs = uniform_indices(n, self.desc_segments)
        desc_frames = [jpeg_bytes_to_bgr(jpegs[i]) for i in desc_idxs]

        try:
            if self.use_frames_direct:
                qa_desc = run_frames_inference(
                    model=self.vlm_model, tokenizer=self.vlm_tokenizer,
                    frames=desc_frames,
                    generation_config={"max_new_tokens": self.desc_max_tok,
                                       "do_sample": self.desc_sample},
                    num_segments=len(desc_frames), max_num=1
                )
                saved_path = ""
            else:
                # frames_direct 불가 → 설명용 임시 비디오로 추론
                desc_clip = jpegs_sample_to_video(jpegs, desc_idxs, size_wh, fps=min(fps, 10.0),
                                                  prefer_container="avi")
                try:
                    qa_desc = run_video_inference(
                        model=self.vlm_model, tokenizer=self.vlm_tokenizer,
                        video_path=desc_clip,
                        generation_config={"max_new_tokens": self.desc_max_tok,
                                           "do_sample": self.desc_sample},
                        num_segments=len(desc_idxs), max_num=1
                    )
                finally:
                    saved_path = ""  # 추후 save_mode에 따라 별도 저장
                    try: os.remove(desc_clip)
                    except Exception: pass
        except Exception as e:
            self.get_logger().error(f"desc inference failed: {e}")
            return
        t_desc1 = time.monotonic()

        # ---- 저장(메모리 절약: JPEG→Video 스트리밍) ----
        saved_path = ""
        if self.keep_raw:
            try:
                if self.save_mode == "skim":
                    # 설명에 사용한 프레임만 차례로 기록 (FPS는 save_skim_fps)
                    tmp_path = jpegs_sample_to_video(jpegs, desc_idxs, size_wh,
                                                     fps=self.save_skim_fps,
                                                     prefer_container=self.save_container)
                else:
                    # full: 전체 JPEG 시퀀스를 스트리밍 기록
                    tmp_path = jpegs_to_temp_video(jpegs, size_wh, fps=fps,
                                                   prefer_container=self.save_container)
                # scenes/로 이동
                ext = os.path.splitext(tmp_path)[1] or (".avi" if self.save_container=="avi" else ".mp4")
                dest = unique_path(self.scenes_dir, f"fall_{now_str()}{ext}")
                try:
                    shutil.move(tmp_path, dest)
                except Exception:
                    dest = tmp_path  # 이동 실패 시 원경로 사용
                saved_path = dest
                self.get_logger().info(f"[ALERT] FALL detected → saved: {saved_path}")
            except Exception as e:
                self.get_logger().warn(f"Save streaming failed: {e}")
                saved_path = ""

        # ---- 출력 + 로깅 ----
        qa_all = qa_gate + [("-----", "-----")] + qa_desc
        ts = datetime.now().isoformat(timespec='seconds')
        print(f"\n=== [{ts}] FALL detected ===")
        for q,a in qa_all:
            print(f"Q: {q}\nA: {a}\n")

        rec = {
            "timestamp": ts,
            "source_video": self.video_path,
            "saved_path": saved_path,
            "roi": self.roi,
            "duration_sec": round(elapsed_sec,3),
            "fps_est": round(fps,2),
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
            self.get_logger().warn(f"JSONL append failed: {e}")

        self._log_perf(t0, elapsed_sec,
                       vlm=(t_gate1 - t_gate0) + (t_desc1 - t_desc0),
                       frames=n)

    def _log_perf(self, t0, collect_s, vlm, frames):
        t1 = time.monotonic()
        self.get_logger().info(
            f"[Perf] collect={collect_s:.2f}s, vlm={vlm:.2f}s, total={t1 - t0:.2f}s, frames={frames}"
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
            try:
                node._shutdown = True
                if getattr(node, "_video_thread", None):
                    node._video_thread.join(timeout=1.0)
            except Exception:
                pass
            node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
