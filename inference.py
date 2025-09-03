# -*- coding: utf-8 -*-
# 카메라/RTSP를 '파일 모드' 파이프라인(run_inference)로 지속적으로 녹화-분석
# TensorFlow는 강제로 비활성화

import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")   # Transformers가 TF 백엔드 로드하지 않도록
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import time
import json
import csv
import shutil
import tempfile
import re
from datetime import datetime

# OpenCV 멀티 스레드 레이스 이슈 완화
try:
    cv2.setNumThreads(1)
except Exception:
    pass

from utils.inf_utils import run_inference
from utils.video_vlm import init_model


# -------------------------------
# 유틸
# -------------------------------
def parse_source(src: str):
    """'0' 같은 숫자 문자열은 int(웹캠 인덱스)로 변환."""
    if isinstance(src, str) and src.isdigit():
        return int(src)
    return src

def _backend_to_flag(name: str):
    name = (name or "auto").lower()
    if name == "v4l2":
        return cv2.CAP_V4L2
    if name == "ffmpeg":
        return cv2.CAP_FFMPEG
    if name == "gstreamer":
        return cv2.CAP_GSTREAMER
    return 0  # auto

def _resolve_fps(cap, default=30.0):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1.0:
        return float(default)
    return float(fps)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def text_norm(s: str) -> str:
    return re.sub(r'[\W_]+', ' ', (s or '')).strip().lower()

NO_TOKENS  = {"no", "n", "false", "0", "nope", "nah"}

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
    - Q/A 목록의 첫 번째 답변이 'Yes(계열)'이면 Positive
    - 질문 내용(키워드)은 더 이상 사용하지 않음
    """
    if not qa_pairs:
        return False
    # qa_pairs: List[Tuple[question, answer]]
    first_answer = qa_pairs[0][1]
    return is_yes(first_answer)


def unique_path(base_dir: str, base_name: str) -> str:
    """중복 시 _1, _2 ... 붙여서 유니크 경로 반환."""
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

def append_csv(csv_path: str, row: dict, field_order: list):
    ensure_dir(os.path.dirname(csv_path) or ".")
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


# -------------------------------
# 녹화 -> 파일
# -------------------------------
def _try_open_writer(path, size, fps, prefer="avi"):
    """MJPG(AVI) 우선, 실패 시 mp4v(MP4) 폴백."""
    w, h = size
    if prefer == "avi":
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(path, fourcc, max(fps, 1.0), (w, h))
        if writer.isOpened():
            return writer, path
        base, _ = os.path.splitext(path)
        path_mp4 = base + ".mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(path_mp4, fourcc, max(fps, 1.0), (w, h))
        if writer.isOpened():
            return writer, path_mp4
        raise RuntimeError("Failed to open VideoWriter for both MJPG(AVI) and mp4v(MP4).")
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(path, fourcc, max(fps, 1.0), (w, h))
        if writer.isOpened():
            return writer, path
        base, _ = os.path.splitext(path)
        path_avi = base + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(path_avi, fourcc, max(fps, 1.0), (w, h))
        if writer.isOpened():
            return writer, path_avi
        raise RuntimeError("Failed to open VideoWriter for both mp4v(MP4) and MJPG(AVI).")


def record_camera_to_file(
    source,
    duration_secs=10.0,
    backend="auto",
    width=None,
    height=None,
    prefer_container="avi",
    preview=False
):
    """
    카메라/RTSP/GStreamer 입력을 duration_secs 만큼 녹화하여 임시 파일로 저장.
    반환: (out_path, fps)
    """
    cap_flag = _backend_to_flag(backend)
    # GStreamer 파이프 문자열이면 CAP_GSTREAMER 강제
    if isinstance(source, str) and "!" in source:
        cap_flag = cv2.CAP_GSTREAMER

    cap = cv2.VideoCapture(source, cap_flag)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source} (backend={backend})")

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

    ok, first = cap.read()
    if not ok or first is None:
        cap.release()
        raise RuntimeError("Failed to read first frame from source.")

    h, w = first.shape[:2]
    fps = _resolve_fps(cap, default=30.0)

    tmp = tempfile.NamedTemporaryFile(suffix=".avi" if prefer_container == "avi" else ".mp4", delete=False)
    raw_path = tmp.name
    tmp.close()

    writer, out_path = _try_open_writer(raw_path, (w, h), fps, prefer="avi" if prefer_container == "avi" else "mp4")

    # 첫 프레임 기록
    writer.write(first)

    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        writer.write(frame)

        if preview:
            try:
                cv2.imshow("Recording (press q to stop this chunk)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception:
                preview = False

        if time.time() - t0 >= float(duration_secs):
            break

    writer.release()
    cap.release()
    if preview:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    return out_path, fps


# -------------------------------
# 메인: 지속 녹화 -> 파일 모드 분석 -> 결과 게이팅/저장/로깅
# -------------------------------
def main():
    """
    지속(무한 루프)으로 카메라/RTSP를 녹화하여 파일 모드(run_inference)로 분석.
    VLM Q/A의 첫 질문이 'Did someone fallen?'이고 답변이 'Yes'이면:
      - 해당 세그먼트 비디오를 alerts/scenes/ 에 저장
      - VLM 결과를 alerts/logs/fall_events.csv 에 로깅
    그 외는 저장 안 함(세그먼트 파일 즉시 삭제).
    """
    import argparse
    parser = argparse.ArgumentParser(description='Continuous anomaly detection via file-mode on recorded camera chunks')

    # 기존 파일 모드 인자
    parser.add_argument('--video', default=None, help='(옵션) 직접 파일 분석 시 경로. 제공 시 한 번만 분석하고 종료.')
    parser.add_argument('--model', default='anomaly_single_balanced.pth', help='Model .pth file path')
    parser.add_argument('--seq_len', type=int, default=8, help='Sequence length for GRU inference')
    parser.add_argument('--threshold', type=float, default=0.3, help='Anomaly probability threshold')
    parser.add_argument('--roi', nargs=4, type=int, default=None,
                        help='ROI as x1 y1 x2 y2 (omit to use full frame)')
    parser.add_argument('--vlm_segs', type=int, default=8, help='Number of segments for VLM inference')
    parser.add_argument('--vlm_max', type=int, default=1, help='Max patches per frame for VLM inference')

    # 카메라/RTSP 관련
    parser.add_argument('--source', default=None,
                        help='Camera index (e.g., 0) or RTSP/HTTP/GStreamer pipeline. '
                             '지정되면 지속적으로 녹화-분석을 반복.')
    parser.add_argument('--backend', default='auto', choices=['auto','v4l2','ffmpeg','gstreamer'],
                        help='Force a specific capture backend for camera/RTSP')
    parser.add_argument('--record_secs', type=float, default=10.0, help='Chunk duration in seconds')
    parser.add_argument('--rec_width', type=int, default=None, help='Optional capture width')
    parser.add_argument('--rec_height', type=int, default=None, help='Optional capture height')
    parser.add_argument('--preview', action='store_true', help='Show preview while recording each chunk')

    # 출력/로그
    parser.add_argument('--alerts_dir', default='alerts', help='Base directory to store fall scenes and logs')
    parser.add_argument('--keep_raw', action='store_true', help='Keep the recorded raw chunk files')
    parser.add_argument('--keep_nonfall_segments', action='store_true',
                        help='Keep anomaly segments even when fall=No (기본은 삭제)')

    args = parser.parse_args()

    # ROI 파싱 및 검증
    roi = tuple(args.roi) if args.roi is not None else None
    if roi is not None:
        x1, y1, x2, y2 = roi
        if x2 <= x1 or y2 <= y1:
            parser.error("--roi must be x1 y1 x2 y2 with x2>x1 and y2>y1")

    # 디렉터리 준비
    scenes_dir = os.path.join(args.alerts_dir, "scenes")
    logs_dir   = os.path.join(args.alerts_dir, "logs")
    ensure_dir(scenes_dir)
    ensure_dir(logs_dir)

    csv_log_path = os.path.join(logs_dir, "fall_events.csv")
    csv_fields = [
        "timestamp", "source", "raw_clip", "segment_path", "saved_path",
        "fall_positive", "first_question", "first_answer", "qa_json",
        "roi", "seq_len", "threshold", "vlm_segments", "vlm_max_num"
    ]

    # VLM 로드
    vlm_model, vlm_tokenizer = init_model()

    # 1) 파일만 지정된 경우: 한 번만 돌고 종료
    if args.video is not None:
        outputs = run_inference(
            video_path=args.video,
            model_path=args.model,
            seq_len=args.seq_len,
            roi=roi,
            threshold=args.threshold,
            vlm_model=vlm_model,
            vlm_tokenizer=vlm_tokenizer,
            generation_config={"max_new_tokens": 512, "do_sample": True},
            vlm_segments=args.vlm_segs,
            vlm_max_num=args.vlm_max
        )
        # 게이팅 및 저장/로깅
        for seg_path, qa in outputs.items():
            fall_pos = is_fall_positive(qa)
            first_q = qa[0][0] if qa else ""
            first_a = qa[0][1] if qa else ""
            if fall_pos:
                base_name = f"fall_{now_str()}_{os.path.basename(seg_path)}"
                dest_path = unique_path(scenes_dir, base_name)
                ensure_dir(os.path.dirname(dest_path))
                shutil.move(seg_path, dest_path)
                print(f"[ALERT] FALL detected → saved: {dest_path}")
                append_csv(csv_log_path, {
                    "timestamp": datetime.now().isoformat(timespec='seconds'),
                    "source": args.video,
                    "raw_clip": args.video,
                    "segment_path": seg_path,
                    "saved_path": dest_path,
                    "fall_positive": True,
                    "first_question": first_q,
                    "first_answer": first_a,
                    "qa_json": json.dumps(qa, ensure_ascii=False),
                    "roi": str(roi),
                    "seq_len": args.seq_len,
                    "threshold": args.threshold,
                    "vlm_segments": args.vlm_segs,
                    "vlm_max_num": args.vlm_max
                }, csv_fields)
            else:
                if not args.keep_nonfall_segments:
                    try:
                        os.remove(seg_path)
                    except Exception:
                        pass
        return

    # 2) 지속 녹화 루프
    if args.source is None:
        parser.error("Provide either --video (single run) or --source (continuous camera/RTSP mode).")

    source = parse_source(args.source)
    print(f"[INFO] Continuous mode. Press Ctrl+C to stop.\n"
          f"       Recording chunks of {args.record_secs:.1f}s from source={args.source} (backend={args.backend})")

    try:
        while True:
            # (a) 녹화
            try:
                raw_clip, fps = record_camera_to_file(
                    source=source,
                    duration_secs=args.record_secs,
                    backend=args.backend,
                    width=args.rec_width,
                    height=args.rec_height,
                    prefer_container="avi",
                    preview=args.preview
                )
            except Exception as e:
                print(f"[ERROR] Recording failed: {e}")
                break

            # (b) 파일 모드 분석
            try:
                outputs = run_inference(
                    video_path=raw_clip,
                    model_path=args.model,
                    seq_len=args.seq_len,
                    roi=roi,
                    threshold=args.threshold,
                    vlm_model=vlm_model,
                    vlm_tokenizer=vlm_tokenizer,
                    generation_config={"max_new_tokens": 512, "do_sample": True},
                    vlm_segments=args.vlm_segs,
                    vlm_max_num=args.vlm_max
                )
            except Exception as e:
                print(f"[ERROR] Inference failed on {raw_clip}: {e}")
                # 원본 클립 정리
                if not args.keep_raw:
                    try:
                        os.remove(raw_clip)
                    except Exception:
                        pass
                continue  # 다음 루프로 진행

            # (c) 게이팅 + 저장/로깅
            any_saved = False
            for seg_path, qa in outputs.items():
                fall_pos = is_fall_positive(qa)
                first_q = qa[0][0] if qa else ""
                first_a = qa[0][1] if qa else ""
                if fall_pos:
                    base_name = f"fall_{now_str()}_{os.path.basename(seg_path)}"
                    dest_path = unique_path(scenes_dir, base_name)
                    ensure_dir(os.path.dirname(dest_path))
                    try:
                        shutil.move(seg_path, dest_path)
                        any_saved = True
                        print(f"[ALERT] FALL detected → saved: {dest_path}")
                    except Exception as e:
                        print(f"[WARN] Failed to move segment: {e}")
                        dest_path = ""  # 로깅상 빈값

                    append_csv(csv_log_path, {
                        "timestamp": datetime.now().isoformat(timespec='seconds'),
                        "source": args.source,
                        "raw_clip": raw_clip,
                        "segment_path": seg_path,
                        "saved_path": dest_path,
                        "fall_positive": True,
                        "first_question": first_q,
                        "first_answer": first_a,
                        "qa_json": json.dumps(qa, ensure_ascii=False),
                        "roi": str(roi),
                        "seq_len": args.seq_len,
                        "threshold": args.threshold,
                        "vlm_segments": args.vlm_segs,
                        "vlm_max_num": args.vlm_max
                    }, csv_fields)
                else:
                    # No인 경우 저장 안 함 → 세그먼트 파일 삭제(기본)
                    if not args.keep_nonfall_segments:
                        try:
                            os.remove(seg_path)
                        except Exception:
                            pass

            # (d) 원본 녹화 파일 정리
            if not args.keep_raw and os.path.exists(raw_clip):
                try:
                    os.remove(raw_clip)
                except Exception:
                    pass

            if not any_saved:
                print(f"[INFO] No fall detected in chunk ({os.path.basename(raw_clip)}).")

            # 다음 chunk로 계속 (무한 루프). 종료는 Ctrl+C
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")


if __name__ == '__main__':
    main()
