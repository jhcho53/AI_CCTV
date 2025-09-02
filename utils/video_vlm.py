# video_vlm.py
import math
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig

# ---------- Optional backends ----------
# Try decord first (best for random access). If unavailable, fall back to OpenCV or PyAV.
try:
    from decord import VideoReader, cpu
    _USE_DECORD = True
except Exception:
    VideoReader = None
    cpu = None
    _USE_DECORD = False

try:
    import cv2
    _HAVE_CV2 = True
except Exception:
    cv2 = None
    _HAVE_CV2 = False

try:
    import av
    _HAVE_AV = True
except Exception:
    av = None
    _HAVE_AV = False
# --------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def find_closest_aspect_ratio(ar, target_ratios, w, h, image_size):
    best, diff = (1, 1), float('inf')
    area = w * h
    for i, j in target_ratios:
        tar = i / j
        d = abs(ar - tar)
        if d < diff or (d == diff and area > 0.5 * image_size * image_size * i * j):
            best, diff = (i, j), d
    return best


def dynamic_preprocess(image: Image.Image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    w, h = image.size
    ar = w / h
    ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1]
    )
    i, j = find_closest_aspect_ratio(ar, ratios, w, h, image_size)
    tw, th = image_size * i, image_size * j
    img_r = image.resize((tw, th))
    blocks = i * j
    tiles = []
    cols = tw // image_size
    for idx in range(blocks):
        x0 = (idx % cols) * image_size
        y0 = (idx // cols) * image_size
        tiles.append(img_r.crop((x0, y0, x0 + image_size, y0 + image_size)))
    if use_thumbnail and blocks > 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def split_model(model_name):
    """Heuristic device_map splitter; if no CUDA, return {} to keep on CPU."""
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    # Some configs nest LLM depth under cfg.llm_config
    layers = getattr(getattr(cfg, 'llm_config', cfg), 'num_hidden_layers', None)
    if layers is None:
        raise ValueError("Cannot determine num_hidden_layers from config.")
    gpus = torch.cuda.device_count()
    if gpus <= 0:
        return {}  # CPU fallback

    per = math.ceil(layers / (gpus - 0.5))
    counts = [math.ceil(per * 0.5)] + [per] * (gpus - 1)
    dm = {}
    cnt = 0
    for gpu, num in enumerate(counts):
        for _ in range(num):
            if cnt >= layers:
                break
            dm[f'language_model.model.layers.{cnt}'] = gpu
            cnt += 1
        if cnt >= layers:
            break
    # put all other modules on gpu0
    for key in ['vision_model', 'mlp1',
                'language_model.model.tok_embeddings',
                'language_model.model.embed_tokens',
                'language_model.output',
                'language_model.model.norm',
                'language_model.model.rotary_emb',
                'language_model.lm_head',
                f'language_model.model.layers.{layers-1}']:
        dm[key] = 0
    return dm


def init_model(path='OpenGVLab/InternVL3-1B', device_map=None):
    if device_map is None:
        device_map = split_model(path)
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    return model, tokenizer


def get_frame_indices(bound, fps, max_frame, num_segments, first_idx=0):
    if bound:
        s, e = bound
    else:
        s, e = -1e5, 1e5
    si = max(first_idx, round(s * fps))
    ei = min(round(e * fps), max_frame)
    span = max((ei - si) / float(max(1, num_segments)), 0.0)
    # Ensure indices lie within [si, ei] and are ints
    return [int(min(max_frame, max(first_idx, si + span/2 + span * i))) for i in range(num_segments)]


# ------------------------- Backend helpers -------------------------
def _safe_fps(val, default=25.0):
    try:
        v = float(val)
        return v if v > 1e-6 else default
    except Exception:
        return default


def _meta_decord(video_path):
    """Open decord VR and return (vr, fps, max_frame)."""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    fps = _safe_fps(vr.get_avg_fps(), default=25.0)
    max_frame = len(vr) - 1
    return vr, fps, max_frame


def _frames_decord(vr, indices):
    """Return list of RGB numpy arrays using decord (random access)."""
    frames = []
    for idx in indices:
        arr = vr[idx].asnumpy()  # RGB
        frames.append(arr)
    return frames


def _meta_opencv(video_path):
    """OpenCV meta: (cap, fps, max_frame)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video via OpenCV: {video_path}")
    fps = _safe_fps(cap.get(cv2.CAP_PROP_FPS), default=25.0)
    # CAP_PROP_FRAME_COUNT can be 0 on some containers; handle later
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frame = total - 1 if total > 0 else None
    return cap, fps, max_frame


def _frames_opencv(cap, indices):
    """Random-access reads using OpenCV; falls back to nearest repeat if missing."""
    frames = []
    last_valid = None
    for idx in indices:
        ok = cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok2, bgr = cap.read()
        if ok and ok2 and bgr is not None:
            rgb = bgr[:, :, ::-1]
            last_valid = rgb
            frames.append(rgb)
        else:
            if last_valid is None:
                raise RuntimeError(f"Failed to read frame {idx} via OpenCV and no prior frame to fallback.")
            frames.append(last_valid)
    return frames


def _meta_pyav(video_path):
    """PyAV meta: (fps, max_frame). max_frame may require a counting pass."""
    container = av.open(video_path)
    streams = [s for s in container.streams if s.type == 'video']
    if not streams:
        container.close()
        raise RuntimeError("No video stream found in file.")
    vstream = streams[0]
    fps = _safe_fps(float(vstream.average_rate) if vstream.average_rate else None, default=25.0)
    # Try using stream.frames if available; otherwise count
    if getattr(vstream, "frames", 0):
        max_frame = int(vstream.frames) - 1
        container.close()
        return fps, max_frame
    # Fallback: count frames
    cnt = 0
    for _ in container.decode(video=0):
        cnt += 1
    container.close()
    max_frame = max(0, cnt - 1)
    return fps, max_frame


def _frames_pyav(video_path, indices):
    """Sequentially decode with PyAV and pick requested frames."""
    idx_set = set(int(i) for i in indices)
    max_idx = max(idx_set) if idx_set else -1
    out = {}
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        # Decode sequentially until we covered needed frames
        for i, frame in enumerate(container.decode(stream)):
            if i in idx_set:
                out[i] = frame.to_ndarray(format="rgb24")
            if i >= max_idx and len(out) == len(idx_set):
                break
    # Assemble in original order, repeat last valid if missing
    frames, last_valid = [], None
    for i in indices:
        arr = out.get(int(i), None)
        if arr is None:
            if last_valid is None:
                raise RuntimeError(f"Missing frame {i} in PyAV decode and no prior frame.")
            arr = last_valid
        frames.append(arr)
        last_valid = arr
    return frames
# ------------------------------------------------------------------


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    """
    Load frames from video_path using one of:
      - decord (if available),
      - OpenCV (fallback),
      - PyAV (fallback of fallback).
    Returns:
      pixel_values: torch.FloatTensor [sum_patches, 3, H, W] (bfloat16 casting is done later)
      patch_counts: List[int] number of image patches per frame
    """
    transform = build_transform(input_size)

    # Try decord first
    backend_used = None
    frames_np = []
    patch_counts = []

    try:
        if _USE_DECORD:
            vr, fps, max_frame = _meta_decord(video_path)
            indices = get_frame_indices(bound, fps, max_frame, num_segments)
            frames_np = _frames_decord(vr, indices)
            backend_used = 'decord'
        else:
            raise RuntimeError("Decord not available")
    except Exception:
        # OpenCV fallback
        if _HAVE_CV2:
            try:
                cap, fps, max_frame = _meta_opencv(video_path)
                if max_frame is None:
                    # OpenCV didn't give frame count; try PyAV to get max_frame if available
                    if _HAVE_AV:
                        try:
                            fps2, max_frame2 = _meta_pyav(video_path)
                            # Prefer OpenCV's fps if valid; else use PyAV's
                            fps = fps if fps > 1e-6 else fps2
                            max_frame = max_frame2
                        except Exception:
                            # Leave max_frame as None; we'll still try reading sequentially
                            pass
                    if max_frame is None:
                        # As a last resort, estimate max_frame by probing sequentially once (may be slow)
                        count = 0
                        ok = True
                        while ok:
                            ok, frm = cap.read()
                            if ok:
                                count += 1
                        cap.release()
                        cap = cv2.VideoCapture(video_path)
                        max_frame = max(0, count - 1)

                indices = get_frame_indices(bound, fps, max_frame, num_segments)
                frames_np = _frames_opencv(cap, indices)
                cap.release()
                backend_used = 'opencv'
            except Exception:
                backend_used = None

        # PyAV fallback
        if backend_used is None and _HAVE_AV:
            fps, max_frame = _meta_pyav(video_path)
            indices = get_frame_indices(bound, fps, max_frame, num_segments)
            frames_np = _frames_pyav(video_path, indices)
            backend_used = 'pyav'

    if backend_used is None or not frames_np:
        raise ImportError(
            "No usable video backend found. Install one of:\n"
            "- decord (source build on Jetson), or\n"
            "- OpenCV (apt-get install -y python3-opencv), or\n"
            "- PyAV (pip install av)."
        )

    # Preprocess into tiles/patches
    pixel_vals = []
    for arr in frames_np:
        img = Image.fromarray(arr).convert('RGB')
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        tv = torch.stack([transform(t) for t in tiles])
        patch_counts.append(tv.shape[0])
        pixel_vals.append(tv)

    return torch.cat(pixel_vals, dim=0), patch_counts


def prepare_video_prompts(num_patches_list):
    return ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])


def run_video_inference(
    model, tokenizer, video_path,
    generation_config, bound=None,
    input_size=448, max_num=1, num_segments=8
):
    """
    Loads video frames, prepares prompts, and runs a two-turn chat:
      1) 'Did someone fallen?'
      2) 'Describe this scene.'
    Returns: list of (question, response) tuples.
    """
    pv, patch_list = load_video(video_path, bound, input_size, max_num, num_segments)
    device = next(model.parameters()).device
    pv = pv.to(torch.bfloat16).to(device)
    prefix = prepare_video_prompts(patch_list)

    # 첫 질문
    q1 = prefix + 'Did someone fallen?'
    resp1, hist = model.chat(
        tokenizer, pv, q1, generation_config,
        num_patches_list=patch_list, history=None, return_history=True
    )

    # 후속 질문
    q2 = 'Describe this scene.'
    resp2, hist2 = model.chat(
        tokenizer, pv, q2, generation_config,
        num_patches_list=patch_list, history=hist, return_history=True
    )

    return [(q1, resp1), (q2, resp2)]
