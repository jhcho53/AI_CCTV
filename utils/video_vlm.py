# video_vlm.py
import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode!='RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(ar, target_ratios, w, h, image_size):
    best, diff = (1,1), float('inf')
    area = w*h
    for i,j in target_ratios:
        tar = i/j
        d = abs(ar - tar)
        if d < diff or (d==diff and area > 0.5*image_size*image_size*i*j):
            best, diff = (i,j), d
    return best

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    w,h = image.size; ar = w/h
    ratios = sorted(
      {(i,j) for n in range(min_num, max_num+1)
             for i in range(1,n+1) for j in range(1,n+1)
       if min_num<=i*j<=max_num},
      key=lambda x: x[0]*x[1]
    )
    i,j = find_closest_aspect_ratio(ar, ratios, w,h,image_size)
    tw,th = image_size*i, image_size*j
    img_r = image.resize((tw, th))
    blocks = i*j
    tiles = []
    cols = tw//image_size
    for idx in range(blocks):
        x0 = (idx%cols)*image_size; y0 = (idx//cols)*image_size
        tiles.append(img_r.crop((x0,y0, x0+image_size, y0+image_size)))
    if use_thumbnail and blocks>1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles

def split_model(model_name):
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    layers = cfg.llm_config.num_hidden_layers
    gpus = torch.cuda.device_count()
    per = math.ceil(layers/(gpus-0.5))
    counts = [math.ceil(per*0.5)] + [per]*(gpus-1)
    dm = {}
    cnt = 0
    for gpu, num in enumerate(counts):
        for _ in range(num):
            dm[f'language_model.model.layers.{cnt}'] = gpu
            cnt += 1
    # put all other modules on gpu0
    for key in ['vision_model','mlp1',
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
        s,e = bound
    else:
        s,e = -1e5,1e5
    si = max(first_idx, round(s*fps))
    ei = min(round(e*fps), max_frame)
    span = (ei - si) / num_segments
    return [int(si + span/2 + span*i) for i in range(num_segments)]

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr)-1; fps = float(vr.get_avg_fps())
    transform = build_transform(input_size)
    indices = get_frame_indices(bound, fps, max_frame, num_segments)
    pixel_vals, patch_counts = [], []
    for idx in indices:
        img = Image.fromarray(vr[idx].asnumpy()).convert('RGB')
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        tv = torch.stack([transform(t) for t in tiles])
        patch_counts.append(tv.shape[0])
        pixel_vals.append(tv)
    return torch.cat(pixel_vals), patch_counts

def prepare_video_prompts(num_patches_list):
    return ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])

def run_video_inference(
    model, tokenizer, video_path,
    generation_config, bound=None,
    input_size=448, max_num=1, num_segments=8
):
    pv, patch_list = load_video(video_path, bound, input_size, max_num, num_segments)
    device = next(model.parameters()).device
    pv = pv.to(torch.bfloat16).to(device)
    prefix = prepare_video_prompts(patch_list)

    # 첫 질문
    q1 = prefix + 'Did someone break in?'
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
