from utils.inf_utils import run_inference
from utils.video_vlm import init_model

def main():
    """
    Command-line interface for anomaly inference.
    """
    import argparse
    parser = argparse.ArgumentParser(description='Anomaly detection and export')
    parser.add_argument('--video', default='/home/jaehyeon/Desktop/졸작/BTS_graduate_project/video/E01_001.mp4',
                        help='Path to input video')
    parser.add_argument('--model', default='anomaly_single_balanced.pth', help='Model .pth file path')
    parser.add_argument('--seq_len', type=int, default=8, help='Sequence length for GRU inference')
    parser.add_argument('--threshold', type=float, default=0.5, help='Anomaly probability threshold')
    parser.add_argument('--roi', nargs=4, type=int,
                        default=[1150,300,1600,700],
                        help='ROI as x1 y1 x2 y2')
    parser.add_argument('--vlm_segs', type=int, default=8,
                        help='Number of segments for VLM inference')
    parser.add_argument('--vlm_max', type=int, default=1,
                        help='Max patches per frame for VLM inference')
    args = parser.parse_args()

    # VLM 모델/토크나이저 로드
    vlm_model, vlm_tokenizer = init_model()

    # 전체 파이프라인 실행
    outputs = run_inference(
        video_path=args.video,
        model_path=args.model,
        seq_len=args.seq_len,
        roi=tuple(args.roi),
        threshold=args.threshold,
        vlm_model=vlm_model,
        vlm_tokenizer=vlm_tokenizer,
        generation_config={"max_new_tokens":512, "do_sample":True},
        vlm_segments=args.vlm_segs,
        vlm_max_num=args.vlm_max
    )

    for path, qa in outputs.items():
        print(f"=== {path} ===")
        for q, a in qa:
            print(f"Q: {q}\nA: {a}\n")

if __name__ == '__main__':
    main()