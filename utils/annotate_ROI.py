import os
import glob
import json
roi = (1050, 100, 1300, 350)

def add_roi_to_jsons(label_dir, default_roi=(900, 0, 1300, 300)):
    """
    label_dir 내 모든 JSON 파일을 열어 annotations['roi'] 필드를 추가
    이미 'roi'가 있으면 덮어쓰며, 없으면 새로 삽입
    
    Args:
        label_dir (str): JSON 파일들이 위치한 디렉토리 경로
        default_roi (tuple): 삽입할 ROI 값 (x1, y1, x2, y2)
    """
    json_paths = glob.glob(os.path.join(label_dir, '*.json'))
    for path in json_paths:
        # JSON 로드
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # annotations 필드가 없으면 생성
        if 'annotations' not in data:
            data['annotations'] = {}

        # roi 추가/갱신
        data['annotations']['roi'] = list(default_roi)

        # 덮어쓰기 방식으로 저장
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Updated ROI in: {os.path.basename(path)}")

if __name__ == "__main__":
    label_dir = "/home/jaehyeon/Desktop/졸작/BTS_graduate_project/Dataset/Label/trash"  # JSON들이 들어있는 디렉토리
    add_roi_to_jsons(label_dir, default_roi=(550, 100, 800, 250))
