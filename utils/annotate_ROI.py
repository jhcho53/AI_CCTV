import os
import glob
import json

def add_roi_to_jsons(label_dir, default_roi=(1200, 380, 1700, 850)):
    """
    label_dir 내 모든 JSON 파일을 열어 annotations['roi'] 필드를 추가합니다.
    이미 'roi'가 있으면 덮어쓰며, 없으면 새로 삽입합니다.
    
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
    label_dir = "/home/jaehyeon/trash"  # JSON들이 들어있는 디렉토리
    # 기본 ROI 대신 파일별로 다른 값을 쓰고 싶다면, 위 함수 호출 전에
    # 파일명에 따라 ROI 값을 매핑해 두고 default_roi 대신 매핑값을 넣어주면 됩니다.
    add_roi_to_jsons(label_dir, default_roi=(860, 170, 1200, 450))
