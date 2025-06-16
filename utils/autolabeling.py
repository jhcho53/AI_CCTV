import os
import json
from PIL import Image
import torch
import language_tool_python
from transformers import LlavaProcessor, LlavaForConditionalGeneration

MODEL_NAME = "liuhaotian/llava-v1.5-13b"
processor = LlavaProcessor.from_pretrained(MODEL_NAME)
model = LlavaForConditionalGeneration.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

tool = language_tool_python.LanguageTool("en-US")

# 후처리용 키워드 목록
HAZARD_KEYWORDS = {
    "fire", "smoke", "explosion", "weapon", "knife", "gun",
    "fight", "accident", "collision", "fall", "bleeding",
    "spill", "electrocution", "danger", "hazard"
}

def clean_and_validate(description: str) -> str:
    """
    1) 불필요한 특수문자/여백 제거
    2) 길이 필터링
    3) 키워드 포함 여부 확인
    4) 문법 검사: 오류가 3개 초과면 자동 교정 시도, 
       교정 후에도 오류가 5개 초과하면 'No hazardous situation detected'
    """
    # 1) 기본 정제
    desc = description.strip().replace("\n", " ")
    # 2) 길이 필터링
    if len(desc) < 10:
        return "No hazardous situation detected"
    if len(desc) > 200:
        desc = desc[:197].rstrip() + "..."
    # 3) 키워드 포함 여부
    low = desc.lower()
    if not any(kw in low for kw in HAZARD_KEYWORDS):
        return "No hazardous situation detected"
    # 4) 문법 검사
    matches = tool.check(desc)
    # 오류가 많으면 자동 교정
    if len(matches) > 3:
        corrected = tool.correct(desc)
        desc = corrected
        matches = tool.check(desc)
    # 교정 후에도 오류가 너무 많으면 필터링
    if len(matches) > 5:
        return "No hazardous situation detected"
    return desc

def describe_hazard(image_path: str) -> dict:
    """
    이미지에서 위험 상황 묘사를 생성하고 후처리하여 반환.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(
        images=image,
        text="Please describe any dangerous or hazardous situations in the image.",
        return_tensors="pt"
    ).to(device)
    gen_ids = model.generate(**inputs, max_new_tokens=60)
    raw_desc = processor.decode(gen_ids[0], skip_special_tokens=True)
    clean_desc = clean_and_validate(raw_desc)
    return {
        "image": os.path.basename(image_path),
        "description": clean_desc
    }

def main(
    image_folder: str = "./images",
    output_json: str = "hazard_descriptions.json"
):
    results = []
    for fname in os.listdir(image_folder):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(image_folder, fname)
            result = describe_hazard(path)
            results.append(result)
            print(f"[Processed] {fname} → \"{result['description']}\"")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(results)} descriptions to {output_json}")

if __name__ == "__main__":
    main(
        image_folder="path/to/your/images",
        output_json="hazard_descriptions.json"
    )
