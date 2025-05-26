# BTS_graduate_project

## 프로젝트 개요
본 프로젝트는 **영상 기반 이상 행동(Anomaly) 탐지**를 목적으로 한 딥러닝 파이프라인을 구현한 리포지터리입니다.  
`dataset.py`를 통해 데이터셋을 구성하고, `model/` 디렉토리 내 다양한 모델을 정의하여 `train.py`와 `sigle_train.py`로 학습을 수행하며, `inference.py`로 실시간 또는 배치 추론을 지원합니다.

---

## 주요 기능
- **데이터 로딩 및 전처리** (`dataset.py`)  
- **모델 정의 모듈** (`model/`):
  - GRU 기반 시퀀스 모델  
- **유틸리티 함수** (`utils/`):
  - 학습 로그 출력, 체크포인트 관리, 시각화 도구 등  
- **학습 스크립트**  
  - `train.py`: 전체 데이터셋 기반 멀티-GPU 학습  
  - `single_train.py`: 단일 비디오 학습  
- **추론 스크립트** (`inference.py`):
  - 저장된 체크포인트 로드 후 동영상 시퀀스 이상 탐지  

---

## 디렉토리 구조

```plaintext
BTS_graduate_project/
├── model/
│   ├── __init__.py
│   ├── gru_model.py
│   └── transformer_model.py
├── utils/
│   ├── logger.py
│   └── metrics.py
├── dataset.py
├── train.py             # 전체 데이터셋 기반 멀티-GPU 학습
├── single_train.py      # 단일 비디오 학습 스크립트
├── inference.py         # 이상 탐지 추론 스크립트
├── requirements.txt     # 의존성 목록
├── .gitignore
└── README.md
