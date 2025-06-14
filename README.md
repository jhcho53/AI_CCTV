# BTS_graduate_project

## Project Overview  
This repository implements a deep learning pipeline for **video-based anomaly detection**. Datasets are constructed via `dataset.py`, various models are defined under the `model/` directory, training is performed with `train.py` and `single_train.py`, and real-time or batch inference is supported by `inference.py`.

## Full Pipeline

![Pipeline Overview](img/Pipeline.png)

### Pipeline Description

1. **Clip Extraction**  
   - Trim the original video to the desired `start`–`end` timestamp.

2. **Frame Sampling**  
   - Uniformly sample N frames from the trimmed clip.

3. **Question Definition**  
   - e.g., `Q1: Did someone break in?`  
   - e.g., `Q2: Describe this scene.`

4. **Tokenizer**  
   - Tokenize the text questions into input sequences understandable by the VLM.

5. **VLM (Vision–Language Model)**  
   - Fuse the extracted frame features with the tokenized questions and process them multimodally.

6. **Output**  
   - Generate descriptive answers or predictions about the anomaly.

## Detector Structure

![Detector Architecture](img/Detector.png)

### Detector Module

The Detector consists of three main stages:

1. **Encoder**  
   A CNN-based backbone (e.g., ResNet-50) that extracts spatial features from each frame and produces per-frame embedding vectors.

2. **GRU**  
   A Gated Recurrent Unit that takes the sequence of embeddings as input to model temporal continuity and learns dynamic changes across frames.

3. **MLP**  
   A multilayer perceptron that takes the final hidden state from the GRU and regresses an **Anomaly Score**.

---
## Directory Structure

```plaintext
BTS_graduate_project/
├── model/
│   └── model.py
├── utils/
│   ├── check_ROI.py
│   ├── inf_utils.py
│   └── video_vlm.py
├── dataset.py
├── train.py             # Multi-GPU training on the full dataset
├── single_train.py      # Single-video training script
├── inference.py         # Anomaly detection inference script
├── requirements.txt     # List of dependencies
├── .gitignore
└── README.md
```

## Installation & Setup

1. Clone the repository
   ```bash
   git clone https://github.com/jhcho53/BTS_graduate_project.git
   cd BTS_graduate_project

2. Create & activate Python environment  
   ```bash
   conda env create -f environment.yml
   conda activate bts

3. Train the model
   ```bash
   python train.py \
   --data_root $BTS_DATA_ROOT \
   --epochs 50 \
   --batch_size 16 \
   --cuda
4. Evaluate the model
   ```bash
   python eval.py
5. Run inference
   ```bash
   python inference.py --video $Video_Root \
   --model $Model_Root\
   
