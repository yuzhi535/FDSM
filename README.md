

# FDSM: Frequency-Aware Diffusion with Curriculum Semantic Guidance for Zero-Shot Skeleton Action Recognition

> Accepted by *The Visual Computer*.

This repository contains the official implementation of **FDSM** (Frequency-Aware Diffusion for Skeleton-Text Matching), as described in our paper "Frequency-Enhanced Diffusion Models: Curriculum-Guided Semantic Alignment for Zero-Shot Skeleton Action Recognition".

---

## 📖 Introduction

Zero-Shot Skeleton Action Recognition (ZSAR) aims to recognize unseen actions by transferring knowledge from seen classes via semantic descriptions. Current generative methods suffer from **spectral bias**, where diffusion models act as low-pass filters, smoothing out fine-grained motion details essential for discrimination.

**FDSM** addresses this by:
1.  **Semantic-Guided Spectral Residual Module (SG-SRM)**: Explicitly amplifies high-frequency dynamics gated by semantic priors.
2.  **Timestep-Adaptive Spectral Loss**: Enforces frequency consistency dynamicially during the denoising process.
3.  **Curriculum-based Semantic Abstraction**: Uses a "rich-to-sparse" curriculum to bridge the gap between detailed kinematic descriptions and sparse class labels.

---

## ⚙️ Requirements
> - Python >= 3.9.19
> - PyTorch >= 2.4.0
> - Platforms: Ubuntu 22.04, CUDA 11.8
> - We have included a dependency file for our experimental environment. To install all dependencies, create a new Anaconda virtual environment and execute the provided file. Run `conda env create -f requirements.yaml`.

## 📁 Data Preparation

We follow the evaluation setup from [SynSE](https://github.com/skelemoa/synse-zsl), [PURLS](https://github.com/azzh1/PURLS), and [SMIE](https://github.com/YujieOuO/SMIE).

Download the **pre-extracted skeleton features** (for SynSE and SMIE settings) and **class descriptions** from [SA-DVAE](https://github.com/pha123661/SA-DVAE).
Then, arrange them as follows:

```bash
data
  ├──sk_feats
  │   ├── shift_ntu60_5_r
  │   ├── shift_ntu60_12_r
  │   ├── shift_ntu60_20_r
  │   ├── shift_ntu60_30_r
  │   ├── shift_ntu120_10_r
  │   ├── shift_ntu120_24_r
  │   ├── shift_ntu120_40_r
  │   └── shift_ntu120_60_r
  │
  ├──label_splits
  └──class_lists
      ├── ntu60.csv       # Sparse labels
      ├── ntu60_llm.txt   # Rich kinematic descriptions
      ├── ntu120.csv
      └── ntu120_llm.txt
```
> **Note:** Pre-extracted skeleton features for the PURLS settings are not provided. Therefore, we extracted the skeleton features ourselves using the official [Shift-GCN](https://github.com/kchengiva/Shift-GCN) code.

## 🚀 Training

### Stage 1: LLM Distillation

```bash
pip install openai
export OPENAI_API_KEY="your-key"
python distill_frequency_classifier.py --dataset ntu60 --openai_api_key $OPENAI_API_KEY
```

### Stage 2: Train FDSM

Add to your config file (e.g., `config/fdsm_ntu60_unseen5.yaml`):
```yaml
use_distilled_gate: true
distilled_classifier_path: ./work_dir/distill/frequency_classifier_best.pth
```

Then train:

```bash
# Train FDSM on SynSE benchmarks for the NTU-60 dataset (55/5 split)
python main.py --config ./config/fdsm_ntu60_unseen5.yaml

# Train FDSM on SynSE benchmarks for the NTU-60 dataset (48/12 split)
python main.py --config ./config/fdsm_ntu60_unseen12.yaml

# Train FDSM on SynSE benchmarks for the NTU-120 dataset (110/10 split)
python main.py --config ./config/fdsm_ntu120_unseen10.yaml

# Train FDSM on SynSE benchmarks for the NTU-120 dataset (96/24 split)
python main.py --config ./config/fdsm_ntu120_unseen24.yaml

# Train FDSM on PURLS benchmarks for the NTU-60 dataset (40/20 split)
python main.py --config ./config/fdsm_ntu60_unseen20.yaml

# Train FDSM on PURLS benchmarks for the NTU-60 dataset (30/30 split)
python main.py --config ./config/fdsm_ntu60_unseen30.yaml

# Train FDSM on PURLS benchmarks for the NTU-120 dataset (80/40 split)
python main.py --config ./config/fdsm_ntu120_unseen40.yaml

# Train FDSM on PURLS benchmarks for the NTU-120 dataset (60/60 split)
python main.py --config ./config/fdsm_ntu120_unseen60.yaml
```


## 📜 License
The source codes can be freely used for research and education only. 

## 🤝 Acknowledgement
This repository is built upon [TDSM](https://github.com/KAIST-VICLab/TDSM) and [SkateFormer](https://github.com/KAIST-VICLab/SkateFormer). We thank the authors for their open-source contributions.
