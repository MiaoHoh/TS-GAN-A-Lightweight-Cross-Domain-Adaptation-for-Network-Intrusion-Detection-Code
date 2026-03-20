
# 🛡️ TS-GAN: A Lightweight Cross-Domain Adaptation Method for Network Intrusion Detection

## 📖 1. Description
**TS-GAN** is a high-performance framework designed to address the critical challenges in network intrusion detection systems (IDS):
* **High Computational Overhead**: Solved via **Knowledge Distillation**, compressing a complex teacher model into a lightweight BiLSTM student model.
* **Performance Degradation**: Solved via **Domain Adaptation**, using Time-Series GANs to align feature distributions between heterogeneous network environments.

This project provides a robust solution for deploying intelligent security defenses in resource-constrained **Edge Computing** environments.

---

## 📊 2. Dataset Information
The model's cross-domain capabilities are validated using two benchmark datasets:

* **Source Domain (Known):** [UNSW-NB15](https://doi.org/10.6084/m9.figshare.31742746) - Used for initial training.
* **Target Domain (Unknown):** [CIC-IDS2017](https://doi.org/10.6084/m9.figshare.31742746) - Used to test adaptation to unknown protocols-(totall_extend.csv).

**Preprocessing Workflow:**
1.  **Feature Alignment**: Mapping 42 common traffic features across domains.
2.  **Normalization**: Min-Max scaling to a [0, 1] range.
3.  **Temporal Windowing**: Segmenting flows into sequences for BiLSTM processing.



---

## 💻 3. Code Information
The repository is modularly structured for ease of maintenance:

* `models/`: Architectures for BiLSTM Student, ResNet-50 Teacher, and TS-GAN (Gen/Disc).
* `utils/`: Scripts for feature engineering, domain alignment, and GAN loss definitions.
* `train.py`: Main entry point for the dual-stage training process (Distillation & Adaptation).
* `evaluate.py`: Scripts for calculating F1-score, Latency, and generating robustness radar charts.

---

## ⚙️ 4. Requirements
This project requires **Python 3.8+**. Install the necessary dependencies via:

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
* `torch >= 1.10.0` (Deep Learning Core)
* `pandas`, `numpy`, `scikit-learn` (Data Science Stack)
* `matplotlib`, `seaborn` (Visualization)
* `tqdm` (Progress Tracking)

---

## 🚀 5. Usage Instructions

### Step 1: Data Preparation
Place your CSV datasets in the `data/` directory and run the preprocessing script:
```bash
python utils/preprocess.py --source unsw --target cicids
```

### Step 2: Model Training
To perform the lightweight distillation and cross-domain GAN adaptation:
```bash
python train.py --use_gan True --batch_size 64 --epochs 70
```

### Step 3: Performance Evaluation
To verify the **1.42ms** inference latency and **90.13%** F1-score on the target domain:
```bash
python evaluate.py --checkpoint ./weights/ts_gan_final.pth
```



---

## 📈 Summary of Results
| Metric | TS-GAN (Proposed) | Baseline (BiLSTM) | Improvement |
| :--- | :---: | :---: | :---: |
| **Inference Latency** | **1.42 ms** | 5.24 ms | **72.9% Faster** |
| **F1-Score (Target)** | **90.13%** | 69.27% | **+20.86%** |
| **Parameters** | **4.80 M** | 5.24 M | **8.4% Smaller** |
```


