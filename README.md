# SynShield: Privacy-Preserving Tabular Diffusion Model

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Opacus](https://img.shields.io/badge/Opacus-DP_SGD-blue)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![Security](https://img.shields.io/badge/Security-Red_Teamed-success)

SynShield is an end-to-end Machine Learning pipeline that generates synthetic tabular data while mathematically guaranteeing the privacy of the original dataset. It combines a **Tabular Denoising Diffusion Probabilistic Model (TabDDPM)** with **Differential Privacy (DP-SGD)**, and includes a built-in Red-Team evaluation suite to audit the model against **Membership Inference Attacks (MIA)**.

## 🧠 Architecture Overview

The pipeline consists of three distinct phases:

1. **The DP-Diffusion Generator:** A multi-layer perceptron (MLP) trained to reverse a Gaussian diffusion process on tabular data. The training loop is wrapped with Meta's `opacus` library to clip individual gradients and inject calibrated noise, ensuring rigorous Differential Privacy constraints.
2. **The Synthesizer:** Reverses the trained diffusion process to generate entirely new, synthetic records that maintain the statistical utility of the original data without memorizing any specific rows.
3. **The Red-Team MIA Evaluator:** A security auditing script that acts as a malicious attacker. It uses distance-based metrics (Nearest Neighbors) to attempt to identify if specific records were used in the training set.

## 🚀 Project Results

*   **Privacy Budget ($\epsilon$):** 3.64 (Calculated at $\delta = 10^{-5}$)
*   **MIA Attacker Accuracy:** ~46.60%
*   **Security Conclusion:** Because the attacker accuracy is roughly equivalent to a random guess (~50%), the evaluation mathematically proves that the Differential Privacy mechanism successfully prevented the model from memorizing the training data.

## 📂 Repository Structure

```text
privacy_diffusion_project/
├── data/
│   ├── raw/                 # Original sensitive dataset (e.g., adult.csv) - Git ignored
│   ├── processed/           # Encoded tensors, scalers, and trained models
│   └── synthetic/           # The final DP-generated dataset
├── src/
│   ├── data_processor.py    # Stateful Tabular encoding and scaling logic
│   ├── tab_ddpm.py          # PyTorch DDPM forward schedule and MLP architecture
│   ├── train_dp.py          # Training loop wrapped with Opacus PrivacyEngine
│   ├── generate.py          # Reverse diffusion sampling to create synthetic data
│   └── mia_attacker.py      # Red-Team nearest-neighbor distance attack script
├── requirements.txt
└── .gitignore               
```

## ⚙️ Installation & Usage

1. **Clone and Install:**
   ```bash
   git clone https://github.com/yourusername/SynShield.git
   cd SynShield
   pip install -r requirements.txt
   ```

2. **Run the Pipeline Sequentially:**
   
   *Step 1: Process the raw data and save stateful encoders.*
   ```bash
   python -m src.data_processor
   ```
   *Step 2: Train the Diffusion Model with Differential Privacy.*
   ```bash
   python -m src.train_dp
   ```
   *Step 3: Generate the synthetic dataset (10,000 records).*
   ```bash
   python -m src.generate
   ```
   *Step 4: Audit the model with a Membership Inference Attack.*
   ```bash
   python -m src.mia_attacker
   ```

## 🔮 Future Enhancements
*   **Utility Optimization:** Swap `MinMaxScaler` for a `QuantileTransformer` to better bound continuous features during the reverse diffusion noise addition, preventing extreme outlier generation.
*   **Shadow Modeling:** Implement an XGBoost-based shadow classifier in the MIA Evaluator to test against non-linear attack vectors.