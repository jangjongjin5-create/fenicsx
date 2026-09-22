#  ALD Physics-Guided Digital Twin

> **Physics-guided machine learning framework for atomic layer deposition (ALD) modeling using FEniCSx.**

---

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FEniCSx](https://img.shields.io/badge/FEniCSx-0.11-00599C?style=flat-square)](https://fenicsproject.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

---

##  Overview

This project builds a **Physics-Guided Digital Twin for Atomic Layer Deposition (ALD)**. By coupling known transport/kinetics PDEs with a data-driven surface state learner, the model accurately predicts precursor behavior, surface saturation, and film uniformity while optimizing pulse and purge timing.

---

## Key Objectives

- **Physics-Informed Modeling:** Integrate advection-diffusion-reaction PDEs with neural operators.
- **Missing Physics Learning:** Recover unobservable hidden byproduct coverage ($\hat{\Theta}_b$) using Physics-Guided Machine Learning (FEML).
- **Process Optimization:** Reduce precursor waste and cycle time while maintaining film conformity.
- **Dimensional Transfer:** Scale predictive capability from 2D reactor domains to complex 3D structures.

---

##  System Architecture

┌──────────────────────────┐
                           │  Advection-Diffusion PDE │
                           └────────────┬─────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
   ┌───────────────────────────┐                 ┌───────────────────────────┐
   │   Known Langmuir Kinetics │                 │  Learned Surface Dynamics │
   │       (Precursor Phase)   │                 │     (Hidden State θ_b)    │
   └─────────────┬─────────────┘                 └─────────────┬─────────────┘
                 │                                             │
                 └──────────────────────┬──────────────────────┘
                                        ▼
                           ┌──────────────────────────┐
                           │   Total Reaction Rate    │
                           │        R_predict         │
                           └──────────────────────────┘


## 🧮 Mathematical Formulation

### 1. Ground Truth (Physical Reality)
$$R_{\text{GT}} = k_p c_p (1 - \Theta_p - \Theta_b)$$

### 2. Learner Formulation (FEML)
$$\frac{d\hat{\Theta}_b}{dt} = f_\phi\left(c_p, c_b, \Theta_p, \hat{\Theta}_b\right)$$

* **Known Physics:** Transport PDE + Langmuir adsorption structure
* **Uncertain/Missing Dynamics:** Hidden surface coverage $\hat{\Theta}_b$ parametrized via Neural Network $f_\phi$

---

## 📁 Repository Structure

```text
ald_optimization/
├── 📂 baseline/                 # Verification & Analytical Models
│   ├── run_01_analytic.py       # 1D Analytical Verification
│   ├── run_02_ideal_2d.py       # Ideal 2D Surface Saturation
│   ├── run_03_validation.py     # Grid Convergence & Validation
│   ├── run_04_flow.py           # Fluid Flow Field Integration
│   ├── run_05_nonideal.py       # Non-ideal Surface Kinetics
│   ├── run_06_virtual_sensors.py# Sensor Placement & Data Extraction
│   └── run_07_3d.py             # 3D Domain Scaling
│
├── 📂 feml/                     # Physics-Guided Machine Learning
│   ├── gt_competitive_standalone.py        # Ground Truth Generator
│   └── train_missing_physics_standalone.py # Explicit Hidden-State Trainer
│
├── 📂 figures/                  # Simulation & Parity Plots
├── 📂 docs/                     # Methodology & Theory Notes
├── 📄 environment.yml           # Conda Environment Setup
└── 📄 README.md
🚀 Quick Start
1. Environment Setup
Bash
conda env create -f environment.yml
conda activate ald-feml
2. Run Baseline Simulation
Bash
python baseline/run_02_ideal_2d.py
3. Train Missing Physics Model
Bash
python feml/train_missing_physics_standalone.py
🛠 Tech Stack
Category	Tools & Libraries
PDE Solver	FEniCSx 0.11, PETSc
Deep Learning	PyTorch
Scientific Computing	NumPy, SciPy
Visualization	Matplotlib, PyVista
📜 License
Distributed under the MIT License. See LICENSE for details.


---

### ✨ 반영된 꾸미기 포인트
1. **상단 배지(Badges):** Python, PyTorch, FEniCSx 버전을 깔끔하게 시각화.
2. **ASCII 아키텍처 다이어그램:** 코드 구성 및 물리-AI 결합 구조를 한눈에 볼 수 있도록 배치.
3. **LaTeX 수식:** $R_{\text{GT}}$ 및 $\hat{\Theta}_b$ 수식을 깔끔하게 정리.
4. **트리 구조(Directory Tree):** 폴더와 파일 구조에 이모지를 곁들여 깔끔하게 정리.
5. **테이블 스택 정돈:** 기술 스택을 표(Table)로 정리하여 가독성 확보.



