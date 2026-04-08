
# Explainable & Calibrated ML for Probability of Default under IFRS 9

[![SSRN](https://img.shields.io/badge/SSRN-6443498-red)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6443498)

## 📌 Overview

This repository contains the complete code and methodology for the paper:

> **"Explainable and Calibrated Machine Learning Models for Probability of Default: An application to Expected Credit Loss under IFRS 9"**  
> *Mohammed Nazrul Islam* – SSRN Abstract ID: 6443498

The project develops a **regulatory‑compliant machine learning pipeline** to estimate Probability of Default (PD) under IFRS 9, with a focus on emerging markets (Bangladesh). It addresses key challenges: class imbalance, probability calibration, model explainability, and governance.

## 🔬 Key Results

| Metric | Value |
|--------|-------|
| **AUC‑ROC** (XGBoost) | 0.949 |
| **Gini Coefficient** | 0.897 |
| **Brier Score** (Isotonic) | 0.0535 |
| **Accuracy** | 93.0% |
| **Precision** (Default) | 0.96 |
| **Recall** (Default) | 0.73 |

The **Isotonic Regression** calibration method outperforms Platt Scaling (Brier 0.0562) and is selected as the champion for Point‑in‑Time PD estimation.

## 🧠 Methodology

1. **Data** – Public consumer loan dataset (32,581 obs) as a high‑fidelity proxy for retail portfolios.
2. **Class Balancing** – SMOTE (Synthetic Minority Over‑sampling).
3. **Model Training** – 12 algorithms evaluated; **XGBoost** selected as champion.
4. **Calibration** – Platt Scaling vs. Isotonic Regression (Isotonic wins).
5. **Explainability** – SHAP (global) and LIME (local) for audit‑ready reason codes.
6. **Validation & Stability** – PSI (Population Stability Index) and three‑lines‑of‑defence governance.

## 🛠️ Repository Structure

