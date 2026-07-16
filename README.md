<div align="center">

# 🔥 Wildfire Evacuation Threat Forecasting

### Predicting wildfire threat to evacuation zones — 12h to 72h ahead — from just the first 5 hours of data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Competition](https://img.shields.io/badge/Kaggle-WiDS%20Datathon%202026-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/)
[![Team](https://img.shields.io/badge/Team-KLE%20VibeCoders-orange)](https://sites.google.com/kletech.ac.in/klevibecoders/about-project)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()

📘 **[Full project writeup — EDA, feature engineering, model building, key findings →](https://sites.google.com/kletech.ac.in/klevibecoders/about-project)**

</div>

---

## 🌍 The Problem

Wildfires don't wait for full information. First responders and evacuation planners need to know — often within the first few hours of a fire's start — how likely it is to threaten a populated zone over the next **12, 24, 48, and 72 hours**.

The **WiDS Datathon 2026** challenge asked us to build exactly that: a model that outputs threat probabilities at four future horizons, using **only the first 5 hours** of a wildfire's observed behavior.

This isn't a standard classification task. It's **survival probability estimation under censored conditions** — some fires in the training set never threaten a zone within the observed window, some do early, some do late, and the model has to reason about *when*, not just *if*.

## 🧠 Approach

- **Feature engineering** on early fire-spread dynamics, weather signals, and terrain/proximity features from the first 5-hour window
- **Gradient Boosted Survival Analysis (GBSA)** to directly model time-to-threat
- **LightGBM** classifiers per horizon as a comparison/ensemble partner
- **Isotonic calibration** to correct probability skew from the boosted models
- **Ensemble blending** across model variants and horizons for the final submission

## 📊 Iteration Summary

Predicted-probability profile across pipeline iterations, leading to the final submission:

| Iteration | Mean prob_12h | Mean prob_24h | Mean prob_48h | Std (prob_12h) |
|---|---|---|---|---|
| v1 | 0.210 | 0.290 | 0.306 | 0.319 |
| v2 | 0.210 | 0.289 | 0.306 | 0.321 |
| v3 | 0.006 | 0.055 | 0.174 | 0.009 |
| v5 | 0.289 | 0.387 | 0.403 | 0.279 |
| **v6 (final)** | **0.190** | **0.276** | **0.301** | **0.245** |

**v6 was selected as the final submission** — it holds the tightest, most stable spread across predictions (lowest std deviation) while still preserving clear separation between low-risk and high-risk events, indicating well-calibrated rather than over/under-confident outputs.

## 🏆 Leaderboard Performance

| Submission | Private Score | Public Score |
|---|---|---|
| **vsub6.csv (final)** | **0.96923** | **0.96840** |

## 📁 Repo Structure

```
notebooks/     EDA and model development notebooks
src/           Feature engineering, training, calibration, ensembling scripts
submissions/
  └── final_submission.csv   ← v6, the submitted model
docs/
  └── model_comparison.md    ← extended notes on all variants
```

## 🚀 Running Locally

```bash
git clone https://github.com/vaishnavi-212/wids-datathon-2026-wildfire-survival.git
cd wids-datathon-2026-wildfire-survival
pip install -r requirements.txt
```

## 👥 Team

**KLE VibeCoders** — KLE Technological University, Hubballi

## 📄 License

Released under the [MIT License](LICENSE).

## 🔗 Links

- [Project site with full EDA & findings](https://sites.google.com/kletech.ac.in/klevibecoders/about-project)
- [WiDS Datathon](https://www.widsconference.org/datathon.html)
