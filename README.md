# Session-Based Recommendation System using Liquid Neural Networks (LTC)

A research and development project for session-based next-item recommendation that evaluates and compares a standard **Gated Recurrent Unit (GRU)** baseline against **Liquid Time-Constant (LTC)** Neural Networks.

---

## 📌 Project Overview

Traditional session-based recommendation models (e.g., GRU4Rec) process sequential user interactions using discrete time steps. Liquid Time-Constant (LTC) networks and continuous-time neural models adapt dynamically to varying time intervals between user clicks and session dynamics.

This project investigates:
- Baseline sequential session recommendation using **GRU**.
- Dynamic session modeling using **Liquid Time-Constant (LTC)** networks.
- Comparative evaluation on ranking metrics including **Hit Rate (HR@K)** and **Mean Reciprocal Rank (MRR@K)**.

---

## 📁 Project Structure

```
Session-Based-Recommendation-System/
├── data/
│   ├── raw/
│   │   ├── .gitkeep
│   │   └── yoochoose_clicks.dat
│   └── processed/
│       └── .gitkeep
├── src/
│   ├── models/
│   └── utils/
├── scripts/
├── results/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

- **Dataset**: Yoochoose Clickstream Dataset (RecSys Challenge 2015)
- **Required Raw Dataset Path**: `data/raw/yoochoose_clicks.dat`

> **Note on Dataset Download:**  
> The raw dataset is large (~1.48 GB) and is **not** tracked or committed to Git / GitHub (ignored via `.gitignore`).  
> Download the Yoochoose clickstream dataset separately and place the extracted clicks file at `data/raw/yoochoose_clicks.dat`.
