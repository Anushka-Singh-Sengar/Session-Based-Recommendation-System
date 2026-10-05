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
│       ├── __init__.py
│       └── preprocessing.py
├── scripts/
│   ├── inspect_data.py
│   └── preprocess_data.py
├── tests/
│   └── test_preprocess_synthetic.py
├── results/
│   ├── inspection/
│   └── preprocessing/
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

---

## ⚙️ Data Preprocessing

The preprocessing pipeline converts the raw clickstream into clean, chronological session splits with right-padded inputs, target tokens, and inter-click time intervals ($\Delta t$).

### 1. Finalized Default Parameters

| Parameter | CLI Argument | Default Value | Description |
| :--- | :--- | :--- | :--- |
| **Raw Data Path** | `--raw-path` | `data/raw/yoochoose_clicks.dat` | Path to raw clicks file |
| **Output Root** | `--output-root` | `data/processed` | Destination for `.npz` arrays |
| **Results Root** | `--results-root` | `results/preprocessing` | Destination for logs and reports |
| **Chunksize** | `--chunksize` | `2000000` | Rows per streaming chunk |
| **Subset Fraction** | `--subset-fraction` | `1/64` | Most recent session fraction |
| **Min Session Length**| `--min-session-length` | `2` | Minimum clicks per session |
| **Min Item Support** | `--min-item-support` | `5` | Train-only minimum item frequency |
| **Max Sequence Length**| `--max-len` | `20` | Truncation history window |
| **Split Fractions** | `--val-fraction`, `--test-fraction` | `0.1`, `0.1` | Chronological split ratios (80/10/10) |
| **Collapse Repeats** | `--collapse-consecutive-repeats` | `True` | Collapse adjacent clicks $A \to A$ |
| **Save Time Deltas** | `--save-time-deltas` | `True` | Generate `input_delta_t` in seconds |

### 2. Pipeline Summary (P1 – P13)

1. **P1 (Load):** Chunked single-pass read with `usecols=[0,1,2]`, ISO timestamp parsing to epoch ms, and int32 validation.
2. **P2 (Global Sort):** Stable global sort by `(session_id, timestamp_ms, row_indices)`.
3. **P3 (Raw Session Table):** Extraction of raw session click counts and `raw_end_ms` (max timestamp).
4. **P4 (Exact Duplicates):** Deduplication on `(session_id, timestamp_ms, item_id)` keeping lowest row index.
5. **P5 (Collapse Repeats):** Collapse consecutive repetitions ($A, A, A, B \to A, B$).
6. **P6 (Eligibility):** Filter sessions with cleaned length $\ge \text{min\_session\_length}$.
7. **P7 (Subset):** Order eligible sessions by `(raw_end_ms, session_id)` and retain the most recent fraction.
8. **P8 (Chronological Split):** Split retained sessions chronologically into train, validation, and test.
9. **P9 (Vocabulary Pruning):** Iterative rare item filtering ($< \text{min\_item\_support}$) on train split only.
10. **P10 (Val/Test Cleaning):** Remove OOV clicks from eval splits and re-verify minimum length.
11. **P11 (Encoding):** Sort vocabulary ascending by raw item ID, map to $1 \dots N$ (PAD=0), build index tables.
12. **P12 (Vectorized Examples):** Generate right-padded input windows, target tokens, and delta-t arrays.
13. **P13 (Save & Validate):** Atomic write to `.partial/`, run validation suite V1–V14, compute SHA-256 fingerprints, write metadata/reports, rename to final folder, and write `_SUCCESS`.

### 3. Processed Run Folder Layout

```
data/processed/<run_name>/
├── train.npz             # inputs (int32 [N, 20]), lengths (int32 [N]), targets (int32 [N]),
│                         # session_index (int32 [N]), position (int32 [N]), input_delta_t (float32 [N, 20])
├── val.npz               # Validation examples
├── test.npz              # Test examples
├── sessions_train.npz    # items, offsets, timestamps_ms, session_ids, end_time_ms
├── sessions_val.npz
├── sessions_test.npz
├── item_index.csv        # item_index, raw_item_id, train_click_count
├── item2idx.json         # Raw ID to encoded index map
├── metadata.json         # Complete provenance, drop ledger, and fingerprints
├── preprocessing_report.md
└── _SUCCESS
```

### 4. Running Preprocessing

#### Local Synthetic Unit Tests:
```bash
python tests/test_preprocess_synthetic.py
```

#### Local Smoke Test (First 200k Rows):
```bash
python scripts/preprocess_data.py --debug-max-rows 200000
```

#### Full Run (in Google Colab):
```bash
# 1/64 development subset (default)
python scripts/preprocess_data.py --subset-fraction 1/64

# 1/8 benchmark subset
python scripts/preprocess_data.py --subset-fraction 1/8

# 1/4 large subset
python scripts/preprocess_data.py --subset-fraction 1/4
```

#### Verifying an Existing Run:
```bash
python scripts/preprocess_data.py --verify-only frac_1-64_minlen2_minsup5_maxlen20_collapse_split80-10-10
```

---

## 🤖 Modeling Pipeline (GRU Baseline, In Progress)

The modeling stage implements a shared session recommender framework with pluggable sequence encoders (`GRUEncoder`, and in a later stage `LTCEncoder`).

### 1. Existing Components
- **Data Loader (`src/utils/data.py`)**: `load_run` with `_SUCCESS` validation and fingerprint checking, combined OOV & time-delta statistics loading, dynamic sequence trimming, and in-memory batch iterator.
- **Metrics (`src/utils/metrics.py`)**: Full-ranking `Recall@K` and `MRR@K` ($K \in \{5, 10\}$) over all item indices $1 \dots N$ with pessimistic tie-breaking and PAD token exclusion.
- **Model Wrapper & Encoders (`src/models/`)**: `SessionRecommender` wrapper with $\mathcal{N}(0, 0.1)$ embedding, PAD logit masking, linear prediction head, and pluggable `SequenceEncoder` interface (`GRUEncoder`).
- **Training Script (`scripts/train.py`)**: Full training loop with validation early stopping driven by `val MRR@10`, gradient norm clipping, and standard JSON/CSV output logging.
- **Test Suite (`tests/`)**:
  - `python tests/test_metrics.py` (Unit tests for metrics)
  - `python tests/test_model_pipeline.py` (Integration & invariant tests T1-T6)

### 2. Experimental-Design Invariants
- **E1 (Item-Only Isolation)**: In item-only mode, time deltas ($\Delta t$) are set to `None` and never passed to the sequence encoder.
- **E2 (Time-Aware Control)**: In `--use-time-deltas` mode, normalized input time gaps ($\log(1 + \Delta t) / \sigma_{\text{train}}$) are concatenated to item embeddings.
- **E3 (Shared Infrastructure)**: Data loaders, loss function (full softmax cross-entropy), embedding initialization, head, optimizer (Adam), learning rate, and early-stopping rules are identical across architectures.
- **E4 (No Architectural Bias)**: Models are evaluated fairly without pre-supposing performance differences.
- **E5 (Validation-Driven Stopping)**: Early stopping is strictly driven by validation `MRR@10`. The test split remains untouched during training and development.

### 3. Running Synthetic Model Tests
```bash
python tests/test_metrics.py
python tests/test_model_pipeline.py
```

