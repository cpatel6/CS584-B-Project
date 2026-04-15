# Implementation Plan: Comparative Analysis of Statistical and Neural Text Learning Techniques for Sentiment Analysis under Domain Shift

## Phase 0: Project Framing & Scope
**Goal:** Avoid scope creep and confusion later.

**Deliverables:**
*   **Clear task definition:** Binary sentiment classification (positive / negative).
*   **Domain mapping:** Clear in‑domain vs out‑of‑domain definition.
*   **Model roster:** Finalized list of models to evaluate (no last‑minute additions).

**Key Decisions (to be frozen early):**
*   Tokenization strategy per model.
*   Fixed random seeds for reproducibility.
*   Unified preprocessing rules wherever applicable to ensure fair comparison.

> [!IMPORTANT]
> **Mentor Note:** If these components are not frozen early, downstream comparisons become meaningless.

---

## Phase 1: Dataset Pipeline & Preprocessing

### 1.1 Dataset Acquisition
**Datasets (via HuggingFace):**
*   **Dataset A (In‑Domain): Amazon Reviews**
    *   Train Split
    *   Validation Split
    *   In‑domain Test Split
*   **Dataset B (Out‑of‑Domain): IMDb Reviews**
    *   Test Split only (for domain shift evaluation)

**Deliverables:**
*   Dataset loading scripts.
*   Dataset statistics table (size, label balance).

### 1.2 Text Preprocessing
**Shared Processing (Baseline):**
*   Lowercasing.
*   Punctuation handling.
*   Stopword handling (with documented justification).
*   Max sequence length definition (for NN models).

**Model‑Specific Preprocessing:**
*   *TF‑IDF / n‑grams:* Vectorization.
*   *GloVe:* Embedding lookup maps.
*   *DistilBERT:* Tokenizer application only (no heavy manual cleaning).

**Deliverables:**
*   Reusable, modular preprocessing code.
*   Ablation note detailing which preprocessing steps helped or degraded specific models.

---

## Phase 2: Statistical Models

### 2.1 Models to Implement
*   **TF‑IDF + Logistic Regression**
*   **n-grams + SVM**
*   **GloVe embeddings + Logistic Regression / SVM (Averaged/Pooled)**

### Implementation Steps:
1.  Perform feature vectorization.
2.  Train models.
3.  Execute minimal, justified hyperparameter tuning.
4.  Conduct validation‑set selection.

**Deliverables:**
*   Performance table (Accuracy, Macro‑F1).
*   Metrics on training time & inference speed.
*   Feature importance visualization (e.g., coefficient analysis).

> [!IMPORTANT]
> **Mentor Note:** Skipping feature importance visualisations here weakens the interpretability argument of the broader study.

---

## Phase 3: Neural Models

### 3.1 Models to Implement
*   **CNNText + GloVe embeddings**
*   **BiLSTM + GloVe embeddings**
*   **DistilBERT (Fine-tuned / Frozen encoder + neural head)**

### Implementation Steps:
1.  Define model architectures (CNN, BiLSTM).
2.  Integrate GloVe as input embeddings for CNNText and BiLSTM.
3.  Adopt DistilBERT specific tokenization.
4.  Conduct fine-tuning experiments comparing frozen vs. unfrozen layer configurations.
5.  Train with early stopping callbacks and regularizations.
6.  Perform validation‑based model selection.

**Deliverables:**
*   Learning curves (loss vs. epochs).
*   Confusion matrices for test data.
*   Visual heatmaps (attention/saliency representations if applicable).
*   Best model checkpoint saved for DistilBERT.
*   Discussion analyzing performance gains relative to compute cost overhead.

> [!TIP]
> **Mentor Note:** If the NN models do not clearly beat the statistical baselines, state this honestly. It is a valid and insightful research outcome.

---

## Phase 6: Experimental Design & Robustness Testing

### 6.1 Data Regimes
Train each selected model utilizing scaled data subsets:
*   1% of training data
*   5% of training data
*   10% of training data
*   100% of training data

**Deliverables:**
*   Graphical representations of Performance vs. Data Size graphs.
*   Comprehensive sample efficiency analysis.

### 6.2 Robustness Tests
Evaluate models under pertubed conditions:
*   Word order shuffling.
*   Noise injection (simulating typos / random token swaps).
*   Domain shift.

**Deliverables:**
*   Degradation plots highlighting robustness falls.
*   Ranked list of models organized by their robustness to noise.

---

## Phase 7: Domain Shift Evaluation (Critical Section)

### 7.1 Setup & Strategy
*   **Train Environment:** Amazon Reviews.
*   **Test Environment:** IMDb Reviews.

**Metrics Tracked:**
*   Absolute Accuracy drop.
*   Absolute Macro‑F1 drop.

**Deliverables:**
*   Comprehensive domain shift comparison table.
*   Deep analysis elucidating which models generalize most effectively and postulating why.

> [!WARNING]
> **Mentor Note:** This is the core novelty of your project. Dedicate maximum analytical effort here—do not rush it.

---

## Phase 8: Error Analysis & Interpretability

### 8.1 Qualitative Analysis
*   Establish an error taxonomy.
*   Identify and document representative failure cases.
*   Extract feature importance markers and map them.
*   Render interpretability heatmaps.

**Deliverables:**
*   Detailed error analysis document segment.
*   Model‑specific behavioral insights text.

---

## Phase 9: Final Comparison & Reporting

### 9.1 Final Synthesis
Synthesize and cross-evaluate all models on the following pillars:
1.  Performance Target Metrics
2.  Inherent Interpretability
3.  Robustness Under Perturbation
4.  Compute & Inference Efficiency
5.  Sensitivity to Data Availability

**Deliverables:**
*   Culminating comparison table.
*   Clear adoption/recommendation section based on context.
*   Documentation of scope limitations and paths for future work.
