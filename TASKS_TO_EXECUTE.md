# 1. Overview
This file serves as an actionable execution checklist derived from the original implementation plan. It breaks down the comparative research project on sentiment analysis into practical, trackable tasks. 
- **Source File Used**: `implementation_plan.md`
- **Mandatory Directives Found**: Yes. Several `IMPORTANT` and `WARNING` mentor notes were identified regarding freezing project components early, emphasizing feature importance visualizations, honest reporting of NN performance, and dedicating maximum analytical effort to domain shift evaluation.

# 2. Mandatory Requirements from IMPORTANT Sections
These requirements must be strictly adhered to across all phases of execution:
- [ ] **Freeze early components**: Tokenization strategy per model, fixed random seeds for reproducibility, and unified preprocessing rules must be finalized before any model training begins. Downstream comparisons become meaningless otherwise. (IMPORTANT)
- [ ] **Include feature importance visualizations**: Do not skip feature importance visualizations for classical ML models. They are crucial for the interpretability argument of the broader study. (IMPORTANT)
- [ ] **Report NN performance honestly**: If Neural Network models do not clearly beat the TF-IDF baseline, state this honestly. This is a valid and insightful research outcome. (TIP recognized as Mandatory Constraint)
- [ ] **Prioritize Domain Shift Analysis**: Dedicate maximum analytical effort to the Domain Shift Evaluation (Phase 7). Do not rush this, as it is the core novelty of the project. (WARNING)

# 3. Tasks by Phase

## Phase 0: Project Framing & Scope
- **Objective:** Establish the project foundation to avoid scope creep and confusion later.
- **Actionable Tasks:**
  - [ ] Define binary sentiment classification task (positive / negative).
  - [ ] Define what constitutes in-domain vs. out-of-domain data.
  - [ ] Finalize the exact list of models to evaluate.
  - [ ] Establish and freeze tokenization strategy per model.
  - [ ] Set fixed random seeds for reproducibility.
  - [ ] Define unified preprocessing rules wherever applicable.
- **Dependencies:** None.
- **Expected Outputs:** Documented definitions, frozen model list, reproducible seed setting script.
- **Definition of Done:** All project framing decisions are documented and frozen, with no further additions allowed.

## Phase 1: Dataset Pipeline & Preprocessing
- **Objective:** Acquire datasets and establish reusable preprocessing pipelines.
- **Actionable Tasks:**
  - [ ] **1.1 Dataset Acquisition:**
    - [ ] Load Dataset A (In-Domain: Amazon Reviews - Train, Val, Test split).
    - [ ] Load Dataset B (Out-of-Domain: IMDb Reviews - Test split only).
    - [ ] Generate dataset statistics table (size, label balance).
  - [ ] **1.2 Text Preprocessing:**
    - [ ] Implement shared processing baseline (lowercasing, punctuation handling, stopword handling with justification, max sequence length for NNs).
    - [ ] Implement TF-IDF/n-grams vectorization pipeline.
    - [ ] Implement GloVe embedding lookup maps.
    - [ ] Implement DistilBERT tokenizer application.
- **Dependencies:** Phase 0.
- **Expected Outputs:** Dataset loading scripts, preprocessing codebase, test splits, dataset stats table, ablation note on preprocessing impacts.
- **Definition of Done:** Both datasets are loaded, preprocessed according to model-specific needs, and statistics/ablation notes are documented.

## Phase 2: Statistical Models
- **Objective:** Train and interpret statistical machine learning models, including baseline lexical features and static embeddings.
- **Actionable Tasks:**
  - [ ] Train TF-IDF + Logistic Regression model.
  - [ ] Train n-grams + SVM model.
  - [ ] Load pre-trained GloVe vectors (or fallback dummy vectors).
  - [ ] Implement sentence embedding aggregation strategy (mean/pooled).
  - [ ] Train GloVe embeddings + Logistic Regression / SVM model.
  - [ ] Perform minimal, justified hyperparameter tuning.
  - [ ] Extract and visualize feature importance for classical models.
- **Dependencies:** Phase 1.
- **Expected Outputs:** Performance table (Accuracy, Macro-F1), training/inference speed metrics, feature importance visualizations.
- **Definition of Done:** All three statistical models are trained, evaluated, and their deliverables are generated.

## Phase 3: Neural Models
- **Objective:** Train deep learning models with embeddings (GloVe) and contextual models (DistilBERT).
- **Actionable Tasks:**
  - [ ] Define architectures for CNN Text Classifier and BiLSTM.
  - [ ] Integrate GloVe embeddings as input layers for CNN and BiLSTM.
  - [ ] Implement DistilBERT with model-specific tokenization as a fine-tuned / neural-head model.
  - [ ] Implement training loops with early stopping callbacks, dropout, and validation-based model selection.
  - [ ] Generate learning curves (loss vs. epochs) and test data confusion matrices.
  - [ ] Analyze performance gains relative to statistical baselines or compute cost overhead.
- **Dependencies:** Phase 1, Phase 2 (if reusing embeddings).
- **Expected Outputs:** Best model checkpoints, Learning curves, confusion matrices, performance discussion.
- **Definition of Done:** All neural models (CNN, BiLSTM, DistilBERT) are trained natively as deep learning models, evaluated, and honest performance conclusions are made against statistical baselines.

## Phase 6: Experimental Design & Robustness Testing
- **Objective:** Test model sample efficiency and resilience to noise.
- **Actionable Tasks:**
  - [ ] **6.1 Data Regimes:**
    - [ ] Train each selected model on scaled subsets: 1%, 5%, 10%, 100% of training data.
  - [ ] **6.2 Robustness Tests:**
    - [ ] Evaluate models under word order shuffling perturbation.
    - [ ] Evaluate models under noise injection perturbation (typos/swaps).
- **Dependencies:** Phases 2, 3, 4, 5 (Models must be implemented and selected).
- **Expected Outputs:** Performance vs. Data Size graphs, sample efficiency analysis, degradation plots, ranked list of models by robustness.
- **Definition of Done:** All selected models undergo data scaling and robustness testing, with visual plots and rankings produced.

## Phase 7: Domain Shift Evaluation (Critical Section)
- **Objective:** Evaluate the generalization capability of each model to an unseen domain.
- **Actionable Tasks:**
  - [ ] Evaluate all trained models on IMDb Reviews (Test Environment).
  - [ ] Track Absolute Accuracy drop and Absolute Macro-F1 drop.
  - [ ] Perform deep analysis detailing which models generalize best and why.
- **Dependencies:** Phase 1 (for IMDb split), Phases 2, 3.
- **Expected Outputs:** Domain shift comparison table, deep analytical write-up.
- **Definition of Done:** Evaluation is fully complete, maximum analytical effort reflects in the deep analysis of generalization.

## Phase 8: Error Analysis & Interpretability
- **Objective:** Qualitatively understand model failures and feature importance.
- **Actionable Tasks:**
  - [ ] Establish an error taxonomy.
  - [ ] Identify and document representative failure cases.
  - [ ] Extract and map feature importance markers.
  - [ ] Render interpretability heatmaps.
- **Dependencies:** Phases 2, 3, 4, 5, 7.
- **Expected Outputs:** Error analysis document segment, model-specific behavioral insights.
- **Definition of Done:** Cross-model qualitative analysis and visual interpretability markers are fully documented.

## Phase 9: Final Comparison & Reporting
- **Objective:** Synthesize all findings into a final project report.
- **Actionable Tasks:**
  - [ ] Cross-evaluate models on: Performance Metrics, Interpretability, Robustness, Compute Efficiency, Data Sensitivity.
  - [ ] Write clear adoption/recommendation section based on usage context.
  - [ ] Document project scope limitations and future work paths.
- **Dependencies:** Phases 6, 7, 8.
- **Expected Outputs:** Culminating comparison table, recommendation section, limitations documentation.
- **Definition of Done:** Final synthesis and documentation are fully complete.

# 4. Prioritized Execution Checklist
- [CRITICAL] Phase 0: Freeze all key framing and scope decisions.
- [CRITICAL] Phase 1: Implement dataset pipeline and preprocessing.
- [CRITICAL] Phase 7: Evaluate models for Domain Shift (core novelty).
- [CRITICAL] Phase 9: Final synthesis and final comparison table.
- [HIGH] Phase 2: Train Statistical Models and create feature importance visualisations.
- [HIGH] Phase 3: Train Neural Models (CNN, BiLSTM, DistilBERT).
- [HIGH] Phase 6: Perform Experimental Design & Robustness Testing.

# 5. Dependency Order
1. **Core Base (Must happen first):** Phase 0 -> Phase 1
2. **Model Training (Parallelizable):** 
   - Phase 1 -> Phase 2 (Statistical Models)
   - Phase 1 & 2 -> Phase 3 (Neural Models)
3. **Evaluation & Stress Testing (Needs Models):**
   - Completed Models -> Phase 6 (Robustness / Data Regimes)
   - Completed Models -> Phase 7 (Domain Shift)
4. **Synthesis (Needs Evaluation):**
   - Phase 6 & 7 -> Phase 8 (Error Analysis)
   - Phase 8 -> Phase 9 (Final Reporting)

# 6. Deliverables Checklist
- [ ] Documented Task Definitions & Domain Definitions.
- [ ] Dataset Loading Scripts & Preprocessing Codebase.
- [ ] Dataset Statistics Table & Preprocessing Ablation Note.
- [ ] Classical ML Performance Table & Feature Importance Visualizations.
- [ ] Static Embeddings Performance Comparison & Error Examples.
- [ ] Neural Network Learning Curves, Confusion Matrices, Heatmaps.
- [ ] Transformer Best Checkpoint & Compute-Cost Discussion.
- [ ] Performance vs. Data Size Graphs & Sample Efficiency Analysis.
- [ ] Robustness Degradation Plots & Model Ranking List.
- [ ] Domain Shift Comparison Table & Deep Generalization Analysis.
- [ ] Error Taxonomy, Representative Failure Cases, Interpretability Heatmaps.
- [ ] Culminating Final Comparison Table.
- [ ] Contextual Adoption Recommendations.
- [ ] Scope Limitations & Future Work Document.


