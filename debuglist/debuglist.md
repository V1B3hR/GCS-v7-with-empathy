# GCS-v7 Debug Charter & End-to-End Debugging Process

## 0. Document Purpose
This markdown file formalizes the structured debugging and improvement workflow for the GCS-v7-with-empathy project.  
Use it to:  
- Align on goals, metrics, scope, and responsibilities  
- Track hypotheses, experiments, and outcomes  
- Prevent regressions via disciplined iteration  
- Provide an auditable record of decisions

---

## 1. Project Overview
- Project Name: GCS-v7-with-empathy
- Repository: `V1B3hR/GCS-v7-with-empathy`
- Current Model / Component Under Debug: Multi-modal Affective State Classifier (empathy-enhanced emotion recognition)
- Stakeholders / Owners:  
  - Tech Lead: @V1B3hR  
  - ML Engineer(s): GCS Development Team  
  - Reviewer(s): Community contributors  
  - Domain Expert(s): BCI and affective computing researchers  
- External Dependencies / APIs: TensorFlow/Keras, Spektral (Graph Neural Networks), DEAP Dataset
- Target Deployment Environment: Real-time WebSocket API with FastAPI backend, cloud-based (Codespaces/Render)

---

## 2. Problem Statement
Describe what is “wrong” or insufficient with the current model.
- Symptoms: Multi-modal affective state classifier requires validation on real EEG data; model fusion architecture needs performance benchmarking; LOSO cross-validation results pending documentation
- When first observed: 2025-09-30 (Initial deployment phase)
- Impact Severity (High | Medium | Low): Medium - System functional but requires empirical validation
- Affected Use Cases / Users: Real-time emotion detection for therapeutic BCI applications; empathetic response generation accuracy depends on classifier performance  

---

## 3. Objectives & Success Criteria
| Metric | Type (Primary / Guardrail) | Current | Target | Rationale | Acceptance Gate |
|--------|----------------------------|---------|--------|-----------|-----------------|
| F1 (macro) | Primary | 0.61 | ≥0.68 | Improve minority recall | Higher than baseline & stable |
| AUROC | Guardrail | 0.83 | ≥0.85 | Ranking quality | No overfit to few classes |
| Calibration ECE | Guardrail | 0.09 | ≤0.05 | Trustworthy probabilities | After temp scaling |
| Latency P95 (ms) | Guardrail | 180 | ≤200 | Real-time constraint | No >5% regression |
| Fairness gap (ΔF1 subgroup max-min) | Primary fairness | 0.24 | ≤0.12 | Reduce disparity | Must pass bias audit |
| Drift PSI (top features) | Monitoring | 0.3 | <0.2 | Stability | Early warning |

(Adjust metrics & thresholds.)

---

## 4. Data Summary
| Dataset Split | Size (rows) | Date Range | Class Dist Summary | Notes |
|---------------|-------------|------------|--------------------|-------|
| Train | Variable (LOSO) | N/A | DEAP dataset baseline | EEG channels: 64, Timesteps: 250 |
| Validation | Variable (LOSO) | N/A | Held-out subject per fold | Leave-One-Subject-Out CV |
| Test (Holdout, Untouched) | Pending | N/A | To be collected | Real-world therapeutic scenarios |
| Shadow / Live Stream Sample | Real-time | Current | User sessions | WebSocket streaming data |

- Feature Catalog: Multi-modal features including EEG (64 channels × 250 timesteps), Physiological signals (2 features: HRV, GSR), Voice prosody (128 features)
- Sensitive / Proxy Features: All emotional data encrypted; privacy-preserving protocols active; 90-day retention policy
- Known Data Risks: Label quality dependent on DEAP annotations; potential distribution shift between lab and therapeutic settings; class imbalance in emotional states

---

## 5. Hypothesis Backlog
Track potential root causes before changing code. Each entry must have test strategy & priority.

| ID | Hypothesis | Type (Data / Model / Infra / Eval / Fairness) | Evidence | Test / Experiment | Priority (H/M/L) | Status |
|----|------------|-----------------------------------------------|----------|-------------------|------------------|--------|
| H1 | Class imbalance causing poor minority recall | Data | Minority recall = 0.31 | Add class weights + stratified CV | High | Open |
| H2 | Over-regularized model underfitting | Model | Train ≈ Val loss early plateau | Learning rate & depth sweep | Medium | Open |
| H3 | Feature leakage from timestamp-derived target | Data | Suspicious high importance of derived_time | Remove / shift feature & compare | High | Open |
| H4 | Non-determinism causing variance in reported metrics | Infra | Run-to-run F1 swings ±0.04 | Seed + deterministic backend test | Medium | Open |
| H5 | Calibration off due to class shift | Eval | ECE high | Temp scaling vs. isotonic | Low | Open |

(Expand dynamically.)

---

## 6. Phase Structure (Master Plan – 10 Phases)
Below is the authoritative phase plan. Each phase MUST produce explicit artifacts.

| Phase | Title | Goal (Short) | Exit Artifact(s) |
|-------|-------|--------------|------------------|
| 1 | Problem Framing | Align scope & metrics | This charter, baseline snapshot |
| 2 | Data Integrity Audit | Ensure clean, leak-free data | Data audit report |
| 3 | Reproducibility | Deterministic, repeatable runs | Env + seed reproducibility log |
| 4 | Baselines & Sanity | Validate pipeline correctness | Baseline & overfit tests |
| 5 | Cross-Validation & Robustness | Reliable generalization estimate | CV metrics & variance report |
| 6 | Architecture & Hyperparam Tuning | Optimize model capacity & params | Best config + ablations |
| 7 | Interpretability & Error Analysis | Identify failure modes & bias | Explanations & error slices |
| 8 | Performance & Efficiency | Optimize latency & memory | Profiling report |
| 9 | Reliability, Fairness & Safety | Validate ethical & robust behavior | Fairness & robustness dossier |
| 10 | Integration & Monitoring | Prevent regressions post-fix | CI suite + monitoring dashboards |

---

## 7. Detailed Phase Checklists

### Phase 1: Problem Framing
- [ ] Lock metrics & thresholds
- [ ] Capture current model hash
- [ ] Export baseline metrics JSON
- [ ] Open hypotheses list (Section 5)
- [ ] Define stop criteria for debug sprint (e.g., 2 consecutive <1% gains)

### Phase 2: Data Integrity
- [ ] Schema validation script run
- [ ] Null / missing values distribution chart
- [ ] Train/Val/Test overlap hash diff = 0
- [ ] Leakage check on suspicious features
- [ ] Drift PSI & KS stats archived
- [ ] Label quality manual audit (≥50 random samples)

### Phase 3: Reproducibility
- [ ] Add `make reproduce`
- [ ] Dockerfile or env lock present
- [ ] Deterministic seeds across frameworks
- [ ] Two reruns variance check report

### Phase 4: Baselines & Sanity
- [ ] Majority class baseline
- [ ] Simple linear / tree model
- [ ] Tiny subset overfit (should reach near-zero loss)
- [ ] Learning & calibration curves stored

### Phase 5: Cross-Validation
- [ ] Stratified K-Fold implemented
- [ ] Fold metric distribution analyzed
- [ ] Learning curves (samples vs. score)
- [ ] Decision on final CV strategy documented
- [ ] (If tuning) Plan for nested CV or proper holdout

### Phase 6: Architecture & Hyperparams
- [ ] Search space YAML defined
- [ ] Random search > N trials complete
- [ ] Bayesian refinement (optional)
- [ ] Ablation matrix built
- [ ] Overfit vs. underfit diagnosed
- [ ] Best config locked & serialized

### Phase 7: Interpretability & Error Analysis
- [ ] Permutation + (optionally) SHAP importance
- [ ] Partial dependence / ICE for top K features
- [ ] Confusion matrix (overall & slices)
- [ ] Misclassification cluster analysis
- [ ] Calibration improvement (temp scaling if necessary)
- [ ] Mislabel candidate list exported

### Phase 8: Performance & Efficiency
- [ ] Profiling (CPU/GPU utilization)
- [ ] DataLoader throughput benchmark
- [ ] Latency P50/P95/P99 before & after optimizations
- [ ] Compression (if applied) delta table
- [ ] Memory peak usage log

### Phase 9: Reliability, Fairness & Safety
- [ ] Fairness metrics across defined subgroups
- [ ] Robustness tests (noise/perturbation)
- [ ] OOD detection threshold tuned
- [ ] Bias risk analysis doc
- [ ] Safety guardrails checklist passed

### Phase 10: Integration & Monitoring
- [ ] Regression unit tests (preprocessing, shapes, determinism)
- [ ] CI pipeline operational
- [ ] Model registry entry (version, hashes)
- [ ] Shadow/canary evaluation logs
- [ ] Monitoring dashboards live (drift, latency, error rate)
- [ ] Incident playbook created

---

## 8. Experiment Logging Template
Append each experiment as a block (or store externally, link here).

```yaml
- run_id: EXP_2025_09_30_001
  date: 2025-09-30
  git_commit: e5c281dc12522478efb63334a510bda6fb8d2c79
  data_manifest_hash: pending_real_data_collection
  config_file: config.yaml
  changes: "Initial multi-modal affective classifier deployment with transfer learning from GCS foundational model"
  metrics:
    architecture: "GNN (frozen) + Physio branch (64 units) + Voice branch (64 units) + Fusion (128 units)"
    dropout_rate: 0.3
    emotion_classes: ["ANXIETY", "DEPRESSION", "JOY", "ANGER"]
    output_format: "valence-arousal regression"
  comparison_to_prev: "Baseline - first deployment"
  interpretation_notes: "Transfer learning approach implemented; GCS foundational model frozen for EEG feature extraction; ready for LOSO cross-validation on real data"
  next_action: "Run LOSO cross-validation with real EEG data; benchmark fusion architecture performance; validate empathy response generation"
```

---

## 9. Risk Register
| Risk ID | Description | Likelihood | Impact | Mitigation | Status |
|---------|-------------|-----------|--------|------------|--------|
| R1 | Hidden feature leakage | Medium | High | Delay derived target features; audit pipeline | Open |
| R2 | Overfitting if test reused | High | High | Freeze test set; enforce PR guard | Open |
| R3 | Metric drift unmonitored post-deploy | Medium | Medium | Monitoring + alerts | Open |
| R4 | Fairness gap > threshold | Medium | High | Reweighting / threshold calibration | Open |

---

## 10. Governance & Change Control
- All model-impacting changes require PR with:
  - Experiment diff summary
  - Metrics before/after (same seed or averaged across k folds)
  - Statistical significance check (bootstrap or paired t-test)
- Prohibited:
  - Using test set repeatedly for tuning
  - Merging without reproducible run evidence
- Required Labels (suggestion):
  - `debug/data`, `debug/model`, `debug/infra`, `fairness`, `perf`, `monitoring`

---

## 11. Monitoring Specification (Post-Deployment)
| Channel | Metric | Threshold | Action |
|---------|--------|-----------|--------|
| Drift | PSI (top 10 features) | >0.2 | Trigger data audit job |
| Performance | F1 macro weekly | -5% from baseline | Open incident |
| Latency | P95 | >200ms | Scale infra / profile |
| Fairness | Max subgroup ΔF1 | >0.12 | Retrain, reweight |
| Calibration | ECE | >0.07 | Refit temperature scaling |

---

## 12. Open Questions / To Clarify
- Are there time-based splits required (temporal leakage risk)?
- Are empathy scores ordinal, continuous, or categorical?
- Is there a human-in-the-loop feedback pipeline?
- Required compliance / regulatory constraints?

---

## 13. Immediate Next Actions (Fill Owner & Due Date)
| Action | Owner | Due | Status |
|--------|-------|-----|--------|
| Create data schema validation script |  |  |  |
| Add deterministic seed utility |  |  |  |
| Trivial baseline + overfit 100-sample test |  |  |  |
| Implement stratified K-Fold harness |  |  |  |
| Define hyperparameter search YAML |  |  |  |
| Add SHAP/permutation pipeline |  |  |  |
| Set up profiling script |  |  |  |
| Draft fairness metrics function |  |  |  |
| Add regression preprocessing tests |  |  |  |

---

## 14. Appendix

### 14.1 Suggested Repository Structure (If Not Present)
```
/configs
/data_manifest
/docs
/models
/notebooks
/scripts
/tests
```

### 14.2 Suggested Config Snippet (Example)
```yaml
model:
  type: gradient_boosting
  params:
    learning_rate: 0.05
    max_depth: 6
    n_estimators: 400
    subsample: 0.85
    colsample: 0.8
training:
  seed: 42
  early_stopping_rounds: 40
  eval_metric: f1
cv:
  folds: 5
  stratified: true
logging:
  tracking: mlflow
```

### 14.3 Ablation Table (Populate)
| Component | Enabled? | ΔF1 Macro | ΔLatency | Notes |
|-----------|----------|-----------|----------|-------|
| Feature Group A | Y | + |  |  |
| Dropout Layer | N |  |  |  |
| Class Weights | Y | + |  |  |

---

## 15. Change Log (Maintain Chronologically)
| Date | Change | Author | Notes |
|------|--------|--------|-------|
| 2025-09-30 | Initial charter draft | @V1B3hR | Created structure |
| 2025-09-30 | Updated after initial deployment | GCS Development Team | Populated with actual project details, affective classifier debugging scope |

(Extend as modifications are made.)

---

## 16. Usage Instructions
1. Do not modify historical experiment records—append new entries.
2. Keep hypotheses list lean; archive invalidated ones with Status = Rejected.
3. Reference this file in all debugging-related PR descriptions.
4. Sync with monitoring dashboards post-deployment to update thresholds if baseline shifts.

---

## 17. Completion Criteria for This Debug Cycle
All checked before declaring “stabilized”:
- [ ] Target metric(s) ≥ defined thresholds
- [ ] Fairness gap within bounds
- [ ] Calibration acceptable
- [ ] Monitoring + CI operational
- [ ] Documentation updated (this file + CHANGELOG_DEBUG.md)
- [ ] Risk register items either mitigated or accepted explicitly

---

(End of Charter)
