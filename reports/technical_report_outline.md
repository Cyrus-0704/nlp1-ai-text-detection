# Technical Report Outline

> Anonymous submission draft. Remove team, school, member names, local paths, and repository-only details before submission.

## Abstract

Task, final method, and verified result: [complete after the solution is frozen].

## 1. Task and Motivation

Binary human-versus-machine text classification for the preliminary round. Distinguish source detection from factuality assessment.

## 2. Data and Evaluation

### 2.1 Data processing and fixed splits

[A supplies the frozen data version, preprocessing, grouping, and split description.]

### 2.2 Metrics and protocol

[A supplies the agreed macro-F1, short-text definition, threshold, and score formula. Mark any provisional convention clearly.]

## 3. Methods

### 3.1 Character TF-IDF and logistic-regression baseline

[Record feature range, vocabulary limits, class weighting, regularization, and train-only fitting.]

### 3.2 Main model

[B supplies the encoder, initialization/revision, training setup, and inference method.]

### 3.3 Fusion or calibration

[Include only if dev experiments show a repeatable benefit; otherwise explain why the main model was retained.]

## 4. Experiments and Results

### 4.1 Baseline and main-model comparison

[Use matched data versions and the agreed evaluation protocol.]

### 4.2 Ablations and short-text analysis

[Report all planned comparisons, sample counts, and limitations.]

### 4.3 Error analysis and generalization

[Use internal development data; do not claim hidden-generator or official-test results without evidence.]

## 5. Reproducibility

[Software versions, hardware, commands, data hashes, seeds, model source/revision, and external-resource declarations.]

## 6. Limitations

[Data coverage, provisional metric conventions, model coverage, and remaining uncertainty.]

## References and Declarations

[Add verified references, pretrained-model/license declarations, and required competition statements.]