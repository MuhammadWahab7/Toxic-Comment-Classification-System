# Model Card

## Intended Use

The repository provides an academic baseline for binary toxic-comment screening.
It is suitable for coursework, reproducible experiments, feature comparisons,
and human-in-the-loop moderation research.

It is not suitable as the sole mechanism for account suspension, content removal,
law-enforcement referral, employment decisions, or safety-critical intervention.

## Baseline Architecture

- Word TF-IDF features using unigrams and bigrams
- Character TF-IDF features using 3-5 character n-grams
- Balanced Logistic Regression
- Stratified train/test split with random state `42`

The baseline is trained when the user supplies a compatible dataset. No fitted
model is committed to this repository.

## Historical Experiment

The preserved report records an earlier ensemble experiment. Random Forest had
the highest reported weighted F1 (`87.12%`), but toxic-class F1 was only `31.84%`.
These metrics were not reproduced during repository packaging because the source
dataset is absent. The report also references XGBoost and LightGBM, while the
preserved notebook source currently uses scikit-learn ensemble models instead.

## Risks

- Identity terms and reclaimed language may receive false-positive predictions.
- Sarcasm, quotation, counterspeech, spelling variation, and context are difficult.
- Training labels may reflect annotator culture and platform-specific norms.
- Overall accuracy can conceal poor minority-class detection.
- Distribution shift can invalidate thresholds and performance estimates.

## Required Deployment Controls

- human review and user appeals;
- platform-specific validation and threshold selection;
- toxic-class and subgroup metrics;
- drift monitoring and periodic reassessment;
- privacy-preserving logs with limited retention;
- documented rollback and incident-response procedures.
