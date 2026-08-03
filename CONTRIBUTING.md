# Contributing

Contributions should improve correctness, reproducibility, responsible evaluation,
documentation, privacy, or moderation safeguards.

## Setup

1. Create and activate a virtual environment.
2. Install `requirements.txt`.
3. Install the package with `python -m pip install -e . --no-deps`.
4. Run `python -m compileall -q src tests`.
5. Run `python -m unittest discover -s tests -v`.

## Pull Request Checklist

- Keep raw datasets, trained models, logs, and secrets out of Git.
- Add tests for validation, preprocessing, metrics, or CLI behavior changes.
- Report toxic-class metrics, not overall accuracy alone.
- Document the dataset, split, threshold, random state, and sampling changes.
- Discuss expected fairness, privacy, and moderation impact.
- Do not silently rewrite the preserved historical report or notebook evidence.

## Research Integrity

Distinguish reproduced results from historical or third-party claims. Do not claim
production readiness without deployment-specific validation and human oversight.
