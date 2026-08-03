# Security Policy

## Reporting

Do not publish sensitive details in a public issue. Use the repository owner's
private GitHub contact or a GitHub Security Advisory when available. Describe the
affected component, impact, safe reproduction steps, and recommended remediation.

## Data Handling

- Do not commit private comments, direct identifiers, credentials, or access tokens.
- Keep downloaded datasets and generated model files outside version control.
- Sanitize logs and screenshots before sharing them.
- Follow the source dataset's license and access rules.

## Serialized Models

Joblib and Pickle files can execute code during loading. Only load model artifacts
created locally or obtained from a trusted, verified source. Generated artifacts
are ignored by default.

## Dependency and Notebook Safety

Review notebook cells before execution. The preserved notebook installs packages
and uses Colab upload/download APIs. Run it only in an isolated environment and do
not upload confidential datasets to third-party notebook services.
