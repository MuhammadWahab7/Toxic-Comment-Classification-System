# Dataset Card

## Expected Data

The baseline requires two columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `comment_text` | string | User-generated text to classify |
| `toxic` | integer | `0` for non-toxic, `1` for toxic |

CSV, XLSX, XLS, and JSON files are supported. Additional columns are ignored by
the baseline. The preserved notebook may inspect or transform additional fields.

## Original Course Dataset

The historical report describes 223,549 comments and 11 columns, with 21,384
toxic rows (9.57%). That 388 MB file is not included because it exceeds GitHub's
standard file limit and its redistribution terms were not established here.

A compatible public source is the
[Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data).
Download data directly under the applicable Kaggle competition rules and terms.

## Privacy and Safety

Toxic-comment datasets may contain slurs, threats, identity references, personal
information, or other distressing text. Before use:

- confirm a lawful basis and the dataset license;
- minimize access and avoid publishing raw comments in logs or screenshots;
- remove direct identifiers and secrets;
- document annotation methods and known demographic bias;
- restrict exposure for reviewers who did not consent to harmful content;
- never upload private production conversations to public notebooks.

## Evaluation Guidance

Always report toxic-class precision, recall, and F1 alongside overall accuracy.
Use subgroup and false-positive analysis where identity-related text is present.
Do not compare results across datasets without documenting split, preprocessing,
sampling, label definitions, and threshold differences.
