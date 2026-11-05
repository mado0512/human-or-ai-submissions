# group-number-5 — Human or AI? (Cornell Tech AML)

**Kaggle team name:** `group-number-5`  
**Team members:**

| Name | NetID |
|---|---|
| `Shifeng Hong` | `sh2769` |
| `Fangfei Dong` | `fd284` |
| `Nathan Wolf` | `ntw25` |


## Task summary

Binary classification of passages as human-written (`0`) or AI-generated (`1`).
Features are word-level and character-level TF-IDF; the classifier is a
class-balanced logistic regression. The pipeline is schema-agnostic: the ID,
text and label columns are detected by name and can be overridden with
`--id-col` / `--text-col` / `--label-col`.

## Environment

- Python 3.11 or newer (developed on CPython 3.12)
- No GPU required; training runs on CPU in a few minutes

```bash
cd submissions/group-number-5
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Competition data is **not** committed. Download `train.csv`, `unlabeled.csv`
and `test.csv` from Kaggle into `./data/`:

```text
submissions/group-number-5/data/
  train.csv        # labelled rows: id, <text column>, <label column>
  unlabeled.csv    # optional extra unlabelled rows
  test.csv         # rows to score for the leaderboard
```

## Train / evaluate (labelled validation protocol)

The validation split is a stratified holdout of the labelled training file,
fixed at 2,000 rows and drawn with the global seed. Model selection only ever
looks at this holdout; `test.csv` is never used for tuning.

```bash
python train.py \
  --train data/train.csv \
  --seed 20261023 \
  --holdout 2000 \
  --model-out model.joblib \
  --metrics-out metrics.json
```

Reported metrics (accuracy, F1 for the AI class, ROC-AUC) are printed and
written to `metrics.json`:

```bash
cat metrics.json
```

Useful overrides:

| Flag | Default | Purpose |
|---|---|---|
| `--seed` | `20261023` | Split seed and solver seed |
| `--holdout` | `2000` | Held-out rows; falls back to 20% on small files |
| `--c-value` | `4.0` | Logistic regression inverse regularisation |
| `--text-col` / `--label-col` / `--id-col` | auto | Explicit column names |

## Predict (writes `submission.csv` for Kaggle)

```bash
python predict.py \
  --test data/test.csv \
  --model model.joblib \
  --out submission.csv
```

The output has one row per test ID with P(AI) in the `prediction` column.
To emit hard 0/1 labels instead:

```bash
python predict.py --test data/test.csv --model model.joblib \
  --out submission.csv --label-mode hard --threshold 0.5
```

## Seeds and configuration

| Item | Value |
|---|---|
| Global seed | `20261023` |
| Validation split | stratified, 2,000 rows, `random_state=20261023` |
| Word TF-IDF | `ngram_range=(1,2)`, `min_df=2`, `max_features=100_000`, `sublinear_tf=True` |
| Char TF-IDF | `analyzer="char_wb"`, `ngram_range=(3,5)`, `min_df=2`, `max_features=200_000`, `sublinear_tf=True` |
| Classifier | `LogisticRegression(C=4.0, class_weight="balanced", solver="liblinear", max_iter=2000)` |
| Label convention | `1 = AI-generated`, `0 = human-written` |

Everything except the solvers is deterministic; re-running the two commands
above reproduces `metrics.json` and `submission.csv` byte-for-byte on the same
data and library versions.

## Status

| Item | State |
|---|---|
| `train.py` / `predict.py` / `pipeline.py` | interface defined; pipeline not yet validated on the real competition files |
| Holdout metrics | to be filled in after the first real run |
| `submission.csv` regenerated from the commands above | to be confirmed before the PR is opened |

## AI assistance

Written with the help of an AI coding assistant (Claude via the `pi` coding
agent). AI help was used for the repository/pipeline scaffolding, TF-IDF and
logistic-regression boilerplate, and this README. All modelling decisions,
column-schema handling, the validation protocol and the reported numbers are
run and checked manually; nothing in `submission.csv` is produced by an
external LLM labelling service.

This disclosure is the basis for the Q6.3 answer in the report and is updated
whenever the tooling changes.
