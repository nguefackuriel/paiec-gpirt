# GP-IRT with per-benchmark Bayesian model averaging — PAIEC 2026 submission

Team **Pooled Priors** — Uriel Nguefack Yefou. Technical report: [`report/report.pdf`](report/report.pdf).

Predictor for the [Predictive AI Evaluation Competition](https://aimslab.stanford.edu/competition)
(NeurIPS 2026). Given an AI system (subject) and a benchmark item, it predicts the
probability of a correct response, and adapts online as labels are revealed
(budgets 0, 1, 3, 7, 15, 31 per subject–benchmark pair).

## Method in one paragraph

For each (anonymous) benchmark, all revealed labels — from every subject — are pooled
into a hierarchical logistic model `logit p = m + a_subject + d_item`. Item effects
have a Gaussian-process prior whose kernel combines (i) an RBF on the cosine
similarity of text embeddings, (ii) a cosine kernel on `key=value` tokens of the item
metadata, (iii) a **linear kernel on benchmark-centered embeddings** (Bayesian linear
regression of difficulty on embedding directions, learned online per benchmark), and
(iv) an identity term. The benchmark shift has an empirical prior centred at -0.6 on the logit scale
(a mixture of three scales), because groups of realistic size have a mean accuracy near 0.36 in the training
pool and near 0.35 in the formative test draws. The model is fit by a Laplace approximation (Newton's method
with a Schur complement over the diagonal subject block, so cost is linear in the
number of labels), for a small grid of prior scales; predictions are averaged with
weights proportional to the Laplace marginal likelihood (per-benchmark BMA) and
mapped to probabilities with a probit approximation of the predictive integral.
A cross-benchmark prior offset `0.3 * z_item` comes from a ridge regression of item accuracy
on Qwen2.5-7B-Instruct hidden states (trained on the public training pool), and the final
predictor averages two such models whose item kernels use e5-small embeddings and Qwen hidden
states respectively (falls back to the e5-only model without prior if no GPU is available).
Labels are acquired with the platform's default random policy.

Simulated Brier ALC (leave-benchmark-out, 1000 draws whose composition matches the formative
feedback): constant 0.2500, group mean 0.1869, final model **0.1797** (see `report/report.pdf`).
On 1000 new draws the centred shift prior gains 0.0022 ALC over the same model with a prior centred at zero.

## Repository layout

```
paiec/            library: simulator (sim.py), GP-IRT (gpirt.py), models, features, tokens, policies
scripts/          data download/processing, CV experiments, embedding, packaging, remote helpers
submission/       runtime package (model.py + gpirt.py + tokens.py + embed.py + config.json + models.txt)
dist/             built submission zips (generated)
data/             downloaded and processed data (generated, not distributed)
```

## Reproducing

Python 3.10+ with `numpy pandas pyarrow scipy torch transformers huggingface_hub`
(`lightgbm` only for the negative budget-0 experiment).

1. Request access to the gated dataset `aims-foundations/measurement-db`, then
   ```bash
   HF_TOKEN=... python scripts/download_data.py data/mdb      # tables only, pinned revision
   python scripts/make_catalog.py         # data/benchmarks_catalog.csv
   python scripts/build_dataset.py        # data/proc/{obs,items,subjects}.parquet
   python scripts/draw_stats.py           # eligible groups + benchmark folds
   python scripts/prep_codes.py           # integer-coded tables
   python scripts/embed_items_tf.py intfloat/multilingual-e5-small data/proc/emb_e5small.npy
   python scripts/make_item_stats.py
   # Qwen hidden states (GPU, ~1-2.5 h on an A100) and the cross-benchmark prior
   python scripts/llm_embed.py Qwen/Qwen2.5-7B-Instruct qwen7b 0.5,0.75,1.0
   python scripts/norm_emb.py data/proc/emb_qwen7b_L21.npy
   python scripts/b0_ridge.py data/proc/emb_qwen7b_L21_z.npy      # out-of-fold prior for CV
   python scripts/build_qwen_prior.py                              # final prior -> submission/qwen_prior.npz
   ```
2. Cross-validated simulation of the official protocol (leave-benchmark-fold-out,
   draws limited to 1,000 subject–item pairs):
   ```bash
   python scripts/make_group_acc.py       # data/group_acc.csv (accuracy of every training group)
   PAIEC_MAX_GROUP=200 python scripts/final_table.py 200 data/table_small.csv     # Table 1, realistic regime
   PAIEC_MAX_GROUP=1000 python scripts/final_table.py 200 data/table_pairs.csv    # large-group regime
   PAIEC_DRAW=distinct python scripts/final_table.py 8 data/table_distinct.csv    # many-subject regime
   ```
   PAIEC_MAX_GROUP=200 python scripts/confirm_v8.py 200 1     # final vs previous model on new draws
   PAIEC_MAX_GROUP=200 python scripts/oracle.py 200            # oracle levels
   python scripts/fetch_feedback.py <id>  # your own formative logs (not distributed), for formative_acc.py
   python scripts/formative_acc.py && python scripts/plot_prior.py && python scripts/plot_curves.py   # figures
   Other experiments (including the negative results) are in `scripts/try_*.py`.
3. Build and check the submission:
   ```bash
   python scripts/parity_v8.py            # runtime == simulator predictions (needs GPU + Qwen)
   python scripts/build_zip.py v8_final   # dist/v8_final.zip
   python <starting_kit>/tools/check_submission_zip.py dist/v8_final.zip
   python scripts/submit.py dist/v8_final.zip    # Codabench API (token in ~/.config/paiec/codabench-token)
   ```

## Use of tools and models

Pretrained models: `intfloat/multilingual-e5-small` and `Qwen/Qwen2.5-7B-Instruct` (item
representations, used offline and at runtime). Training data: only `aims-foundations/measurement-db` (public training pool).
Development used an LLM coding assistant (Claude Code); see the technical report.

## License and data

Code: MIT (see `LICENSE`). No evaluation data are distributed in this repository. The only
data-derived artifact is `submission/qwen_prior.npz` (ridge-regression weights fit on the public
training pool `aims-foundations/measurement-db`, whose curation is licensed CC BY-SA 4.0; individual
benchmarks keep their upstream licenses). The starting kit and baseline repository are available from
the organizers and are not included.
