# GP-IRT fit at test time, PAIEC 2026

Team **Pooled Priors**, Uriel Nguefack Yefou. Report: [`report/report.pdf`](report/report.pdf).

This is our entry to the [Predictive AI Evaluation Competition](https://aims.stanford.edu/competition)
(NeurIPS 2026). For an AI system (a subject) and a benchmark item, it predicts the chance that the
answer is correct. Labels are shown a few at a time (0, 1, 3, 7, 15 and 31 per subject and benchmark),
and the predictions are updated each time.

## How it works

For each benchmark we fit a logistic model, `logit p = c_item + m + a_subject + d_item`, with the labels
of all subjects on that benchmark.

- `m` is the level of the benchmark. Its prior is centred at -0.6 on the logit scale, with three
  scales mixed together. We centre it there because groups of the size seen at test time are hard:
  their mean accuracy is about 0.36 in the training pool and 0.35 in the formative test draws.
- `a_subject` is the level of the subject on this benchmark.
- `d_item` is the item effect. It has a Gaussian process prior, so items with close text get close
  difficulty. The kernel adds up an RBF on text embeddings, a cosine kernel on `key=value` metadata
  tokens, a linear kernel on embeddings centred inside the benchmark, and an identity term.
- `c_item` is a fixed offset, `0.3 * z_item`, where `z_item` comes from a ridge regression of item
  accuracy on Qwen2.5-7B-Instruct hidden states, trained on the public training pool.

We fit the model with a Laplace approximation (Newton steps, with the subject block removed by a Schur
complement, so the cost grows linearly with the number of labels). We do this for a small grid of prior
scales and average the predictions with weights from the marginal likelihood. The final predictor is
the mean of two such models: one uses e5-small embeddings in the kernel, the other uses Qwen hidden
states. Without a GPU it falls back to the e5 model alone, with no offset. Labels are chosen with the
platform's default random order.

In our simulation of the protocol on held-out benchmarks (1000 draws that look like the formative
ones), the Brier ALC is 0.2500 for a constant 0.5, 0.1869 for a group mean, and **0.1797** for the
final model. On 1000 new draws, centring the benchmark prior gains 0.0022 ALC over the same model with
a prior centred at zero.

## Folders

```
paiec/            library: simulator (sim.py), GP-IRT (gpirt.py), models, features, tokens, label policies
scripts/          data download and processing, cross-validation experiments, embeddings, packaging
submission/       the code that runs on Codabench (model.py, gpirt.py, tokens.py, embed.py, config.json, models.txt)
report/           the report and its figures
```

`data/` and `dist/` are made by the scripts and are not in this repository.

## How to reproduce

You need Python 3.10 or newer with `numpy pandas pyarrow scipy torch transformers huggingface_hub`
(`lightgbm` is only used in one experiment that did not help).

1. Ask for access to the gated dataset `aims-foundations/measurement-db`, then build the data:
   ```bash
   HF_TOKEN=... python scripts/download_data.py data/mdb      # tables only, fixed revision
   python scripts/make_catalog.py         # data/benchmarks_catalog.csv
   python scripts/build_dataset.py        # data/proc/{obs,items,subjects}.parquet
   python scripts/draw_stats.py           # groups that can be drawn, and benchmark folds
   python scripts/prep_codes.py           # tables with integer codes
   python scripts/embed_items_tf.py intfloat/multilingual-e5-small data/proc/emb_e5small.npy
   python scripts/make_item_stats.py
   # Qwen hidden states (GPU, 1 to 2.5 hours on an A100) and the prior across benchmarks
   python scripts/llm_embed.py Qwen/Qwen2.5-7B-Instruct qwen7b 0.5,0.75,1.0
   python scripts/norm_emb.py data/proc/emb_qwen7b_L21.npy
   python scripts/b0_ridge.py data/proc/emb_qwen7b_L21_z.npy      # out-of-fold prior for cross-validation
   python scripts/build_qwen_prior.py                              # final prior, saved to submission/qwen_prior.npz
   ```
2. Run the simulation of the official protocol (benchmark folds held out, draws of at most 1000
   subject and item pairs):
   ```bash
   python scripts/make_group_acc.py       # data/group_acc.csv, accuracy of every training group
   PAIEC_MAX_GROUP=200 python scripts/final_table.py 200 data/table_small.csv     # Table 1, realistic setting
   PAIEC_MAX_GROUP=1000 python scripts/final_table.py 200 data/table_pairs.csv    # large groups
   PAIEC_DRAW=distinct python scripts/final_table.py 8 data/table_distinct.csv    # many subjects
   PAIEC_MAX_GROUP=200 python scripts/confirm_v8.py 200 1     # final model against the previous one, new draws
   PAIEC_MAX_GROUP=200 python scripts/oracle.py 200            # oracle levels
   python scripts/fetch_feedback.py <id>  # your own formative logs (not shared here), used by formative_acc.py
   python scripts/formative_acc.py && python scripts/plot_prior.py && python scripts/plot_curves.py   # figures
   ```
   The other experiments, including the ones that did not help, are in `scripts/try_*.py`.
3. Build and check the submission:
   ```bash
   python scripts/parity_v8.py            # checks that the runtime gives the simulator's predictions (needs GPU and Qwen)
   python scripts/build_zip.py v8_final   # dist/v8_final.zip
   python <starting_kit>/tools/check_submission_zip.py dist/v8_final.zip
   python scripts/submit.py dist/v8_final.zip    # Codabench API (token in ~/.config/paiec/codabench-token)
   ```

## Models, data and tools

Pretrained models: `intfloat/multilingual-e5-small` and `Qwen/Qwen2.5-7B-Instruct`, used to represent
items, both offline and at run time. Training data: only the public training pool
`aims-foundations/measurement-db`. We used an LLM coding assistant (Claude Code) during development; the
report says more.

## License

Code: MIT (see `LICENSE`). This repository has no evaluation data. The only file made from data is
`submission/qwen_prior.npz`, the ridge weights fit on `aims-foundations/measurement-db` (its curation is
under CC BY-SA 4.0, and each benchmark keeps its own license). The starting kit and the baseline come
from the organizers and are not included.
