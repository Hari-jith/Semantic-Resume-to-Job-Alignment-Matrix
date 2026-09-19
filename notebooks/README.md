# Fine-tuning notebook

`finetune_resume_job_mpnet.ipynb` fine-tunes `sentence-transformers/all-mpnet-base-v2`
on real, labeled resume-job fit data and evaluates it against pretrained-MPNet and
TF-IDF baselines.

**This notebook has not been run yet.** No results exist until you run it — see
the honesty note in the notebook's first cell.

## Dataset
[`cnamuangtoun/resume-job-description-fit`](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit)
— 8,000 real resume/job-description pairs, labeled `Good Fit` / `Potential Fit` / `No Fit`.
Loaded automatically via the `datasets` library; no download or manual setup needed.

## How to run
1. Open the notebook in Google Colab: File → Upload notebook, or drag it into
   [colab.research.google.com](https://colab.research.google.com).
2. Runtime → Change runtime type → select a GPU (T4 is fine, and free-tier).
3. Run all cells top to bottom.
4. Training takes roughly 15-40 minutes on a free-tier T4, depending on the
   final dataset size after cleaning (printed in Section 5).

## What you get
- A fine-tuned model folder (`resume_job_mpnet_finetuned/`)
- A comparison table: fine-tuned vs. pretrained vs. TF-IDF, across retrieval
  metrics (Precision@1, Precision@5, Recall@5, MRR, nDCG@5) and classification
  metrics (AUC-ROC, F1) — computed for real, on the untouched test split
- Similarity-distribution plots showing whether fine-tuning actually separated
  "Good Fit" from "No Fit" pairs better than the pretrained baseline did

## Plugging the result into the backend
The last cell zips and downloads the model folder. Unzip it and place it at:
```
backend/models/resume_job_mpnet_finetuned/
```
`backend/app/ml/embedding_model.py` (Module 3) automatically detects and loads
it from there on the next backend restart — no code changes needed. Check
`/health` afterward; `embedding_model.model_type` should read `"fine_tuned"`
instead of `"pretrained_fallback"`.

## If the results are underwhelming
That's a real possible outcome, not a failure of the notebook — 8,000 rows is
a modest dataset for fine-tuning a 110M-parameter model. Section 12 of the
notebook lists concrete next steps in rough order of expected effort-to-benefit.
Whatever the numbers turn out to be, report them as-is — don't round up.
