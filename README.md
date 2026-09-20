# Semantic Resume ATS

AI-powered resume analysis and job matching. Upload a resume for a general
ATS-style review, or add a job description to get a match score, skill-gap
analysis, and tailored improvement suggestions.

## What it does

**Resume only:**
- Extracts and analyzes resume text (PDF/DOCX)
- Detects skills against a curated taxonomy (110+ skills, 12 categories)
- Produces a transparent, weighted ATS score (completeness, skill breadth,
  structure quality, role-alignment signal)
- Recommends the top matching job roles, ranked by a combination of semantic
  embedding similarity and skill overlap, each with a grounded explanation

**Resume + job description:**
- Computes an ATS match score specific to that resume/JD pair (semantic
  similarity, skill match, keyword relevance, experience alignment, resume
  completeness)
- Shows matching and missing skills
- Generates tiered improvement suggestions (Critical / Recommended / Optional),
  grounded in the actual detected gaps — never suggests claiming skills or
  experience the candidate doesn't have

## Why it was built

Most ATS tools are simple keyword matchers. This project explores a more
semantic approach — fine-tuning a sentence embedding model specifically for
resume-job matching — while keeping the system honest about what that
approach can and can't do, and keeping a fast, free LLM (Groq) in a
supporting role for language tasks rather than as the source of truth for
scoring.

## Architecture

\```
Resume/JD (PDF or DOCX or text)
        │
        ▼
  Text extraction (pdfplumber / python-docx)
        │
        ▼
  Skill extraction (spaCy PhraseMatcher + curated taxonomy)
        │
        ▼
  Sentence embedding (fine-tuned MPNet, falls back to pretrained if unavailable)
        │
        ▼
  Transparent weighted ATS scoring  ◄────  Groq (structured JSON: summaries,
        │                                   JD analysis, tiered improvements)
        ▼
  FastAPI JSON response  ──────────────►  React frontend
\```

**Backend:** Python, FastAPI, Sentence Transformers, spaCy, pdfplumber,
python-docx, Groq API.
**Frontend:** React (Vite), Tailwind CSS v4.
**No database** — the app is stateless; the frontend holds resume/JD text
in memory between the initial analysis and follow-up calls (recommended
jobs / improvements).

## Machine learning approach

**Base model:** `sentence-transformers/all-mpnet-base-v2` (768-dim embeddings).

**Fine-tuning:** `MultipleNegativesRankingLoss`, trained on
[`cnamuangtoun/resume-job-description-fit`](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit) —
a public dataset of 8,000 real resume/job-description pairs labeled `Good Fit`
/ `Potential Fit` / `No Fit` (6,240 train / 1,760 test, official split).

Training pairs are built from `Good Fit` labels as positives. Where the same
resume also has a `No Fit` job description in the training data, that pair
becomes an explicit **hard negative** (a real anchor/positive/negative
triplet); otherwise the pair trains with in-batch negatives only.
`Potential Fit` rows are excluded from both training and evaluation as
genuinely ambiguous. Full training/evaluation code:
[`notebooks/finetune_resume_job_mpnet.ipynb`](notebooks/finetune_resume_job_mpnet.ipynb).

### Results (4 epochs, held-out test set, 233 evaluation queries)

Retrieval — ranking the correct job description against all unique test JDs:

| Method | Precision@1 | Precision@5 | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|
| **Fine-tuned MPNet** | **0.258** | **0.130** | **0.411** | **0.415** | **0.331** |
| Pretrained MPNet | 0.060 | 0.058 | 0.193 | 0.181 | 0.128 |
| TF-IDF | 0.099 | 0.067 | 0.202 | 0.216 | 0.148 |

Classification — distinguishing `Good Fit` from `No Fit` pairs:

| Method | AUC-ROC | F1 (at selected threshold) |
|---|---|---|
| **Fine-tuned MPNet** | **0.703** | 0.357 |
| Pretrained MPNet | 0.508 | 0.366 |
| TF-IDF | 0.589 | 0.442 |

**What this means:** fine-tuning produced a clear, consistent improvement on
every retrieval metric (roughly 2-4x over the pretrained baseline) and on
AUC-ROC — pretrained MPNet's 0.508 is barely above chance at separating fits
from non-fits, while the fine-tuned model reaches 0.703, a real if moderate
level of separation. This is why the app's role-matching and semantic-similarity
scoring rely on the fine-tuned model rather than the pretrained one whenever
it's present.

**A real limitation, not hidden:** F1-at-threshold does *not* show the same
win — fine-tuned (0.357) actually trails TF-IDF (0.442). AUC measures ranking
quality across all thresholds; F1 depends on one threshold, which here was
selected from a small validation slice and likely doesn't generalize well.
This is a genuine weak point of a single fixed decision boundary — which is
part of why the app never uses one: `ats_service.py` treats semantic
similarity as one continuous, weighted score component, and `role_matcher.py`
does ranking rather than binary classification, so the metrics that transfer
most directly to how this model is actually used (retrieval, AUC) are exactly
the ones where fine-tuning won clearly.

This is an experimental semantic matching component with demonstrated,
moderate improvement over baselines — not a production-grade or
commercial-ATS-equivalent scoring system, and the app's UI and score
explanations say so directly.

## Setup

### Backend
\```bash
cd backend
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r ../requirements.txt
python -m spacy download en_core_web_sm 2>/dev/null || true  # optional; app works without it
cp ../.env.example .env   # add your GROQ_API_KEY (free tier at console.groq.com)
uvicorn app.main:app --reload
\```
On first run, if no fine-tuned model is present at
`backend/models/resume_job_mpnet_finetuned/`, the app automatically downloads
and uses the pretrained `all-mpnet-base-v2` instead — everything still works,
just with the pretrained-baseline numbers shown above rather than the
fine-tuned ones. To use the fine-tuned model, run the notebook (see
`notebooks/README.md`) and place its output at that path.

### Frontend
\```bash
cd frontend
npm install
cp .env.example .env   # VITE_API_URL, defaults to http://localhost:8000
npm run dev
npm test                # run the interaction test suite
\```

## API endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service status + which embedding model is active |
| `POST /analyze/resume` | Resume-only analysis (upload a PDF/DOCX) |
| `POST /recommend/jobs` | Ranked role recommendations from resume text |
| `POST /analyze/match` | Resume + JD match score (JD as text or file) |
| `POST /recommend/improvements` | Tiered, JD-specific improvement suggestions |

Full request/response schemas are visible at `/docs` when the backend is running.

## Project structure

\```
SemanticResumeATS/
├── backend/
│   ├── models/resume_job_mpnet_finetuned/   # fine-tuned model goes here (gitignored)
│   └── app/
│       ├── main.py, config.py
│       ├── api/routes/                       # health, resume, matching
│       ├── parsers/                          # PDF/DOCX extraction, validation
│       ├── ml/                               # embeddings, similarity, skills, role matching
│       ├── data/                             # skills taxonomy, role profiles
│       ├── services/                         # ATS scoring, Groq integration, orchestration
│       └── schemas/                          # Pydantic request/response models
├── frontend/
│   └── src/
│       ├── api/client.js
│       └── components/{layout,upload,results}/
├── notebooks/
│   └── finetune_resume_job_mpnet.ipynb
└── requirements.txt, .env.example
\```

## Limitations

- Fine-tuned on a moderate-sized public dataset (8,000 rows, 642 unique
  resumes × 280 unique job descriptions) — real improvement over baselines,
  but not a large-scale or production-grade result.
- F1-at-threshold is unstable (see Results above) — the model is stronger at
  ranking/retrieval than at binary classification with a fixed threshold.
- Skill extraction is taxonomy-based (110+ skills) — it will miss skills,
  tools, or niche terminology not in that list.
- Experience-alignment scoring is a heuristic based on detected year ranges
  in resume text, not real structured work-history parsing.
- No database — resume/JD text lives only in the browser session; nothing
  is persisted between visits.
- Scanned/image-only PDFs (no embedded text layer) aren't supported — no OCR.

## Possible future improvements

- Mine multiple hard negatives per resume, not just the first available
- Incorporate `Potential Fit` as a graded/soft label instead of discarding it
- Combine with additional datasets to increase resume/JD diversity
- Hyperparameter sweep using the validation evaluator already in the notebook
- A lightweight cross-encoder re-ranking stage on top of retrieval results
- Error analysis on the test pairs the model gets most wrong
