<div align="center">

# NCERT Science RAG

### A Multi-Subject Retrieval-Augmented Generation System  
### for NCERT Class 11–12 Science Question Answering

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Colab](https://img.shields.io/badge/Run_on-Colab_L4-orange?logo=googlecolab)](https://colab.research.google.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![arXiv](https://img.shields.io/badge/Paper-IEEE_Conference-red)](paper/)
[![HuggingFace](https://img.shields.io/badge/Model-Qwen2.5--1.5B--Instruct-yellow?logo=huggingface)](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)

**Physics · Chemistry · Mathematics · Biology**  
**14,450 enriched chunks · 99 chapters · 573-question evaluation · Recall@10 = 94.6%**

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [What to Upload](#what-to-upload)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Dataset](#dataset)
- [Evaluation](#evaluation)
- [Figures](#figures)
- [Paper](#paper)
- [Citation](#citation)

---

## Overview

This repository contains the **complete implementation** of a production-ready, multi-subject Retrieval-Augmented Generation (RAG) system for question answering grounded in the NCERT Class 11 and 12 Science curriculum, spanning **Physics, Chemistry, Mathematics, and Biology**.

The system builds on the foundational RAG paradigm introduced by [Lewis et al. 2020 (NeurIPS)](https://arxiv.org/abs/2005.11401) and extends it with:

- A **five-stage retrieval pipeline** combining BM25 sparse retrieval and FAISS dense embeddings via Reciprocal Rank Fusion (RRF), cross-encoder reranking, subject/chapter-aware score boosting, and parent–child chunk expansion
- A **RAG-optimised corpus** of 14,450 enriched chunks from 99 chapters across four NCERT textbooks, with dual retrieval texts, extracted formulas, key terms, and structured problem objects
- A **573-question evaluation framework** with 95% bootstrap confidence intervals, paired McNemar significance tests, and manually categorised failure analysis
- **Comparison against 5 open-source LLM baselines** (Gemma-2-2B, Qwen2.5-3B, Llama-3.2-3B, Mistral-7B, Llama-3.1-8B)

> **Counter-intuitive finding:** On this curriculum-aligned corpus, BM25 alone outperforms dense retrieval (84.8% vs 79.4% Recall@10) — yet hybrid fusion beats both (89.5%), confirming the complementary nature of the two signals.

---

## Key Results

### Retrieval Performance (N = 573 questions, 95% bootstrap CIs)

| Configuration     | Recall@5 (%)         | Recall@10 (%)        | MRR                   |
|-------------------|----------------------|----------------------|-----------------------|
| BM25-Only         | 79.93 [76.6–82.9]   | 84.82 [81.9–87.8]   | 0.692 [0.658–0.723]  |
| Dense-Only        | 66.67 [62.8–70.5]   | 79.41 [75.9–82.7]   | 0.513 [0.477–0.547]  |
| Hybrid RRF        | 81.15 [77.7–84.5]   | 89.53 [86.9–92.0]   | 0.636 [0.604–0.670]  |
| Hybrid + Rerank   | 89.70 [87.3–92.2]   | 93.72 [91.6–95.6]   | 0.779 [0.749–0.806]  |
| **Full Pipeline** | **91.62 [89.4–93.7]** | **94.59 [92.7–96.5]** | **0.810 [0.782–0.835]** |

### LLM Baseline Comparison (N = 200, matched subset)

| System                            | Params      | BERTScore F1            | ROUGE-L               | Concept Cov. |
|-----------------------------------|-------------|-------------------------|-----------------------|--------------|
| **Full Pipeline (1.5B + RAG, ours)** | **1.5B + RAG** | **0.809 [0.805–0.813]** | **0.125 [0.116–0.134]** | **35.5%** |
| Llama-3.1-8B (zero-shot)          | 8.0B        | 0.808 [0.764–0.858]    | 0.122 [0.037–0.257]   | 39.5%        |
| Mistral-7B v0.3 (zero-shot)       | 7.2B        | 0.810 [0.762–0.852]    | 0.118 [0.039–0.219]   | 34.8%        |
| Llama-3.2-3B (zero-shot)          | 3.2B        | 0.809 [0.763–0.851]    | 0.123 [0.031–0.229]   | 37.0%        |
| Qwen2.5-3B (zero-shot)            | 3.0B        | 0.808 [0.752–0.849]    | 0.120 [0.032–0.220]   | 36.2%        |
| Gemma-2-2B (zero-shot)            | 2.6B        | 0.803 [0.770–0.841]    | 0.116 [0.045–0.218]   | 40.2%        |

> **Parameter efficiency:** Our 1.5B+RAG system matches the answer quality of 7–8B zero-shot models with **10× narrower confidence intervals**, confirming RAG's per-question consistency advantage.

### Per-Subject Recall@10 (Full Pipeline)

| Subject     | n   | Recall@10 (%)      | MRR                  |
|-------------|-----|---------------------|----------------------|
| Biology     | 179 | **96.6** [93.9–98.9] | **0.849** [0.80–0.89] |
| Chemistry   | 114 | 95.6 [91.2–99.1]   | 0.826 [0.76–0.88]    |
| Physics     | 122 | 93.4 [88.5–97.5]   | 0.808 [0.75–0.87]    |
| Mathematics | 158 | 92.4 [88.0–96.2]   | 0.756 [0.70–0.81]    |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    OFFLINE · Data & Indexing                     │
│                                                                 │
│  4 NCERT PDFs (2,409 pp)  →  v5 Preprocessor  →  14,450 chunks │
│                                    ↓                            │
│              BM25 Index (rank_bm25, 11 MB)                     │
│              Dense Index (FAISS + bge-base-en-v1.5, 44 MB)    │
└─────────────────────────────────────────────────────────────────┘
                              ↓ query time
┌─────────────────────────────────────────────────────────────────┐
│                    ONLINE · Query & Retrieval                    │
│                                                                 │
│  User Query + subject/grade/chapter hints                       │
│       ↓                                                         │
│  Stage 1 · Hybrid RRF  (BM25 + Dense → top-100 fused, 100ms)  │
│       ↓                                                         │
│  Stage 2 · Cross-Encoder Rerank  (ms-marco-MiniLM-L-6, 138ms) │
│       ↓                                                         │
│  Stage 3 · Subject + Chapter Boost  (α=2.0, β=4.0, <1ms)      │
│       ↓                                                         │
│  Stage 4 · Sibling Chunk Expansion  (parent-child, <1ms)       │
│       ↓                                                         │
│  Stage 5 · LLM Generation  (Qwen2.5-1.5B-Instruct, ~9.2s)    │
│                                                                 │
│  Total end-to-end: ~9.4 s on Colab L4 GPU                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
ncert-science-rag/
│
├── 📓 notebooks/
│   ├── RAG_Pipeline_v5_robust.ipynb        ← Main pipeline (use this)
│   ├── RAG_Pipeline_v4_fast.ipynb          ← Faster variant (lighter eval)
│   ├── RAG_Pipeline_v3_bytez.ipynb         ← v3 prototype (reference)
│   ├── LLM_Baseline_Evaluation.ipynb       ← Baseline comparison notebook
│   └── NCERT_RAG_Pipeline_v2.ipynb         ← Original v2 prototype
│
├── 📊 data/
│   ├── ncert_complete_science_v5_FINAL.json ← Full corpus (14,450 chunks, 51 MB)
│   └── ncert_eval_set.json                  ← 573-question evaluation set
│
├── 📈 results/
│   ├── full_pipeline.csv                    ← Full pipeline predictions + scores
│   ├── bm25.csv                             ← BM25-only ablation results
│   ├── dense.csv                            ← Dense-only ablation results
│   ├── rrf.csv                              ← Hybrid RRF ablation results
│   ├── rerank.csv                           ← Hybrid+Rerank ablation results
│   ├── zeroshot.csv                         ← Zero-shot LLM baseline results
│   ├── full_pipeline_latencies.csv          ← Per-query latency breakdown
│   ├── qualitative_examples.csv             ← Qualitative QA examples
│   ├── failure_samples.csv                  ← Manual failure analysis cases
│   └── baseline_results/
│       ├── baseline_gemma2_2b_scored.csv
│       ├── baseline_llama31_8b_scored.csv
│       ├── baseline_llama32_3b_scored.csv
│       ├── baseline_mistral_7b_scored.csv
│       ├── baseline_qwen25_3b_scored.csv
│       └── SUMMARY_baseline_results.csv
│
├── 🖼️ figures/                               ← All 14 publication figures (300 dpi)
│   ├── fig_architecture.png
│   ├── fig_pipeline.png
│   ├── fig_schema.png
│   ├── fig_math.png
│   ├── fig_corpus_stack.png
│   ├── fig_content_donut.png
│   ├── fig_evalset.png
│   ├── fig_ablation.png
│   ├── fig_per_subject.png
│   ├── fig_signif.png
│   ├── fig_failure_modes.png
│   ├── fig_hyperparam.png
│   ├── fig_scaling.png
│   └── fig_baselines.png
│
├── 📄 paper/
│   ├── main.tex                             ← IEEE conference LaTeX source
│   └── main.pdf                             ← Compiled paper (13 pages)
│
├── 🐍 scripts/                              ← Helper/analysis scripts
│   ├── build_data_figs.py                   ← Regenerate all data figures
│   └── build_system_figs.py                 ← Regenerate system diagrams
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## What to Upload

### ✅ Required — upload these

| File / Folder | Size | Description |
|---|---|---|
| `notebooks/RAG_Pipeline_v5_robust.ipynb` | ~96 KB | **Main pipeline notebook** — run this first |
| `notebooks/LLM_Baseline_Evaluation.ipynb` | ~40 KB | LLM baseline comparison |
| `data/ncert_eval_set.json` | 544 KB | 573-question evaluation set |
| `figures/` (all 14 PNGs) | ~3 MB | Publication figures |
| `paper/main.tex` | 65 KB | LaTeX paper source |
| `paper/main.pdf` | ~3 MB | Compiled PDF |
| `scripts/build_data_figs.py` | ~25 KB | Figure generation code |
| `scripts/build_system_figs.py` | ~20 KB | System diagram code |
| `results/SUMMARY_baseline_results.csv` | <10 KB | LLM comparison summary |
| `results/failure_samples.csv` | ~4 KB | Manual failure analysis |
| `results/qualitative_examples.csv` | ~48 KB | Example QA outputs |
| `README.md` | this file | |
| `.gitignore` | | |
| `LICENSE` | | MIT license |

### ⚠️ Large files — use Git LFS or host separately

| File | Size | Where to host |
|---|---|---|
| `data/ncert_complete_science_v5_FINAL.json` | **51 MB** | [HuggingFace Datasets](https://huggingface.co/datasets) or Google Drive |
| `embeds_49f8583183d0.npy` | **43 MB** | Google Drive (link in notebook) |
| `results/full_pipeline.csv` | 1.6 MB | Git LFS OK |
| `results/bm25.csv` / `dense.csv` / `rrf.csv` / `rerank.csv` | ~640 KB each | Git LFS OK |
| All ablation CSVs combined | ~3.5 MB | Git LFS OK |

> **Recommended:** Put the 51 MB JSON corpus and the `.npy` embeddings file on a Google Drive or HuggingFace Dataset and add the download link to the notebook's Cell 1 as `!gdown <id>`.

### ❌ Do NOT upload

| File | Reason |
|---|---|
| `embeds_49f8583183d0.npy` | 43 MB binary — too large for Git |
| `paper_final.zip` / `paper_source.zip` | Redundant — paper folder covers this |
| `RAG_Pipeline.ipynb` (v1) | Superseded by v5 |
| Screenshots | Not useful to others |
| `.env` / any file with your HF token | Never commit secrets |

---

## Setup & Installation

### Option 1 — Google Colab (recommended)

Open the main notebook directly:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/ncert-science-rag/blob/main/notebooks/RAG_Pipeline_v5_robust.ipynb)

> **Runtime:** Select `Runtime → Change runtime type → L4 GPU`

All dependencies are installed inside the notebook. No local setup needed.

### Option 2 — Local installation

```bash
git clone https://github.com/YOUR_USERNAME/ncert-science-rag.git
cd ncert-science-rag

pip install torch transformers accelerate faiss-cpu rank_bm25 \
            sentence-transformers bert-score rouge-score \
            numpy pandas matplotlib seaborn tqdm
```

**Requirements at a glance:**

| Library | Version | Role |
|---|---|---|
| `transformers` | ≥4.45 | Qwen2.5, cross-encoder |
| `faiss-cpu` / `faiss-gpu` | ≥1.7 | Dense vector index |
| `rank_bm25` | ≥0.2.2 | BM25 sparse index |
| `sentence-transformers` | ≥3.0 | BGE embeddings |
| `bert-score` | ≥0.3.13 | Generation evaluation |
| `torch` | ≥2.0 | Backend |

---

## How to Run

### Step 1 — Download the corpus

```python
# Inside the notebook (Cell 1) — downloads from Google Drive
import gdown
gdown.download(id="YOUR_GDRIVE_FILE_ID", output="ncert_complete_science_v5_FINAL.json")
```

Or place `ncert_complete_science_v5_FINAL.json` in `data/` manually.

### Step 2 — Build indices

The notebook automatically builds the BM25 and FAISS indices on first run (~2 min on L4).

```python
# Handled inside RAG_Pipeline_v5_robust.ipynb
# BM25 index:  ~11 MB RAM
# FAISS index: ~44 MB RAM  (768-dim, IndexFlatIP)
```

### Step 3 — Run evaluation

```python
# Single query
answer = query_pipeline(
    question="Explain the working principle of a p-n junction diode.",
    subject="physics",
    grade=12,
    chapter="Semiconductor Electronics"
)

# Full evaluation (573 questions, ~90 min on L4)
results = evaluate_full_pipeline(eval_set_path="data/ncert_eval_set.json")
```

### Step 4 — LLM baseline comparison

Open `notebooks/LLM_Baseline_Evaluation.ipynb` in a **fresh Colab L4 runtime**:

1. Cell 1: paste your HF token (needed for Llama / Gemma gated models)
2. Cell 3: runs the 30-second diagnostic — identifies which models load cleanly
3. Cell 4: runs all passing models on 200 questions (~30 min total)
4. Cell 5: computes BERTScore + ROUGE-L + Concept Coverage

---

## Dataset

### Corpus: `ncert_complete_science_v5_FINAL.json`

| Property | Value |
|---|---|
| Total chunks | 14,450 |
| Chapters | 99 (Physics 21, Chemistry 19, Maths 27, Biology 32) |
| Source pages | 2,409 pp across 4 NCERT textbooks (Reprint 2025–26) |
| Content types | concept, example, exercise, diagram, table, summary, points_to_ponder |
| Chunks with formulas | 4,566 (31.6%) |
| Chunks with key terms | 2,070 (14.3%) |
| Mean chunk length | 376 characters (median 247) |

Each chunk record contains:

```json
{
  "id": "physics_text_ch1_a3f2",
  "hierarchy": { "subject": "physics", "grade": 11, "chapter_number": 1, "chapter_title": "...", "section": "..." },
  "content": { "title": "...", "text": "...", "formulas": [...], "key_terms": [...] },
  "metadata": { "content_type": "concept", "difficulty": "basic", "parent_chunk_id": null },
  "rag_optimised": {
    "searchable_text": "Physics Chapter 1 ... [key terms appended for BM25]",
    "embedding_text": "[Physics Ch1 Title | Section] ..."
  },
  "problems": { "is_problem": false },
  "location": { "source_pdf": "physics_11.pdf", "page_number": 3, "bbox": [0.12, 0.45, 0.88, 0.62] }
}
```

### Evaluation Set: `ncert_eval_set.json`

| Property | Value |
|---|---|
| Total questions | 573 |
| Sources | NCERT exercises (379, 66.1%) + key-term generated (194, 33.9%) |
| Question types | definition (226), conceptual (200), numerical (78), factual (46), reasoning (23) |
| Difficulty levels | basic (288), intermediate (273), advanced (12) |

---

## Evaluation

### Retrieval metrics

- **Recall@K** — lenient: hit if any chunk in `recall_any_of` list appears in top-K
- **MRR** — Mean Reciprocal Rank

### Generation metrics

- **BERTScore F1** — contextual embedding similarity vs reference answer
- **ROUGE-L** — longest common subsequence F-measure
- **Concept Coverage** — fraction of reference `key_concepts` appearing in generated answer

### Statistical significance

All ablation comparisons use **paired McNemar tests** on Recall@10 and **paired bootstrap** (n=1,000) on MRR with 95% CIs:

| Stage transition | McNemar p | Interpretation |
|---|---|---|
| BM25 → Dense | 4.2e-3 ** | Dense is **worse** (71 wins for BM25, 40 for Dense) |
| Dense → RRF | 1.8e-12 *** | Hybrid fusion is highly significant |
| RRF → +Rerank | 2.7e-4 *** | Cross-encoder reranking is highly significant |
| +Rerank → Full | 6.3e-2 n.s. | Boost marginal at N=573 (MRR CI excludes 0) |

---

## Figures

All figures are in `figures/` at 300 dpi, generated by the scripts in `scripts/`.

| File | Content |
|---|---|
| `fig_architecture.png` | End-to-end system architecture (offline + online paths) |
| `fig_pipeline.png` | Five-stage corpus construction pipeline |
| `fig_schema.png` | Enriched chunk schema (7 nested objects) |
| `fig_math.png` | Mathematical formulation of all 5 retrieval stages |
| `fig_corpus_stack.png` | Corpus composition by subject and content type |
| `fig_content_donut.png` | Overall content-type distribution (14,450 chunks) |
| `fig_evalset.png` | Evaluation set distribution (subject/type/difficulty) |
| `fig_ablation.png` | Ablation: incremental pipeline gains with ±pp deltas |
| `fig_per_subject.png` | Per-subject Recall@10 across all 5 retrieval conditions |
| `fig_signif.png` | Paired significance tests (McNemar discordant pairs) |
| `fig_failure_modes.png` | Manual failure categorisation (31 misses) |
| `fig_hyperparam.png` | α and β sensitivity sweeps (robustness analysis) |
| `fig_scaling.png` | Corpus/eval scale vs prototype + feature comparison |
| `fig_baselines.png` | LLM baseline comparison (BERTScore + Concept Coverage) |

To regenerate figures from raw data:

```bash
python scripts/build_data_figs.py      # data-driven charts
python scripts/build_system_figs.py    # system diagrams
```

---

## Paper

The accompanying IEEE conference paper is in `paper/`. Compile with:

```bash
cd paper
pdflatex main.tex
pdflatex main.tex   # second pass for cross-refs
pdflatex main.tex   # third pass to settle floats
```

Requires **TeX Live** with IEEEtran (`texlive-publishers` package).

**Paper highlights:**
- §II — Related Work: 5 subsections covering foundational RAG (base paper: Lewis et al. 2020), dense/sparse retrieval, educational QA, evaluation methodology, and vector infrastructure
- §VII — Results: ablation, per-subject/type/difficulty breakdowns, generation quality, statistical significance, manual failure analysis, cross-subject disambiguation
- §VII-I — Open-source LLM baseline comparison (5 models, 4 families)
- §IX — Discussion: hyperparameter robustness, computational cost, dataset quality, limitations
- 32 references from the literature review

---

## Repository Name

**Recommended:** `ncert-science-rag`

**Alternatives (also good):**
- `ncert-multisubject-rag` — more descriptive
- `ncert-rag-qa` — shorter
- `NCERT-RAG-Science` — capitalised style

**GitHub URL would be:**  
`https://github.com/YOUR_USERNAME/ncert-science-rag`

---

## .gitignore

Create a `.gitignore` in the root:

```gitignore
# Large data files — host on HuggingFace / Google Drive instead
*.npy
ncert_complete_science_v5_FINAL.json

# Python
__pycache__/
*.py[cod]
*.pyo
.env
*.egg-info/
dist/
build/
.eggs/

# Jupyter checkpoints
.ipynb_checkpoints/
*.ipynb_checkpoints

# LaTeX build artifacts
paper/*.aux
paper/*.log
paper/*.out
paper/*.bbl
paper/*.blg
paper/*.synctex.gz

# OS
.DS_Store
Thumbs.db

# Secrets — NEVER commit your HF token
.env
secrets.json
hf_token.txt
```

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{ghosh2026ncertrag,
  title     = {A Multi-Subject Retrieval-Augmented Generation System for NCERT Science Question Answering Across Physics, Chemistry, Mathematics, and Biology},
  author    = {Ghosh, Aniket},
  booktitle = {Proceedings of the IEEE Conference},
  year      = {2026},
  note      = {National Institute of Technology Calicut, M240303CS}
}
```

**Base paper (Lewis et al. 2020):**

```bibtex
@inproceedings{lewis2020rag,
  title     = {Retrieval-Augmented Generation for Knowledge-Intensive {NLP} Tasks},
  author    = {Lewis, Patrick and Perez, Ethan and Piktus, Aleksandra and others},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {33},
  pages     = {9459--9474},
  year      = {2020}
}
```

---

## Acknowledgements

NCERT textbook content is published by the National Council of Educational Research and Training, Government of India. This work was conducted as part of the M.Tech. programme at NIT Calicut (Roll: M240303CS).

---

<div align="center">

**Made with ❤️ at NIT Calicut**  
*Aniket Ghosh · M.Tech Computer Science · 2024–2026*

</div>
