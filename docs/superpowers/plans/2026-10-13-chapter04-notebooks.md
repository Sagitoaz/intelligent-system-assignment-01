# Chapter 4 executable notebooks implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline. User explicitly requested autonomous implementation; do not pause for plan approval.

**Goal:** Deliver executed notebooks, downloaded datasets, working KG/RAG/Agent code and a regenerated Vietnamese DOCX with code and actual results.

**Architecture:** Four notebooks share small reusable modules under src/chapter04. Public UCI/NIDDK/FinQA snapshots already downloaded are reused. Real pretrained CPU models provide dense retrieval and generation; a deterministic tool router exposes KG lookup, safe arithmetic and RAG through FastAPI. Notebook outputs and JSON results are the evidence for the DOCX.

**Tech Stack:** Python 3.12 in existing .venv; NetworkX, pandas, scikit-learn, PyTorch/Transformers, nbclient, FastAPI, python-docx.

**Spec:** User requests notebooks and datasets first, then report including executed code, metrics and tables; no PDF; no repeated permission questions.

## Global constraints

- Preserve unrelated repository work; put new work in chapter04 directories.
- Run every notebook from a clean kernel, retain all outputs and no error cells.
- Never index FinQA answers, programs or gold evidence; use those only for evaluation.
- Separate oracle report filtering from automatic selection, and retrieval from generation metrics.
- Use actual pretrained model generation, no placeholder model responses.
- Report local API deployment accurately; no claim of cloud deployment.

## Review focus

- Financial arithmetic accepts only allowlisted operations and rejects zero division and non-finite numbers.
- Domain selection and report filters prevent unrelated evidence from being returned.
- Graph entries and citations retain source locators and consistent IDs.
- Model downloads record exact revisions and SHA-256; notebook runs load local weights.
- Refusal and generation outputs are stored verbatim, including failures.

## Tasks

- [x] 1. Behavioral tests and modules; expanded suite has 12 passing tests after review fixes.
- [x] 2. Download pinned MiniLM and FLAN-T5-small; smoke real generation succeeded.
- [x] 3. Four clean executed notebooks; finance corrected-scoring rerun completed. 883 dev queries, 8 configurations, 72 financial generation outputs, 24 gold-audit records; diabetes rerun and real HTTP Agent verified.
- [x] 4. DOCX rebuilt from executed results: 17 code blocks, 21 tables, 8 figures. Old unimplemented descriptions replaced with actual scope.
- [x] 5. Artifact verification passed: dataset/model hashes, all notebook hashes/output counts, API requests and DOCX structure. Notebook/code/report ZIP and dataset ZIP prepared.

## Execution ledger

- Starting state: first report has 10 tables and 4 figures, but no notebook. Cached datasets occupy 126 MB. .venv already has Jupyter, PyTorch and FastAPI.
- Ruling: work in dedicated chapter04 paths in current checkout because worktree isolation would require copying large cached data and disturb user's existing modified files; no commits/pushes.
- Fixed notebook generator newline escaping after a failing AST test.
- Fixed long-context prompt truncation: question now precedes context; citation labels kept outside generation context. Diabetes notebook rerun: hybrid Hit@5 87.5%, EM 31.25%, token F1 54.66% on 16 authored questions.
- Independent review found percentage scoring and statistics-routing defects. Fixed with unit-aware qa.answer matching, separate gold annotation audit, supported count intents and shared resolve_task. Reviewer confirms 12 tests pass and no important unresolved regression.
- Local HTTP deployment verified health/Swagger, four success cases, zero-division and invalid-domain rejection. Model/citation semantic correctness remains explicitly unverified.
- Final measured financial generation answer-matching is 0/24 for each of the three small-model configurations; report preserves these failures and separates the successful manual-mapped calculator example (127.4). Six gold-audit records need review. No fabricated success numbers.
