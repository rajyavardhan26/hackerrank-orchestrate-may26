# Support Triage Agent

A terminal-based AI agent that triages support tickets across **HackerRank**, **Claude**, and **Visa** using only the provided support corpus.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Corpus Loader  │────▶│ Embedding Store  │────▶│  Vector Search  │
│  (chunk docs)   │     │(sentence-transformers) │  (cosine sim)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
┌─────────────────┐     ┌──────────────────┐              │
│  Safety Layer   │◀────│  Triage Agent    │◀─────────────┘
│ (escalation)    │     │ (orchestrator)   │
└─────────────────┘     └──────────────────┘
                               │
                        ┌──────┴──────┐
                        ▼             ▼
                  ┌─────────┐   ┌──────────┐
                  │Classifier│   │ LLM Client│
                  │(heuristics)│  │(OpenAI/Anthropic)
                  └─────────┘   └──────────┘
```

## Pipeline

1. **Company Detection** — keyword-based scoring + explicit field
2. **Request Classification** — heuristic classifier (`product_issue`, `feature_request`, `bug`, `invalid`)
3. **Risk Assessment** — keyword + pattern-based risk score
4. **Document Retrieval** — local embeddings + cosine similarity, filtered by company
5. **LLM Reasoning** — structured JSON generation grounded in retrieved excerpts
6. **Safety Override** — deterministic escalation rules override LLM when needed

## Installation

```bash
cd code/
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and fill in your API key:

```bash
cp .env.example .env
```

Edit `.env`:
```
OPENAI_API_KEY=sk-...
# or
ANTHROPIC_API_KEY=sk-ant-...

LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
```

## Running

### Full run (production)
```bash
python code/main.py
```

### Quick validation on sample data
```bash
python code/main.py --sample
```

### Custom paths
```bash
python code/main.py --input support_tickets/support_tickets.csv --output support_tickets/output.csv
```

### Limit to first N tickets (for testing)
```bash
python code/main.py --limit 5
```

## Output Format

`support_tickets/output.csv` contains:

| Column | Description |
|--------|-------------|
| `status` | `replied` or `escalated` |
| `product_area` | Most relevant support category |
| `response` | User-facing answer grounded in corpus |
| `justification` | Concise reasoning trace |
| `request_type` | `product_issue`, `feature_request`, `bug`, `invalid` |

## Design Decisions

- **Local embeddings** (`all-MiniLM-L6-v2`) ensure fast, deterministic retrieval without API dependency.
- **Cached index** speeds up repeated runs.
- **Dual safety**: heuristic risk scoring + LLM reasoning + deterministic override.
- **Graceful degradation**: LLM failures automatically escalate rather than hallucinate.
- **Deterministic**: seeded where possible; same input → same retrieval → same output (given fixed LLM temperature).

## Dependencies

- `sentence-transformers` — local embeddings
- `scikit-learn` — cosine similarity
- `openai` / `anthropic` — LLM clients
- `rich` — terminal UI
- `pandas`, `numpy` — data handling

## Notes

- The agent relies **only** on the provided `data/` corpus. No live web calls are made for ground-truth answers.
- Secrets are read exclusively from environment variables.
- The embedding cache is stored in `code/.cache/` and is gitignored.
