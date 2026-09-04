# Agentic RAG Proof-of-Concept 

An enterprise-grade, local-first **Agentic Retrieval-Augmented Generation (RAG)** system built with **FastAPI**, **LangGraph**, **ChromaDB**, and local LLM orchestration via **Ollama**.

Unlike naive single-pass RAG pipelines that fail silently on poor context, this engine implements an autonomous state-machine feedback loop: evaluating retrieved context relevance, self-correcting suboptimal queries, and grounding generation to reduce hallucinations.

---

## Architecture Overview

```
                      ┌──────────────────────────┐
                      │   User Query (FastAPI)   │
                      └─────────────┬────────────┘
                                    │
                                    ▼
                      ┌──────────────────────────┐
                 ┌───►│      retrieve_node       │
                 │    │   (ChromaDB Semantic)    │
                 │    └─────────────┬────────────┘
                 │                  │
                 │                  ▼
                 │    ┌──────────────────────────┐
                 │    │  evaluate_context_node   │
                 │    │   (LLM-as-a-Judge: Y/N)  │
                 │    └─────────────┬────────────┘
                 │                  │
     Loop < 3 & "No"                ├──────────────────────┐
                 │                  │                      │
                 ▼                  ▼ "Yes"                ▼ Loop >= 3
       ┌───────────────────┐ ┌──────────────┐    ┌───────────────────┐
       │rewrite_query_node │ │generate_node │    │   generate_node   │
       │(Query Optimizer)  │ │ (Synthesizer)│    │(Grounded Refusal) │
       └─────────┬─────────┘ └──────┬───────┘    └─────────┬─────────┘
                 │                  │                      │
                 └──────────────────┴──────────┬───────────┘
                                               │
                                               ▼
                                    ┌─────────────────────┐
                                    │ JSON Response + Meta│
                                    └─────────────────────┘
```

### Key Engineering Pillars

1. **Autonomous Evaluation Loop** — an isolated, deterministic (`temperature=0`) LLM judge verifies that retrieved context actually contains the answer before synthesis begins.
2. **Query Self-Correction** — when retrieval is irrelevant or insufficient, queries are rewritten into cleaner search syntax, with a bounded retry loop (`N ≤ 3`) to avoid infinite retrieval traps.
3. **Strict Fact Grounding** — when relevant context doesn't exist in the vector index, the synthesizer is constrained toward explicit refusals rather than inventing an answer.
4. **Hardware-Agnostic Acceleration** — automatically detects and routes tensor computations to Apple Silicon (`mps`), Nvidia CUDA (`cuda`), or CPU fallback.
5. **Production Observability & CI/CD** — HTTP middleware logs per-request latency (`X-Process-Time`), validated through GitHub Actions CI and automated `pytest` suites.

---

## Tech Stack

| Domain | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend Framework** | FastAPI | Asynchronous REST API layer |
| **Orchestration** | LangGraph / LangChain | State machine graph & agent routing |
| **Local LLM Engine** | Ollama (`llama3`) | Local inference & grading |
| **Vector Database** | ChromaDB | Persistent local semantic vector storage |
| **Embeddings** | HuggingFace (`all-MiniLM-L6-v2`) | 384-dimensional dense vector embeddings |
| **Testing & CI/CD** | Pytest, GitHub Actions | Automated linting, test collection, & regression |
| **Frontend** | React, TypeScript, Vite | Dual-pane ingestion and telemetry chat UI |

---

## Repository Structure

```text
├── app/
│   ├── api/
│   │   ├── documents.py        # Document ingestion (PDF upload & chunking)
│   │   └── query.py            # Agentic query execution endpoint
│   ├── core/
│   │   └── config.py           # Dynamic hardware detection & Pydantic settings
│   ├── rag/
│   │   ├── ingestor.py         # Recursive text splitting & Chroma embedding
│   │   └── retriever.py        # Semantic vector similarity search
│   ├── services/
│   │   └── agent.py            # LangGraph StateGraph, nodes, and routing edges
│   └── main.py                 # FastAPI initialization, CORS, & latency middleware
├── data/
│   ├── eval_dataset.json       # Ground-truth evaluation benchmarks
│   └── sample.pdf              # Reference document used for smoke testing
├── frontend/                   # React + TypeScript control center
├── tests/
│   └── test_api.py             # Pytest endpoint validation & chunking tests
├── .github/workflows/ci.yml    # GitHub Actions automated test workflow
├── evaluate.py                 # LLM-as-a-Judge smoke-test evaluation harness
├── stress_test.py              # Adversarial, 25-question domain stress test
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Container orchestration with host networking
└── requirements.txt            # Minimal top-level dependencies
```

---

##  Quickstart

### Prerequisites

* Python 3.10+
* [Ollama](https://ollama.com/) installed, with the model pulled and running locally:

```bash
ollama run llama3
```

### 1. Local Setup

```bash
git clone https://github.com/Rashid-hussain-mohammed/Agentic-RAG-Proof-of-Concept.git
cd Agentic-RAG-Proof-of-Concept

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the Backend

```bash
uvicorn app.main:app --reload --port 8000
```

Interactive API docs are available at `http://localhost:8000/docs`.

### 3. Run the Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` to access the Control Center UI.

---

##  Evaluation

### Smoke Test (`evaluate.py`)

A small, fixed set of ground-truth questions used to sanity-check the pipeline end-to-end during development. Ollama must be running before this is invoked.

```bash
python evaluate.py
```

```text
--- STARTING EVALUATION ---
Evaluating Question 1: What is the main design goal for Hadoop 1.0 release?
 Result: PASS

Evaluating Question 2: Why do developers create a sampled subset of production data?
 Result: PASS (Query rewritten on Loop 1 -> Context matched on Loop 2)

Evaluating Question 3: Where can you get the patent data sets mentioned for the examples?
 Result: PASS

--- EVALUATION COMPLETE ---
Final Score: 3 / 3 (100% Accuracy)
```

### Adversarial Stress Test (`stress_test.py`)

A 25-question battery run against dense, real-world documents (e.g., SEC 10-K filings), designed to probe failure modes a small smoke test won't catch:

* **Factual & table extraction** — precision retrieving numerical data across columns.
* **Multi-hop reasoning** — synthesizing answers across disparate chunks and sections.
* **Negation & cross-referencing** — filtering false premises, checking data consistency.
* **Hallucination traps** — out-of-scope questions (e.g., forward guidance, unlisted stock prices) that the agent should refuse rather than answer.

```bash
pip install fpdf2 tabulate
python stress_test.py data/apple_10k.pdf
```

The script streams a live execution table to the terminal and generates a detailed `stress_test_report.pdf` with per-question latency, retry-loop counts, and verdicts.

**Latest run — Apple FY23 10-K:** 22/25 (88%) passed. The system correctly refused clearly out-of-scope queries (e.g., forward-looking guidance, unlisted regional store counts) rather than fabricating an answer. One hallucination was identified — a stock-price question the evaluator judged too leniently — and is being addressed by tightening the context-evaluator prompt to require exact entity/metric matching rather than topical similarity.