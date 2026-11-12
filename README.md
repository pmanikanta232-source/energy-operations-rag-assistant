# Energy Operations Document RAG Assistant

A self-contained RAG demonstration using FastAPI, TF-IDF vector representations, cosine-similarity retrieval, extractive answer generation, source citations, and a basic evidence-support check.

## Requirements
- Python 3.10 or 3.11
- pip

## Run locally (Windows PowerShell)
```powershell
cd energy_rag_completed
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn backend.app:app --reload
```
Open http://127.0.0.1:8000. API docs: http://127.0.0.1:8000/docs.

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again. Or use `.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.venv\Scripts\python.exe -m uvicorn backend.app:app --reload` without activating.

## Pipeline
1. Load `.md` files from `data/`.
2. Split paragraphs / long paragraphs into chunks while retaining source filenames.
3. Fit a TF-IDF vectorizer (unigrams and bigrams) to represent chunks numerically.
4. Vectorize each query and rank chunks using cosine similarity.
5. Generate an extractive answer using sentences from retrieved chunks only.
6. Check answer wording against retrieved evidence and abstain when retrieval is weak.
7. Return answer, source citations, evidence, similarity scores, check status, and limitations.

## API
- `GET /api/health` — health and corpus counts.
- `GET /api/demo` — pipeline information.
- `POST /api/run` — JSON body: `{"text":"What should be checked for inverter derating?","top_k":3}`.
- `GET /docs` — interactive API documentation.

## Evaluation
Run the sample questions manually in the UI or through `/api/run`. Check that the inverter question cites `solar_inverter_sop.md` and gearbox question cites `wind_turbine_maintenance.md`. Also test an unrelated question; the assistant should abstain rather than invent facts.

## Design choices and limitations
TF-IDF is used as an offline, reproducible baseline embedding representation, so the app runs without API keys or downloading a neural model. It is lexical rather than deeply semantic. Answer generation is extractive, not an LLM. The support check is a lightweight lexical overlap heuristic, not a formal factuality proof. The provided corpus is tiny dummy data; add domain-approved manuals, metadata, access controls, stronger semantic embeddings (e.g. sentence-transformers), a reranker, and human-reviewed evaluation before production use. Never use this demo as the sole basis for safety-critical operational decisions.

## Railway deployment
1. Push this folder to a GitHub repository.
2. In Railway, create a new project and deploy from that GitHub repository.
3. Railway detects the `Procfile`; ensure the service uses the included start command: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`.
4. Wait for build and deployment, then open the generated public domain.
5. Verify `/api/health`, the UI, and `/docs` on the deployed domain.

No API keys are required for this baseline.
