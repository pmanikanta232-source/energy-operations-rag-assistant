
from pathlib import Path
import re
from typing import List, Dict, Tuple

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Paths and application setup
BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
FRONTEND_DIR = BASE / "frontend"

app = FastAPI(
    title="Energy Operations Document RAG Assistant",
    version="1.0.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    text: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=5)


# 1. Load documents and split into chunks
def load_documents() -> List[Dict]:
    documents = []

    if not DATA_DIR.exists():
        return documents

    for path in sorted(DATA_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8")

        # Split documents into paragraphs
        paragraphs = [
            p.strip()
            for p in re.split(r"\n\s*\n", raw)
            if p.strip()
        ]

        for paragraph in paragraphs:
            paragraph = re.sub(
                r"(?m)^#+\s*", "", paragraph
            ).strip()

            if not paragraph:
                continue

            # Split long paragraphs into sentences
            pieces = (
                re.split(r"(?<=[.!?])\s+", paragraph)
                if len(paragraph) > 450
                else [paragraph]
            )

            for piece in pieces:
                piece = piece.strip()

                if piece:
                    documents.append({
                        "source": path.name,
                        "text": piece
                    })

    return documents


chunks = load_documents()


# 2. Create TF-IDF vectors
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True
)

if chunks:
    document_matrix = vectorizer.fit_transform(
        [chunk["text"] for chunk in chunks]
    )
else:
    document_matrix = None


# 3. Retrieve relevant document chunks
MIN_SIMILARITY = 0.12


def retrieve(question: str, top_k: int = 3):
    if not chunks or document_matrix is None:
        return []

    query_vector = vectorizer.transform([question])

    if query_vector.nnz == 0:
        return []

    scores = cosine_similarity(
        query_vector, document_matrix
    ).flatten()

    indices = np.argsort(scores)[::-1]

    results = []

    for i in indices:
        score = float(scores[i])

        if score < MIN_SIMILARITY:
            continue

        results.append({
            **chunks[i],
            "score": score
        })

        if len(results) >= top_k:
            break

    return results


# 4. Generate answer from retrieved documents
def generate_answer(question: str, results: List[Dict]):

    if not results:
        return (
            "I cannot find supporting information in the "
            "available documents. Please consult the relevant "
            "operations team or provide a supporting document."
        )

    sentences = []
    seen = set()

    for item in results:
        sentence = item["text"].strip()

        if sentence and sentence.lower() not in seen:
            sentences.append(sentence)
            seen.add(sentence.lower())

        if len(sentences) == 2:
            break

    return " ".join(sentences)


# 5. Check answer evidence
def hallucination_check(answer: str, results: List[Dict]):

    if not results or answer.startswith("I cannot"):
        return {
            "passed": False,
            "status": "abstained",
            "explanation": "No sufficiently relevant evidence was found."
        }

    evidence = " ".join(
        item["text"] for item in results
    ).lower()

    answer_words = set(
        re.findall(r"[a-zA-Z0-9]+", answer.lower())
    )

    evidence_words = set(
        re.findall(r"[a-zA-Z0-9]+", evidence)
    )

    meaningful_words = {
        word for word in answer_words if len(word) > 2
    }

    overlap = (
        len(meaningful_words & evidence_words)
        / max(1, len(meaningful_words))
    )

    passed = overlap >= 0.55

    return {
        "passed": passed,
        "status": "supported" if passed else "review_required",
        "explanation": (
            "Answer wording was checked against retrieved evidence."
            if passed
            else "Some answer wording is not sufficiently supported."
        )
    }


# 6. Backend API
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "documents": len({c["source"] for c in chunks}),
        "chunks": len(chunks),
        "retriever": "TF-IDF + cosine similarity"
    }


@app.get("/api/demo")
def demo():
    return {
        "case_study": "Energy Operations Document RAG Assistant",
        "status": "running",
        "documents_loaded": len({c["source"] for c in chunks}),
        "chunks_created": len(chunks)
    }


@app.post("/api/run")
def run_case(query: Query):

    results = retrieve(query.text, query.top_k)

    answer = generate_answer(query.text, results)

    check = hallucination_check(answer, results)

    citations = list({
        item["source"] for item in results
    }) if results else []

    return {
        "question": query.text,
        "answer": answer,
        "citations": citations,
        "retrieved_evidence": results,
        "hallucination_check": check,
        "retrieval_method": "TF-IDF + cosine similarity",
        "limitations": [
            "TF-IDF is a lexical retrieval baseline.",
            "Answer generation is extractive.",
            "No external LLM is connected."
        ]
    }


# 7. Serve frontend
app.mount(
    "/",
    StaticFiles(directory=FRONTEND_DIR, html=True),
    name="frontend"
)
