"""
evaluate_rag.py
================
Computes MRR (Mean Reciprocal Rank) and Hit Rate@k for a RAG retriever against
test_dataset.json (100 hand-crafted questions grounded in metadata_store.json).

USAGE
-----
1. Plug in your own retriever by implementing a function with the signature:

       def my_retriever(query: str, top_k: int) -> list[str]:
           # return a list of chunk_id strings, best match first
           ...

   and pass it to `evaluate(my_retriever)`.

2. Or just run this file directly to see the built-in TF-IDF baseline's scores:

       python3 evaluate_rag.py

METRICS
-------
- Hit Rate@k : fraction of questions where AT LEAST ONE relevant chunk appears
               in the retriever's top-k results.
- MRR        : for each question, 1 / (rank of the first relevant chunk found),
               0 if none of the relevant chunks appear in the retrieved list.
               Averaged across all questions. Uses top-k as the search depth.

Both metrics are also broken down by question `type` and `difficulty` so you
can see exactly which kind of query trips up your retriever (e.g. numerical
lookalikes, negations, multi-hop).
"""
import json
from collections import defaultdict
from pathlib import Path

_JSON_DIR = Path(__file__).resolve().parent
DATASET_PATH = _JSON_DIR / "test_dataset.json"
CORPUS_PATH = _JSON_DIR / "metadata.json"


def load_dataset(path=None):
    path = Path(path) if path else DATASET_PATH
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_corpus(path=None):
    path = Path(path) if path else CORPUS_PATH
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def reciprocal_rank(retrieved_ids, relevant_ids):
    """1/rank of first relevant id found in retrieved_ids (1-indexed); 0 if none found."""
    relevant_set = set(relevant_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_set:
            return 1.0 / rank
    return 0.0


def hit(retrieved_ids, relevant_ids):
    relevant_set = set(relevant_ids)
    return 1.0 if any(cid in relevant_set for cid in retrieved_ids) else 0.0


def evaluate(retriever_fn, top_k=5, dataset=None, verbose=True):
    """
    retriever_fn(query: str, top_k: int) -> list[str] of chunk_id strings, ranked best-first.
    Returns a dict of overall + breakdown metrics.
    """
    dataset = dataset or load_dataset()

    per_question = []
    for q in dataset:
        relevant_ids = [q["chunk_id"]] + q.get("additional_chunk_ids", [])
        retrieved = retriever_fn(q["question"], top_k)
        rr = reciprocal_rank(retrieved, relevant_ids)
        h = hit(retrieved, relevant_ids)
        per_question.append({
            "id": q["id"], "chapter": q["chapter"], "difficulty": q["difficulty"],
            "type": q["type"], "rr": rr, "hit": h, "retrieved": retrieved,
            "relevant": relevant_ids,
        })

    def summarize(rows):
        n = len(rows)
        if n == 0:
            return {"n": 0, "mrr": None, "hit_rate": None}
        return {
            "n": n,
            "mrr": round(sum(r["rr"] for r in rows) / n, 4),
            "hit_rate": round(sum(r["hit"] for r in rows) / n, 4),
        }

    overall = summarize(per_question)

    by_type = defaultdict(list)
    by_difficulty = defaultdict(list)
    by_chapter = defaultdict(list)
    for r in per_question:
        by_type[r["type"]].append(r)
        by_difficulty[r["difficulty"]].append(r)
        by_chapter[r["chapter"]].append(r)

    results = {
        "top_k": top_k,
        "overall": overall,
        "by_type": {k: summarize(v) for k, v in sorted(by_type.items())},
        "by_difficulty": {k: summarize(v) for k, v in sorted(by_difficulty.items())},
        "by_chapter": {k: summarize(v) for k, v in sorted(by_chapter.items())},
        "per_question": per_question,
    }

    if verbose:
        print(f"\n=== RAG Retrieval Evaluation (top_k={top_k}) ===")
        print(f"Overall  -> MRR: {overall['mrr']:.4f}   Hit Rate@{top_k}: {overall['hit_rate']:.4f}   (n={overall['n']})")
        print("\nBy difficulty:")
        for k, v in results["by_difficulty"].items():
            print(f"  {k:8s} -> MRR: {v['mrr']:.4f}   Hit Rate@{top_k}: {v['hit_rate']:.4f}   (n={v['n']})")
        print("\nBy question type:")
        for k, v in results["by_type"].items():
            print(f"  {k:22s} -> MRR: {v['mrr']:.4f}   Hit Rate@{top_k}: {v['hit_rate']:.4f}   (n={v['n']})")
        print("\nBy chapter:")
        for k, v in results["by_chapter"].items():
            print(f"  {k:8s} -> MRR: {v['mrr']:.4f}   Hit Rate@{top_k}: {v['hit_rate']:.4f}   (n={v['n']})")

    return results


def build_faiss_retriever(vector_store, encode_fn):
    """
    Wrap FaissVectorStore + query encoder for use with evaluate().

    encode_fn: callable(list[str]) -> array-like of shape (n, dimension)
    """

    def retriever(query: str, top_k: int = 5) -> list[str]:
        query_vector = encode_fn([query])
        rows = vector_store.search(query_vector, k=top_k)
        if not rows:
            return []
        return [str(hit["chunk_id"]) for hit in rows[0]]

    return retriever


# ----------------------------------------------------------------------------
# Baseline retriever (TF-IDF over the chunk store) so this script runs
# end-to-end out of the box. Swap this out for your real embedding-based
# retriever when evaluating your actual RAG pipeline.
# ----------------------------------------------------------------------------
def build_tfidf_retriever(corpus=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    corpus = corpus or load_corpus()
    ids = list(corpus.keys())
    texts = [corpus[i].get("title", "") + " \n " + corpus[i].get("content", "") for i in ids]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)

    def retriever(query, top_k=5):
        qv = vectorizer.transform([query])
        sims = cosine_similarity(qv, matrix)[0]
        top_idx = sims.argsort()[::-1][:top_k]
        return [ids[i] for i in top_idx]

    return retriever


if __name__ == "__main__":
    tfidf_retriever = build_tfidf_retriever()
    for k in (1, 3, 5, 10):
        evaluate(tfidf_retriever, top_k=k)
