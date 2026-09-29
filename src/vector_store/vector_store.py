# import faiss
# import numpy as np


# class FaissVectorStore:
#     """
#     Object-Oriented wrapper for Faiss vector storage and similarity search.
#     Supports adding vectors, searching, saving, and loading indexes.
#     """

#     def __init__(self, dimension):
#         self.dimension = dimension

#         # Initialize FAISS Index
#         self.index = faiss.IndexFlatL2(dimension)

#     def add(self, embeddings):
#         embeddings = np.asarray(
#             embeddings,
#             dtype='float32'
#         )

#         self.index.add(embeddings)

#     def search(self, query_embedding, k=3):
#         query_embedding = np.asarray(
#             query_embedding,
#             dtype='float32'
#         )

#         distances, indices = self.index.search(query_embedding, k=k)

#         return distances, indices

#     def total_vectors(self):
#         return self.index.ntotal




import faiss
import numpy as np
from pathlib import Path
import pickle

class FaissVectorStore:
    """
    Object-Oriented wrapper for Faiss vector storage and similarity search.
    Bundles vectors (in FAISS) with their metadata (in a parallel dict),
    keyed by the same integer position so the two never drift out of sync.
    """

    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)

        # position (int) -> metadata dict for that chunk
        self.metadata_store = {}

        # tracks the next free position; always equals self.index.ntotal,
        # kept explicit for clarity rather than relying on ntotal everywhere
        self._next_id = 0

    def add(self, embeddings, metadata_list):
        """
        embeddings: array-like, shape (n, dimension)
        metadata_list: list of dicts, length n, e.g.
            {"title": ..., "breadcrumb": ..., "content": ..., "parent_id": ..., "child_ids": [...]}
        Both must be the same length and in the same order.
        """
        embeddings = np.asarray(embeddings, dtype='float32')

        if len(embeddings) != len(metadata_list):
            raise ValueError(
                f"embeddings ({len(embeddings)}) and metadata_list "
                f"({len(metadata_list)}) must be the same length"
            )

        self.index.add(embeddings)

        for meta in metadata_list:
            meta = meta.copy()

            meta["chunk_id"] = str(self._next_id)

            self.metadata_store[self._next_id] = meta

            self._next_id += 1

    def search(self, query_embedding, k=3):
        query_embedding = np.asarray(query_embedding, dtype='float32')
        distances, indices = self.index.search(query_embedding, k=k)

        # join distances/indices back to metadata immediately,
        # so callers never have to touch raw FAISS positions
        results = []
        for dist_row, idx_row in zip(distances, indices):
            row_results = []
            for dist, idx in zip(dist_row, idx_row):
                if idx == -1:
                    continue  # FAISS pads with -1 if fewer than k results exist
                entry = self.metadata_store[idx].copy()
                entry["distance"] = float(dist)
                entry["_faiss_id"] = int(idx)
                row_results.append(entry)
            results.append(row_results)

        return results

    def total_vectors(self):
        return self.index.ntotal

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(path / "faiss.index")
        )

        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata_store, f)

    @classmethod
    def load(cls, path):

        path = Path(path)

        index = faiss.read_index(
            str(path / "faiss.index")
        )

        with open(
            path / "metadata.pkl",
            "rb"
        ) as f:

            metadata_store = pickle.load(f)

        store = cls(index.d)

        store.index = index
        store.metadata_store = metadata_store
        store._next_id = index.ntotal

        return store