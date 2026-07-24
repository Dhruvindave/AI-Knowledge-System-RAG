import faiss
import numpy as np


class FaissVectorStore:
    """
    Object-Oriented wrapper for Faiss vector storage and similarity search.
    Supports adding vectors, searching, saving, and loading indexes.
    """

    def __init__(self, dimension):
        self.dimension = dimension

        # Initialize FAISS Index
        self.index = faiss.IndexFlatL2(dimension)

    def add(self, embeddings):
        embeddings = np.asarray(
            embeddings,
            dtype='float32'
        )

        self.index.add(embeddings)

    def search(self, query_embedding, k=3):
        query_embedding = np.asarray(
            query_embedding,
            dtype='float32'
        )

        distances, indices = self.index.search(query_embedding, k=k)

        return distances, indices

    def total_vectors(self):
        return self.index.ntotal
