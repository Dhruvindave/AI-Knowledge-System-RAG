from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def chunk_encoder(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = model.encode(chunks)

    print(f"Each embedding has {embeddings.shape[1]} dimensions")

    similarities = cosine_similarity(embeddings)
    print("Similarity matrix:")
    print(similarities)

    return embeddings
# You'll see sentences 1, 2, and 4 are more similar to each other
