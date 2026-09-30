from sentence_transformers import SentenceTransformer
from src.config.config import EMBEDDING_MODEL
_MODEL = None


def _embedding_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _MODEL


def encode_texts(texts):
    """Encode a batch of strings; loads the embedding model once."""
    return _embedding_model().encode(texts)


def chunk_encoder(chunks):
    return encode_texts(chunks)
