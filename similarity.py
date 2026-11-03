
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load a pretrained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def compare_papers(texts):
    """
    Compare research papers using semantic similarity.

    texts: list of strings, one string per paper.
    Returns similarity scores between 0 and 1.
    """
    if len(texts) < 2:
        raise ValueError("At least two papers are required.")

    embeddings = model.encode(texts)

    scores = cosine_similarity(embeddings)

    return scores
