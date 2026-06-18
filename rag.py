import pickle
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

from config import *

index = faiss.read_index(FAISS_PATH)

with open(METADATA_PATH,"rb") as f:
    metadata = pickle.load(f)

embed_model = SentenceTransformer(
    EMBED_MODEL,
    device="cuda"
)


def retrieve(query,k=TOP_K):

    embedding = embed_model.encode(
        [query],
        normalize_embeddings=True
    )

    D,I = index.search(
        np.asarray(embedding).astype("float32"),
        k
    )

    return [metadata[i] for i in I[0]]


def build_prompt(question,retrieved):

    context=""

    for i,item in enumerate(retrieved):

        context += f"""
Reference {i+1}

Source:
{item['source']}

Content:
{item['text']}

-----------------------------------
"""

    return f"""
You are an experienced Naval Architect.

Answer ONLY from the supplied references.

If the references don't contain the answer,
say so.

References

{context}

Question

{question}

Answer:
"""