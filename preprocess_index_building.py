import pymupdf as fitz
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer

import faiss
import numpy as np

from pathlib import Path

import pickle

def extract_text(pdf_path):
    doc=fitz.open(pdf_path)
    text=''
    for page in doc:
        text+=page.get_text()
    return text

all_docs=[]
data_folder="data"
for file in os.listdir(data_folder):
    if file.endswith(".pdf"):
        file_path=os.path.join(data_folder,file)
        text=extract_text(file_path)
        all_docs.append({
            'source':file,
            'text':text
        })
print(f"Length of Documents: {len(all_docs)}")

splitter=RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200
)

all_chunks = []

for doc in all_docs:

    chunks = splitter.split_text(
        doc["text"]
    )

    for chunk in chunks:

        all_chunks.append(
            {
                "source": doc["source"],
                "text": chunk
            }
        )

embed_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

texts = [
    chunk["text"]
    for chunk in all_chunks
]

embeddings = embed_model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(
    dimension
)
index.add(
    np.array(embeddings).astype("float32")
)

save_dir = Path("/content/drive/MyDrive/vector_store")
save_dir.mkdir(parents=True, exist_ok=True)

faiss.write_index(
    index,
    str(save_dir / "naval_index.faiss")
)

with open(
    "/content/drive/MyDrive/vector_store/metadata.pkl",
    "wb"
) as f:

    pickle.dump(
        all_chunks,
        f
    )