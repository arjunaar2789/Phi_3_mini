# Phi-3 Mini Fine-Tuned for Naval Architecture using RAG + LoRA

An AI assistant specialized in Naval Architecture and Ship Structural Analysis.

The assistant combines Retrieval-Augmented Generation (RAG) with LoRA fine-tuning on Microsoft Phi-3 Mini to answer engineering questions from ship design and structural analysis.

---

## Features

- 📚 Multi-book Retrieval-Augmented Generation (RAG)
- 🧠 LoRA Fine-tuning of Phi-3 Mini
- 🔍 FAISS Vector Search
- 📖 Semantic Retrieval using BAAI/bge-small-en-v1.5
- 💬 Streamlit Web Interface
- ⚡ GPU Inference Support
- 📄 Synthetic QA Dataset Generation

---

## Project Architecture

```
PDF Books
      │
      ▼
Text Extraction
      │
      ▼
Chunking
      │
      ▼
BGE Embeddings
      │
      ▼
FAISS Vector Store
      │
      ▼
Retrieve Top-k Chunks
      │
      ▼
Prompt Construction
      │
      ▼
Phi-3 Mini + LoRA
      │
      ▼
Engineering Answer
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Microsoft Phi-3 Mini 4K Instruct |
| Fine-tuning | PEFT LoRA |
| Retrieval | FAISS |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Frontend | Streamlit |
| Framework | PyTorch |
| Dataset | Synthetic QA generated from Naval Architecture textbooks |

---

## Dataset

- 6 Naval Architecture textbooks
- ~19,000 semantic chunks
- ~700–1000 synthetic QA pairs
- JSONL instruction tuning format

---

## Training

- LoRA Rank: 8
- Alpha: 16
- Epochs: 3
- Optimizer: AdamW
- Precision: FP16
- GPU: NVIDIA Tesla T4

---

## Results

- Domain-specific answers
- Context-aware retrieval
- Reduced hallucinations using RAG
- Lightweight LoRA adapter (<100 MB)

---

## Repository Structure

```
src/
app.py
vector_store/
adapters/
requirements.txt
README.md
```

---

## Future Improvements

- Hybrid Search (BM25 + Dense Retrieval)
- Cross-Encoder Re-ranking
- Quantized GGUF Deployment
- Docker Support
- Evaluation using RAGAS

## Author

Arjun A A R
