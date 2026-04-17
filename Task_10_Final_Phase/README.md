# Task 10: Context-Aware Chatbot Using RAG

This task builds a conversational chatbot that:

- remembers recent chat context (conversation memory)
- retrieves relevant chunks from a vectorized document store
- answers using Retrieval-Augmented Generation (RAG) style flow
- is deployed with Streamlit

## Project Files

- `Task_10_Final_Phase/build_index.py`: builds the vector index from text documents
- `Task_10_Final_Phase/rag_engine.py`: chunking, embeddings, retrieval, and chat memory
- `Task_10_Final_Phase/app.py`: Streamlit chatbot UI
- `Task_10_Final_Phase/knowledge_base/`: custom corpus documents
- `Task_10_Final_Phase/vector_store/`: saved vector index outputs

## Setup

From project root:

```bash
pip install -r Task_10_Final_Phase/requirements.txt
```

## Build Vector Store

```bash
python Task_10_Final_Phase/build_index.py \
  --corpus-dir Task_10_Final_Phase/knowledge_base \
  --index-path Task_10_Final_Phase/vector_store/index.npz
```

## Run Chatbot

```bash
python -m streamlit run Task_10_Final_Phase/app.py
```

## How It Works

1. Documents are split into chunks.
2. Chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2`.
3. Embeddings are saved in a local vector store (`.npz`).
4. For each user query, top-k relevant chunks are retrieved.
5. The bot answers using retrieved context and recent conversation memory.

## Skills Gained

- Conversational AI development
- Document embedding and vector search
- Retrieval-Augmented Generation (RAG)
- LLM-style chatbot deployment with Streamlit
