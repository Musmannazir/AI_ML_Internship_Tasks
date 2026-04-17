from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


SUPPORTED_EXTENSIONS = {".txt", ".md"}


@dataclass
class RAGConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 700
    chunk_overlap: int = 120
    top_k: int = 4
    memory_turns: int = 4


class LocalVectorStore:
    def __init__(self, embeddings: np.ndarray, chunks: List[str], metadata: List[Dict[str, str]]) -> None:
        self.embeddings = embeddings.astype(np.float32)
        self.chunks = chunks
        self.metadata = metadata

    @staticmethod
    def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        chunks: List[str] = []
        start = 0
        stride = max(1, chunk_size - overlap)
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += stride
        return chunks

    @classmethod
    def from_corpus(cls, corpus_dir: Path, config: RAGConfig) -> "LocalVectorStore":
        if not corpus_dir.exists():
            raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

        model = SentenceTransformer(config.embedding_model)

        all_chunks: List[str] = []
        all_meta: List[Dict[str, str]] = []

        for path in sorted(corpus_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            text = path.read_text(encoding="utf-8", errors="ignore")
            chunks = cls._chunk_text(text, config.chunk_size, config.chunk_overlap)
            for idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_meta.append({"source": str(path), "chunk_id": str(idx)})

        if not all_chunks:
            raise ValueError("No text chunks were created. Add .txt or .md documents to the corpus directory.")

        vectors = model.encode(all_chunks, normalize_embeddings=True, show_progress_bar=False)
        return cls(embeddings=np.array(vectors), chunks=all_chunks, metadata=all_meta)

    def save(self, index_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(index_path, embeddings=self.embeddings)

        meta_path = index_path.with_suffix(".json")
        payload = {"chunks": self.chunks, "metadata": self.metadata}
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_path: Path) -> "LocalVectorStore":
        if not index_path.exists():
            raise FileNotFoundError(f"Vector index not found: {index_path}")

        meta_path = index_path.with_suffix(".json")
        if not meta_path.exists():
            raise FileNotFoundError(f"Vector metadata not found: {meta_path}")

        arr = np.load(index_path)
        embeddings = arr["embeddings"]
        payload = json.loads(meta_path.read_text(encoding="utf-8"))

        return cls(
            embeddings=embeddings,
            chunks=payload["chunks"],
            metadata=payload["metadata"],
        )

    def retrieve(self, query: str, model: SentenceTransformer, top_k: int) -> List[Dict[str, str]]:
        q = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = self.embeddings @ q
        top_idx = np.argsort(scores)[::-1][:top_k]

        results: List[Dict[str, str]] = []
        for i in top_idx:
            results.append(
                {
                    "score": f"{float(scores[i]):.4f}",
                    "text": self.chunks[int(i)],
                    "source": self.metadata[int(i)]["source"],
                    "chunk_id": self.metadata[int(i)]["chunk_id"],
                }
            )
        return results


class ContextAwareRAGChatbot:
    def __init__(self, store: LocalVectorStore, config: RAGConfig | None = None) -> None:
        self.config = config or RAGConfig()
        self.store = store
        self.model = SentenceTransformer(self.config.embedding_model)
        self.history: List[Tuple[str, str]] = []

    def _history_block(self) -> str:
        recent = self.history[-self.config.memory_turns :]
        if not recent:
            return "No previous conversation history."
        lines = []
        for i, (user, assistant) in enumerate(recent, start=1):
            lines.append(f"Turn {i} user: {user}")
            lines.append(f"Turn {i} assistant: {assistant}")
        return "\n".join(lines)

    def _compose_answer(self, query: str, docs: List[Dict[str, str]]) -> str:
        if not docs:
            return "I could not retrieve relevant context from the knowledge base."

        context = "\n\n".join([f"[{i+1}] {d['text']}" for i, d in enumerate(docs)])
        history = self._history_block()

        # Lightweight extractive response from retrieved context + memory reminder.
        answer = (
            f"Based on the retrieved documents, here is the best answer:\n\n"
            f"{docs[0]['text']}\n\n"
            f"Related context:\n{context}\n\n"
            f"Conversation memory considered:\n{history}"
        )
        return answer

    def chat(self, query: str) -> Dict[str, object]:
        docs = self.store.retrieve(query=query, model=self.model, top_k=self.config.top_k)
        answer = self._compose_answer(query, docs)
        self.history.append((query, answer))

        return {"answer": answer, "sources": docs}
