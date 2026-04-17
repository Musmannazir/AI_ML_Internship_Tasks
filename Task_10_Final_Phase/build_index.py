from __future__ import annotations

import argparse
from pathlib import Path

from rag_engine import LocalVectorStore, RAGConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build vector index for Task 10 RAG chatbot")
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default="Task_10_Final_Phase/knowledge_base",
        help="Folder containing .txt or .md files.",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default="Task_10_Final_Phase/vector_store/index.npz",
        help="Output vector index path (.npz).",
    )
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = RAGConfig(
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    corpus_dir = Path(args.corpus_dir)
    index_path = Path(args.index_path)

    store = LocalVectorStore.from_corpus(corpus_dir=corpus_dir, config=config)
    store.save(index_path=index_path)

    print("Task 10 index build completed")
    print(f"Corpus directory: {corpus_dir}")
    print(f"Saved index: {index_path}")
    print(f"Saved metadata: {index_path.with_suffix('.json')}")
    print(f"Indexed chunks: {len(store.chunks)}")


if __name__ == "__main__":
    main()
