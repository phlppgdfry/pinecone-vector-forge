"""Pattern 4: Hierarchical & Contextual Parent-Document Chunking.

Resolves the fundamental RAG dilemma:
- Small chunks provide crisp, focused vector similarity.
- Large documents provide coherent context for generation.
This pattern indexes small semantic chunks linked to rich parent document metadata.
"""

import re
from typing import List, Dict, Any, Optional
from ..core.models import Document, Chunk, VectorRecord
from ..core.embeddings import get_embedding_generator


class HierarchicalChunker:
    """Chunks documents with sliding semantic window and parent-linkage."""

    def __init__(self, chunk_size: int = 400, overlap: int = 80):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embedder = get_embedding_generator()

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """Splits document text into overlapping chunks, preserving parent metadata."""
        text = doc.content.strip()
        words = text.split()

        if len(words) <= self.chunk_size:
            return [
                Chunk(
                    chunk_id=f"{doc.id}_chunk_0",
                    document_id=doc.id,
                    chunk_index=0,
                    text=text,
                    token_count=len(words),
                    parent_title=doc.title,
                    category=doc.category,
                    tags=doc.tags,
                    metadata={
                        **doc.metadata,
                        "author": doc.author,
                        "created_at": doc.created_at.isoformat()
                    }
                )
            ]

        chunks: List[Chunk] = []
        step = max(1, self.chunk_size - self.overlap)
        chunk_idx = 0

        for i in range(0, len(words), step):
            window_words = words[i:i + self.chunk_size]
            if not window_words:
                break

            chunk_text = " ".join(window_words)
            chunk_id = f"{doc.id}_chunk_{chunk_idx}"

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=doc.id,
                    chunk_index=chunk_idx,
                    text=chunk_text,
                    token_count=len(window_words),
                    parent_title=doc.title,
                    category=doc.category,
                    tags=doc.tags,
                    metadata={
                        **doc.metadata,
                        "author": doc.author,
                        "chunk_window_start": i,
                        "chunk_window_end": i + len(window_words),
                        "created_at": doc.created_at.isoformat()
                    }
                )
            )
            chunk_idx += 1

            if i + self.chunk_size >= len(words):
                break

        return chunks

    def process_and_vectorize(self, doc: Document) -> List[VectorRecord]:
        """Splits document into chunks and computes dense & sparse vectors."""
        chunks = self.chunk_document(doc)
        records: List[VectorRecord] = []

        texts = [c.text for c in chunks]
        dense_embeddings = self.embedder.embed_batch(texts)

        for chunk, dense_vec in zip(chunks, dense_embeddings):
            sparse_vec = self.embedder.generate_sparse_vector(chunk.text)
            
            # Pack metadata into Pinecone record (limited to Pinecone metadata spec)
            metadata = {
                "document_id": chunk.document_id,
                "parent_title": chunk.parent_title,
                "chunk_index": chunk.chunk_index,
                "category": chunk.category,
                "tags": chunk.tags,
                "token_count": chunk.token_count,
                "text": chunk.text[:1000],  # Stored text snippet for direct retrieval
                **{k: v for k, v in chunk.metadata.items() if isinstance(v, (str, int, float, bool, list))}
            }

            records.append(
                VectorRecord(
                    id=chunk.chunk_id,
                    values=dense_vec,
                    sparse_values=sparse_vec,
                    metadata=metadata
                )
            )

        return records
