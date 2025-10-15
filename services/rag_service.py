"""
RAG Service: Document embedding and retrieval with hybrid search.

Features:
- Semantic chunking using LangChain
- Rich metadata (filename, chunk_id, prev_chunk_id, span, etc.)
- OpenAI embeddings (text-embedding-3-small)
- FAISS vector store for local storage
- BM25 keyword search
- Hybrid search (vector + BM25)
- Auto-evaluation tools
"""

import os
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import faiss
from openai import OpenAI
from rank_bm25 import BM25Okapi

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings


@dataclass
class ChunkMetadata:
    """Metadata for each document chunk."""
    chunk_id: str
    filename: str
    chunk_index: int
    prev_chunk_id: Optional[str]
    next_chunk_id: Optional[str]
    span_start: int
    span_end: int
    char_count: int
    word_count: int
    created_at: str
    doc_section: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    """Result from retrieval with full metadata."""
    content: str
    metadata: ChunkMetadata
    score: float
    retrieval_method: str  # "vector", "bm25", or "hybrid"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "metadata": self.metadata.to_dict(),
            "score": self.score,
            "retrieval_method": self.retrieval_method
        }


class RAGService:
    """
    RAG Service for document embedding and retrieval.

    Uses:
    - OpenAI text-embedding-3-small for embeddings
    - FAISS for vector similarity search
    - BM25 for keyword-based search
    - Hybrid search combining both approaches
    """

    def __init__(
        self,
        docs_path: str = "data/docs",
        embeddings_path: str = "data/embeddings",
        openai_api_key: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_semantic_chunking: bool = True
    ):
        """
        Initialize RAG Service.

        Args:
            docs_path: Path to document directory
            embeddings_path: Path to store embeddings
            openai_api_key: OpenAI API key
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            use_semantic_chunking: Use semantic chunker vs recursive
        """
        self.docs_path = Path(docs_path)
        self.embeddings_path = Path(embeddings_path)
        self.embeddings_path.mkdir(parents=True, exist_ok=True)

        # OpenAI client
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required")

        self.client = OpenAI(api_key=self.api_key)

        # Chunking configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_semantic_chunking = use_semantic_chunking

        # Text splitter
        if use_semantic_chunking:
            # Semantic chunker (more intelligent splitting)
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=self.api_key
            )
            self.text_splitter = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=75
            )
        else:
            # Recursive character splitter (fallback)
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )

        # Storage
        self.chunks: List[str] = []
        self.metadata: List[ChunkMetadata] = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None  # Inner product (cosine similarity)
        self.bm25: Optional[BM25Okapi] = None

        # Embedding model
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dim = 1536  # Dimension for text-embedding-3-small

    def load_documents(self) -> List[Dict[str, str]]:
        """
        Load all markdown documents from docs directory.

        Returns:
            List of documents with filename and content
        """
        documents = []

        for file_path in sorted(self.docs_path.glob("*.md")):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            documents.append({
                "filename": file_path.name,
                "content": content,
                "path": str(file_path)
            })

        print(f"Loaded {len(documents)} documents")
        return documents

    def chunk_documents(self, documents: List[Dict[str, str]]) -> Tuple[List[str], List[ChunkMetadata]]:
        """
        Chunk documents with semantic splitting and rich metadata.

        Args:
            documents: List of documents to chunk

        Returns:
            Tuple of (chunks, metadata)
        """
        all_chunks = []
        all_metadata = []

        for doc in documents:
            filename = doc["filename"]
            content = doc["content"]

            # Split into chunks
            chunks = self.text_splitter.split_text(content)

            # Create metadata for each chunk
            prev_chunk_id = None
            for idx, chunk_text in enumerate(chunks):
                chunk_id = f"{filename}::chunk_{idx}"
                next_chunk_id = f"{filename}::chunk_{idx+1}" if idx < len(chunks) - 1 else None

                # Calculate span (approximate position in original document)
                # This is an approximation based on cumulative chunk lengths
                span_start = sum(len(chunks[i]) for i in range(idx))
                span_end = span_start + len(chunk_text)

                # Extract section from chunk (look for markdown headers)
                section = self._extract_section(chunk_text)

                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    filename=filename,
                    chunk_index=idx,
                    prev_chunk_id=prev_chunk_id,
                    next_chunk_id=next_chunk_id,
                    span_start=span_start,
                    span_end=span_end,
                    char_count=len(chunk_text),
                    word_count=len(chunk_text.split()),
                    created_at=datetime.utcnow().isoformat(),
                    doc_section=section
                )

                all_chunks.append(chunk_text)
                all_metadata.append(metadata)
                prev_chunk_id = chunk_id

        print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks, all_metadata

    def _extract_section(self, chunk_text: str) -> Optional[str]:
        """Extract section name from chunk (look for markdown headers)."""
        lines = chunk_text.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if line.startswith('#'):
                # Remove markdown header symbols
                return line.lstrip('#').strip()
        return None

    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """
        Embed chunks using OpenAI text-embedding-3-small.

        Args:
            chunks: List of text chunks to embed

        Returns:
            Numpy array of embeddings (n_chunks x embedding_dim)
        """
        print(f"Embedding {len(chunks)} chunks...")

        embeddings_list = []
        batch_size = 100  # OpenAI limit

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]

            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=batch
            )

            batch_embeddings = [item.embedding for item in response.data]
            embeddings_list.extend(batch_embeddings)

            if (i + batch_size) % 500 == 0:
                print(f"  Embedded {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

        embeddings_array = np.array(embeddings_list, dtype=np.float32)

        # Normalize for cosine similarity (used with IndexFlatIP)
        faiss.normalize_L2(embeddings_array)

        print(f"Embeddings shape: {embeddings_array.shape}")
        return embeddings_array

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """
        Build FAISS index for vector similarity search.

        Args:
            embeddings: Numpy array of embeddings

        Returns:
            FAISS index
        """
        print(f"Building FAISS index with {len(embeddings)} vectors...")

        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(embeddings)

        print(f"FAISS index built with {index.ntotal} vectors")
        return index

    def build_bm25_index(self, chunks: List[str]) -> BM25Okapi:
        """
        Build BM25 index for keyword-based search.

        Args:
            chunks: List of text chunks

        Returns:
            BM25 index
        """
        print(f"Building BM25 index with {len(chunks)} documents...")

        # Tokenize chunks for BM25
        tokenized_chunks = [chunk.lower().split() for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)

        print("BM25 index built")
        return bm25

    def initialize(self, force_rebuild: bool = False):
        """
        Initialize RAG service: load, chunk, embed, and index documents.

        Args:
            force_rebuild: Force rebuild even if cache exists
        """
        cache_file = self.embeddings_path / "rag_cache.pkl"

        # Try to load from cache
        if not force_rebuild and cache_file.exists():
            print("Loading from cache...")
            self.load_from_cache(cache_file)
            return

        # Build from scratch
        print("Building RAG index from scratch...")

        # 1. Load documents
        documents = self.load_documents()

        # 2. Chunk documents
        self.chunks, self.metadata = self.chunk_documents(documents)

        # 3. Embed chunks
        self.embeddings = self.embed_chunks(self.chunks)

        # 4. Build FAISS index
        self.faiss_index = self.build_faiss_index(self.embeddings)

        # 5. Build BM25 index
        self.bm25 = self.build_bm25_index(self.chunks)

        # 6. Save to cache
        self.save_to_cache(cache_file)

        print("RAG service initialized successfully")

    def save_to_cache(self, cache_file: Path):
        """Save RAG data to cache file."""
        print(f"Saving to cache: {cache_file}")

        cache_data = {
            "chunks": self.chunks,
            "metadata": [m.to_dict() for m in self.metadata],
            "embeddings": self.embeddings,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
            "use_semantic_chunking": self.use_semantic_chunking
        }

        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

        # Save FAISS index separately
        faiss_file = self.embeddings_path / "faiss_index.bin"
        faiss.write_index(self.faiss_index, str(faiss_file))

        print("Cache saved")

    def load_from_cache(self, cache_file: Path):
        """Load RAG data from cache file."""
        print(f"Loading from cache: {cache_file}")

        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)

        self.chunks = cache_data["chunks"]
        self.metadata = [ChunkMetadata(**m) for m in cache_data["metadata"]]
        self.embeddings = cache_data["embeddings"]

        # Load FAISS index
        faiss_file = self.embeddings_path / "faiss_index.bin"
        self.faiss_index = faiss.read_index(str(faiss_file))

        # Rebuild BM25 (fast)
        self.bm25 = self.build_bm25_index(self.chunks)

        print(f"Loaded {len(self.chunks)} chunks from cache")

    def search_vector(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Search using vector similarity (FAISS).

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of retrieval results
        """
        # Embed query
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=[query]
        )
        query_embedding = np.array([response.data[0].embedding], dtype=np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)

        # Search FAISS
        scores, indices = self.faiss_index.search(query_embedding, top_k)

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            result = RetrievalResult(
                content=self.chunks[idx],
                metadata=self.metadata[idx],
                score=float(score),
                retrieval_method="vector"
            )
            results.append(result)

        return results

    def search_bm25(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Search using BM25 keyword matching.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of retrieval results
        """
        # Tokenize query
        query_tokens = query.lower().split()

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            result = RetrievalResult(
                content=self.chunks[idx],
                metadata=self.metadata[idx],
                score=float(scores[idx]),
                retrieval_method="bm25"
            )
            results.append(result)

        return results

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        rerank: bool = True
    ) -> List[RetrievalResult]:
        """
        Hybrid search combining vector and BM25.

        Args:
            query: Search query
            top_k: Number of results to return
            vector_weight: Weight for vector search scores
            bm25_weight: Weight for BM25 scores
            rerank: Whether to rerank results

        Returns:
            List of retrieval results
        """
        # Get more candidates than needed for reranking
        candidate_k = top_k * 3 if rerank else top_k

        # Get results from both methods
        vector_results = self.search_vector(query, candidate_k)
        bm25_results = self.search_bm25(query, candidate_k)

        # Normalize scores to [0, 1]
        def normalize_scores(results: List[RetrievalResult]) -> List[RetrievalResult]:
            if not results:
                return results
            scores = [r.score for r in results]
            min_score, max_score = min(scores), max(scores)
            if max_score - min_score == 0:
                return results
            for r in results:
                r.score = (r.score - min_score) / (max_score - min_score)
            return results

        vector_results = normalize_scores(vector_results)
        bm25_results = normalize_scores(bm25_results)

        # Combine scores
        combined_scores = {}

        for result in vector_results:
            chunk_id = result.metadata.chunk_id
            combined_scores[chunk_id] = {
                "score": result.score * vector_weight,
                "result": result
            }

        for result in bm25_results:
            chunk_id = result.metadata.chunk_id
            if chunk_id in combined_scores:
                combined_scores[chunk_id]["score"] += result.score * bm25_weight
            else:
                combined_scores[chunk_id] = {
                    "score": result.score * bm25_weight,
                    "result": result
                }

        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )

        # Build final results
        final_results = []
        for chunk_id, data in sorted_results[:top_k]:
            result = data["result"]
            result.score = data["score"]
            result.retrieval_method = "hybrid"
            final_results.append(result)

        return final_results

    def search(
        self,
        query: str,
        top_k: int = 3,
        method: str = "hybrid"
    ) -> List[RetrievalResult]:
        """
        Main search interface.

        Args:
            query: Search query
            top_k: Number of results
            method: "vector", "bm25", or "hybrid"

        Returns:
            List of retrieval results
        """
        if method == "vector":
            return self.search_vector(query, top_k)
        elif method == "bm25":
            return self.search_bm25(query, top_k)
        elif method == "hybrid":
            return self.search_hybrid(query, top_k)
        else:
            raise ValueError(f"Unknown method: {method}")

    def get_context_window(
        self,
        result: RetrievalResult,
        window_size: int = 1
    ) -> str:
        """
        Get surrounding context for a retrieval result.

        Args:
            result: Retrieval result
            window_size: Number of chunks before/after to include

        Returns:
            Combined text with context
        """
        current_idx = result.metadata.chunk_index
        filename = result.metadata.filename

        # Find all chunks from same document
        doc_chunks = [
            (i, chunk, meta) for i, (chunk, meta) in enumerate(zip(self.chunks, self.metadata))
            if meta.filename == filename
        ]

        # Get window
        context_parts = []
        for i, chunk, meta in doc_chunks:
            if abs(meta.chunk_index - current_idx) <= window_size:
                marker = " >>> " if meta.chunk_index == current_idx else " "
                context_parts.append(f"{marker}{chunk}")

        return "\n\n".join(context_parts)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG service."""
        return {
            "total_chunks": len(self.chunks),
            "total_documents": len(set(m.filename for m in self.metadata)),
            "avg_chunk_length": np.mean([m.char_count for m in self.metadata]),
            "embedding_dimension": self.embedding_dim,
            "embedding_model": self.embedding_model,
            "chunking_method": "semantic" if self.use_semantic_chunking else "recursive"
        }


class RAGEvaluator:
    """Evaluation tools for RAG performance."""

    def __init__(self, rag_service: RAGService):
        self.rag = rag_service

    def evaluate_retrieval_quality(
        self,
        test_queries: List[Dict[str, Any]],
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval quality using test queries.

        Args:
            test_queries: List of dicts with 'query' and 'expected_docs'
            top_k: Number of results to retrieve

        Returns:
            Evaluation metrics
        """
        metrics = {
            "precision_at_k": [],
            "recall_at_k": [],
            "mrr": [],  # Mean Reciprocal Rank
            "avg_score": []
        }

        for test in test_queries:
            query = test["query"]
            expected = set(test["expected_docs"])

            # Retrieve
            results = self.rag.search(query, top_k=top_k)
            retrieved = [r.metadata.filename for r in results]

            # Precision@K
            relevant_retrieved = len([r for r in retrieved if r in expected])
            precision = relevant_retrieved / len(retrieved) if retrieved else 0
            metrics["precision_at_k"].append(precision)

            # Recall@K
            recall = relevant_retrieved / len(expected) if expected else 0
            metrics["recall_at_k"].append(recall)

            # MRR (Mean Reciprocal Rank)
            mrr = 0
            for rank, filename in enumerate(retrieved, 1):
                if filename in expected:
                    mrr = 1 / rank
                    break
            metrics["mrr"].append(mrr)

            # Average score
            avg_score = np.mean([r.score for r in results]) if results else 0
            metrics["avg_score"].append(avg_score)

        # Aggregate metrics
        return {
            "precision@k": np.mean(metrics["precision_at_k"]),
            "recall@k": np.mean(metrics["recall_at_k"]),
            "mrr": np.mean(metrics["mrr"]),
            "avg_retrieval_score": np.mean(metrics["avg_score"]),
            "num_queries": len(test_queries)
        }

    def compare_methods(
        self,
        queries: List[str],
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Compare different retrieval methods.

        Args:
            queries: List of test queries
            top_k: Number of results

        Returns:
            Comparison metrics
        """
        methods = ["vector", "bm25", "hybrid"]
        results = {}

        for method in methods:
            scores = []
            for query in queries:
                retrieved = self.rag.search(query, top_k=top_k, method=method)
                avg_score = np.mean([r.score for r in retrieved]) if retrieved else 0
                scores.append(avg_score)

            results[method] = {
                "avg_score": np.mean(scores),
                "std_score": np.std(scores),
                "min_score": np.min(scores),
                "max_score": np.max(scores)
            }

        return results

    def analyze_coverage(self) -> Dict[str, Any]:
        """Analyze document coverage in chunks."""
        doc_chunks = {}
        for meta in self.rag.metadata:
            if meta.filename not in doc_chunks:
                doc_chunks[meta.filename] = []
            doc_chunks[meta.filename].append(meta)

        coverage = {}
        for filename, chunks in doc_chunks.items():
            coverage[filename] = {
                "num_chunks": len(chunks),
                "avg_chunk_size": np.mean([c.char_count for c in chunks]),
                "total_chars": sum([c.char_count for c in chunks])
            }

        return coverage

    def test_query_performance(
        self,
        queries: List[str],
        methods: List[str] = ["vector", "bm25", "hybrid"]
    ):
        """
        Test and display query performance.

        Args:
            queries: List of test queries
            methods: List of methods to test
        """
        import time

        print("\n" + "="*80)
        print("RAG QUERY PERFORMANCE TEST")
        print("="*80)

        for query in queries:
            print(f"\nQuery: {query}")
            print("-" * 80)

            for method in methods:
                start = time.time()
                results = self.rag.search(query, top_k=3, method=method)
                elapsed = time.time() - start

                print(f"\n[{method.upper()}] - {elapsed*1000:.2f}ms")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result.metadata.filename} "
                          f"(chunk {result.metadata.chunk_index}, "
                          f"score: {result.score:.4f})")
                    if result.metadata.doc_section:
                        print(f"     Section: {result.metadata.doc_section}")


def main():
    """Example usage of RAG service."""

    # Initialize RAG service
    rag = RAGService(
        docs_path="data/docs",
        embeddings_path="data/embeddings",
        use_semantic_chunking=True
    )

    # Build or load index
    rag.initialize(force_rebuild=False)

    # Print statistics
    stats = rag.get_stats()
    print("\n" + "="*80)
    print("RAG SERVICE STATISTICS")
    print("="*80)
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Example queries
    test_queries = [
        "How do I fix high latency issues?",
        "What is the deployment process?",
        "How do I monitor error rates?",
        "What is the system architecture?",
        "How do I authenticate with the API?"
    ]

    # Test retrieval
    print("\n" + "="*80)
    print("EXAMPLE RETRIEVAL RESULTS")
    print("="*80)

    for query in test_queries[:2]:  # Show first 2
        print(f"\nQuery: {query}")
        print("-" * 80)

        results = rag.search(query, top_k=3, method="hybrid")

        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.metadata.filename} - {result.metadata.doc_section}")
            print(f"   Score: {result.score:.4f}")
            print(f"   Chunk: {result.metadata.chunk_index}")
            print(f"   Preview: {result.content[:200]}...")

    # Evaluation
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)

    evaluator = RAGEvaluator(rag)

    # Compare methods
    comparison = evaluator.compare_methods(test_queries, top_k=3)
    print("\nMethod Comparison:")
    for method, metrics in comparison.items():
        print(f"\n{method.upper()}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

    # Coverage analysis
    print("\n" + "="*80)
    print("DOCUMENT COVERAGE")
    print("="*80)
    coverage = evaluator.analyze_coverage()
    for filename, info in coverage.items():
        print(f"\n{filename}:")
        print(f"  Chunks: {info['num_chunks']}")
        print(f"  Avg chunk size: {info['avg_chunk_size']:.0f} chars")


if __name__ == "__main__":
    main()
