# RAG Service Documentation

## Overview

The RAG (Retrieval-Augmented Generation) service provides document embedding and retrieval capabilities with hybrid search combining vector similarity and keyword matching.

## Features

### 1. Semantic Chunking
- Uses LangChain's `SemanticChunker` for intelligent text splitting
- Breaks documents at natural semantic boundaries
- Preserves context and meaning

### 2. Rich Metadata
Each chunk includes:
- `chunk_id`: Unique identifier
- `filename`: Source document
- `chunk_index`: Position in document
- `prev_chunk_id` / `next_chunk_id`: Navigation links
- `span_start` / `span_end`: Character positions
- `char_count` / `word_count`: Size metrics
- `doc_section`: Extracted section heading
- `created_at`: Timestamp

### 3. Vector Search (FAISS)
- Uses OpenAI `text-embedding-3-small` (1536 dimensions)
- FAISS IndexFlatIP for cosine similarity
- Fast similarity search
- Normalized embeddings

### 4. BM25 Keyword Search
- Traditional keyword-based retrieval
- Statistical ranking (TF-IDF variant)
- Good for exact term matching

### 5. Hybrid Search
- Combines vector and BM25 results
- Weighted score fusion
- Configurable weights
- Better overall performance

### 6. Evaluation Tools
- Precision@K, Recall@K metrics
- Mean Reciprocal Rank (MRR)
- Method comparison
- Coverage analysis
- Performance benchmarking

## Installation

```bash
pip install -r requirements-rag.txt
```

## Usage

### Basic Usage

```python
from services.rag_service import RAGService

# Initialize
rag = RAGService(
    docs_path="data/docs",
    embeddings_path="data/embeddings",
    use_semantic_chunking=True
)

# Build index (or load from cache)
rag.initialize(force_rebuild=False)

# Search
results = rag.search(
    query="How to fix high latency?",
    top_k=3,
    method="hybrid"  # or "vector" or "bm25"
)

# Display results
for result in results:
    print(f"Score: {result.score:.4f}")
    print(f"File: {result.metadata.filename}")
    print(f"Section: {result.metadata.doc_section}")
    print(f"Content: {result.content[:200]}...")
```

### Vector Search

```python
# Pure semantic search
results = rag.search_vector(query="API authentication", top_k=5)
```

### BM25 Search

```python
# Keyword-based search
results = rag.search_bm25(query="latency performance optimization", top_k=5)
```

### Hybrid Search

```python
# Combined approach (recommended)
results = rag.search_hybrid(
    query="How to deploy services?",
    top_k=5,
    vector_weight=0.7,  # 70% vector
    bm25_weight=0.3,    # 30% BM25
    rerank=True
)
```

### Context Window

```python
# Get surrounding chunks for context
result = results[0]
context = rag.get_context_window(result, window_size=1)
print(context)
```

### Statistics

```python
stats = rag.get_stats()
print(stats)
# Output:
# {
#   'total_chunks': 150,
#   'total_documents': 5,
#   'avg_chunk_length': 1024.5,
#   'embedding_dimension': 1536,
#   'embedding_model': 'text-embedding-3-small',
#   'chunking_method': 'semantic'
# }
```

## Evaluation

```python
from services.rag_service import RAGEvaluator

evaluator = RAGEvaluator(rag)

# Test queries with expected documents
test_queries = [
    {
        "query": "How do I fix high latency?",
        "expected_docs": ["troubleshooting.md", "monitoring.md"]
    },
    {
        "query": "What is the deployment process?",
        "expected_docs": ["deployment.md"]
    }
]

# Evaluate
metrics = evaluator.evaluate_retrieval_quality(test_queries, top_k=3)
print(f"Precision@3: {metrics['precision@k']:.4f}")
print(f"Recall@3: {metrics['recall@k']:.4f}")
print(f"MRR: {metrics['mrr']:.4f}")

# Compare methods
comparison = evaluator.compare_methods(
    queries=["latency", "deployment", "API"],
    top_k=3
)

# Coverage analysis
coverage = evaluator.analyze_coverage()
```

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional (defaults shown)
DOCS_PATH=data/docs
EMBEDDINGS_PATH=data/embeddings
```

### Initialization Parameters

```python
RAGService(
    docs_path="data/docs",              # Document directory
    embeddings_path="data/embeddings",  # Cache directory
    openai_api_key=None,                # API key (or from env)
    chunk_size=1000,                    # Chunk size (if not semantic)
    chunk_overlap=200,                  # Overlap (if not semantic)
    use_semantic_chunking=True          # Use semantic vs recursive
)
```

## Testing

```bash
# Run full test suite
python tests/test_rag_service.py

# Or run as module
python -m pytest tests/test_rag_service.py -v
```

## Performance

### Embedding Speed
- ~100 chunks per batch
- OpenAI rate limits apply
- First run: 2-5 minutes for 150 chunks
- Cached runs: <1 second

### Search Speed
- Vector search: ~10ms for 150 chunks
- BM25 search: ~5ms for 150 chunks
- Hybrid search: ~15ms (combined)

### Memory Usage
- FAISS index: ~1MB per 1000 vectors
- Metadata: ~1KB per chunk
- Total: ~2-5MB for typical setup

## Cache Management

Cache is stored in `data/embeddings/`:
- `rag_cache.pkl`: Chunks, metadata, embeddings
- `faiss_index.bin`: FAISS vector index

To rebuild:
```python
rag.initialize(force_rebuild=True)
```

To clear cache:
```bash
rm -rf data/embeddings/*
```

## Architecture

```
RAGService
├── Document Loading
│   └── Load .md files from docs_path
├── Chunking
│   ├── SemanticChunker (intelligent)
│   └── RecursiveCharacterTextSplitter (fallback)
├── Embedding
│   └── OpenAI text-embedding-3-small
├── Indexing
│   ├── FAISS (vector similarity)
│   └── BM25 (keyword matching)
└── Retrieval
    ├── Vector search
    ├── BM25 search
    └── Hybrid (weighted fusion)
```

## Metadata Schema

```python
ChunkMetadata(
    chunk_id="api_guide.md::chunk_5",
    filename="api_guide.md",
    chunk_index=5,
    prev_chunk_id="api_guide.md::chunk_4",
    next_chunk_id="api_guide.md::chunk_6",
    span_start=5120,
    span_end=6144,
    char_count=1024,
    word_count=187,
    created_at="2025-10-15T10:30:00Z",
    doc_section="Authentication"
)
```

## Retrieval Result Schema

```python
RetrievalResult(
    content="Full chunk text...",
    metadata=ChunkMetadata(...),
    score=0.85,
    retrieval_method="hybrid"
)
```

## Best Practices

### 1. Use Hybrid Search
Hybrid search typically outperforms single-method retrieval:
- Vector search: Good for semantic similarity
- BM25: Good for keyword matching
- Hybrid: Best of both worlds

### 2. Tune Weights
Adjust based on your use case:
- Technical docs: 0.6 vector, 0.4 BM25
- Natural language: 0.8 vector, 0.2 BM25
- Keyword-heavy: 0.4 vector, 0.6 BM25

### 3. Use Context Windows
For better understanding:
```python
context = rag.get_context_window(result, window_size=1)
```

### 4. Cache Embeddings
Always use cache for repeated queries:
```python
rag.initialize(force_rebuild=False)  # Load from cache
```

### 5. Monitor Performance
Use evaluation tools to track quality:
```python
evaluator = RAGEvaluator(rag)
metrics = evaluator.evaluate_retrieval_quality(test_queries)
```

## Troubleshooting

### Issue: Slow initialization
**Solution**: Use cache - only rebuild when documents change

### Issue: Poor retrieval quality
**Solution**: Try different methods, adjust hybrid weights, or use semantic chunking

### Issue: OpenAI API errors
**Solution**: Check API key, rate limits, and network connectivity

### Issue: Memory errors
**Solution**: Reduce chunk count or use smaller embeddings

## API Reference

See docstrings in `rag_service.py` for detailed API documentation.

## Examples

See `tests/test_rag_service.py` for comprehensive examples of all features.
