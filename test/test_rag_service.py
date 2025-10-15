"""
Test script for RAG service.

Tests:
- Document loading and chunking
- Embedding generation
- Vector search
- BM25 search
- Hybrid search
- Evaluation metrics
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.rag_service import RAGService, RAGEvaluator


def test_initialization():
    """Test RAG service initialization."""
    print("\n" + "="*80)
    print("TEST 1: RAG SERVICE INITIALIZATION")
    print("="*80)

    rag = RAGService(
        docs_path="data/docs",
        embeddings_path="data/embeddings",
        use_semantic_chunking=True
    )

    print("\n✓ RAG service created successfully")
    print(f"  Docs path: {rag.docs_path}")
    print(f"  Embeddings path: {rag.embeddings_path}")
    print(f"  Embedding model: {rag.embedding_model}")
    print(f"  Chunking method: {'Semantic' if rag.use_semantic_chunking else 'Recursive'}")

    return rag


def test_document_loading(rag):
    """Test document loading."""
    print("\n" + "="*80)
    print("TEST 2: DOCUMENT LOADING")
    print("="*80)

    documents = rag.load_documents()

    print(f"\n✓ Loaded {len(documents)} documents:")
    for doc in documents:
        print(f"  - {doc['filename']}: {len(doc['content'])} characters")

    return documents


def test_chunking(rag, documents):
    """Test document chunking."""
    print("\n" + "="*80)
    print("TEST 3: DOCUMENT CHUNKING")
    print("="*80)

    chunks, metadata = rag.chunk_documents(documents)

    print(f"\n✓ Created {len(chunks)} chunks")
    print(f"\nSample chunk metadata:")
    for i in range(min(3, len(metadata))):
        meta = metadata[i]
        print(f"\n  Chunk {i}:")
        print(f"    ID: {meta.chunk_id}")
        print(f"    Filename: {meta.filename}")
        print(f"    Index: {meta.chunk_index}")
        print(f"    Prev: {meta.prev_chunk_id}")
        print(f"    Next: {meta.next_chunk_id}")
        print(f"    Span: {meta.span_start}-{meta.span_end}")
        print(f"    Chars: {meta.char_count}")
        print(f"    Words: {meta.word_count}")
        print(f"    Section: {meta.doc_section}")

    return chunks, metadata


def test_vector_search(rag):
    """Test vector search."""
    print("\n" + "="*80)
    print("TEST 4: VECTOR SEARCH")
    print("="*80)

    queries = [
        "How do I troubleshoot high latency?",
        "What is the deployment process?",
        "How do I use the API?"
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 80)

        results = rag.search_vector(query, top_k=3)

        for i, result in enumerate(results, 1):
            print(f"\n  {i}. Score: {result.score:.4f}")
            print(f"     File: {result.metadata.filename}")
            print(f"     Section: {result.metadata.doc_section}")
            print(f"     Chunk: {result.metadata.chunk_index}")
            print(f"     Preview: {result.content[:100]}...")

    print("\n✓ Vector search completed")


def test_bm25_search(rag):
    """Test BM25 search."""
    print("\n" + "="*80)
    print("TEST 5: BM25 SEARCH")
    print("="*80)

    queries = [
        "latency performance optimization",
        "docker kubernetes deployment",
        "authentication JWT token"
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 80)

        results = rag.search_bm25(query, top_k=3)

        for i, result in enumerate(results, 1):
            print(f"\n  {i}. Score: {result.score:.4f}")
            print(f"     File: {result.metadata.filename}")
            print(f"     Section: {result.metadata.doc_section}")
            print(f"     Preview: {result.content[:100]}...")

    print("\n✓ BM25 search completed")


def test_hybrid_search(rag):
    """Test hybrid search."""
    print("\n" + "="*80)
    print("TEST 6: HYBRID SEARCH")
    print("="*80)

    queries = [
        "How to reduce API latency?",
        "Deployment rollback procedures",
        "Monitoring alerts and dashboards"
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 80)

        results = rag.search_hybrid(query, top_k=3)

        for i, result in enumerate(results, 1):
            print(f"\n  {i}. Score: {result.score:.4f}")
            print(f"     File: {result.metadata.filename}")
            print(f"     Section: {result.metadata.doc_section}")
            print(f"     Method: {result.retrieval_method}")
            print(f"     Preview: {result.content[:100]}...")

    print("\n✓ Hybrid search completed")


def test_context_window(rag):
    """Test context window retrieval."""
    print("\n" + "="*80)
    print("TEST 7: CONTEXT WINDOW")
    print("="*80)

    query = "What causes high error rates?"
    results = rag.search(query, top_k=1, method="hybrid")

    if results:
        result = results[0]
        print(f"\nQuery: '{query}'")
        print(f"Top result: {result.metadata.filename} - Chunk {result.metadata.chunk_index}")
        print(f"\nContext window (±1 chunk):")
        print("-" * 80)

        context = rag.get_context_window(result, window_size=1)
        print(context[:500] + "...")

    print("\n✓ Context window retrieval completed")


def test_evaluation(rag):
    """Test evaluation metrics."""
    print("\n" + "="*80)
    print("TEST 8: EVALUATION METRICS")
    print("="*80)

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
        },
        {
            "query": "How do I authenticate with the API?",
            "expected_docs": ["api_guide.md"]
        },
        {
            "query": "What is the system architecture?",
            "expected_docs": ["architecture.md"]
        }
    ]

    print("\nEvaluating retrieval quality...")
    metrics = evaluator.evaluate_retrieval_quality(test_queries, top_k=3)

    print("\n✓ Evaluation Metrics:")
    print(f"  Precision@3: {metrics['precision@k']:.4f}")
    print(f"  Recall@3: {metrics['recall@k']:.4f}")
    print(f"  MRR: {metrics['mrr']:.4f}")
    print(f"  Avg Score: {metrics['avg_retrieval_score']:.4f}")
    print(f"  Test Queries: {metrics['num_queries']}")


def test_method_comparison(rag):
    """Test comparison of different methods."""
    print("\n" + "="*80)
    print("TEST 9: METHOD COMPARISON")
    print("="*80)

    evaluator = RAGEvaluator(rag)

    queries = [
        "How to troubleshoot high latency?",
        "Deployment rollback procedures",
        "API authentication methods",
        "System architecture overview",
        "Monitoring and alerting setup"
    ]

    comparison = evaluator.compare_methods(queries, top_k=3)

    print("\n✓ Method Comparison:")
    for method, metrics in comparison.items():
        print(f"\n{method.upper()}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")


def test_coverage_analysis(rag):
    """Test coverage analysis."""
    print("\n" + "="*80)
    print("TEST 10: COVERAGE ANALYSIS")
    print("="*80)

    evaluator = RAGEvaluator(rag)
    coverage = evaluator.analyze_coverage()

    print("\n✓ Document Coverage:")
    for filename, info in coverage.items():
        print(f"\n{filename}:")
        print(f"  Chunks: {info['num_chunks']}")
        print(f"  Avg chunk size: {info['avg_chunk_size']:.0f} chars")
        print(f"  Total chars: {info['total_chars']}")


def test_performance(rag):
    """Test query performance."""
    print("\n" + "="*80)
    print("TEST 11: QUERY PERFORMANCE")
    print("="*80)

    evaluator = RAGEvaluator(rag)

    queries = [
        "How to fix high latency?",
        "Deployment process overview"
    ]

    evaluator.test_query_performance(queries, methods=["vector", "bm25", "hybrid"])

    print("\n✓ Performance test completed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("RAG SERVICE TEST SUITE")
    print("="*80)

    try:
        # Test 1: Initialization
        rag = test_initialization()

        # Initialize RAG (load or build)
        print("\n" + "="*80)
        print("INITIALIZING RAG SERVICE")
        print("="*80)
        print("\nThis may take a few minutes on first run...")
        rag.initialize(force_rebuild=False)

        # Test 2: Document loading
        documents = test_document_loading(rag)

        # Test 3: Chunking (already done in initialize, but show metadata)
        print("\n" + "="*80)
        print("TEST 3: CHUNK METADATA")
        print("="*80)
        print(f"\nTotal chunks: {len(rag.chunks)}")
        print(f"\nSample metadata:")
        for i in range(min(3, len(rag.metadata))):
            meta = rag.metadata[i]
            print(f"\n  Chunk {i}:")
            print(f"    ID: {meta.chunk_id}")
            print(f"    File: {meta.filename}")
            print(f"    Section: {meta.doc_section}")
            print(f"    Chars: {meta.char_count}")

        # Test 4-6: Search methods
        test_vector_search(rag)
        test_bm25_search(rag)
        test_hybrid_search(rag)

        # Test 7: Context window
        test_context_window(rag)

        # Test 8-10: Evaluation
        test_evaluation(rag)
        test_method_comparison(rag)
        test_coverage_analysis(rag)

        # Test 11: Performance
        test_performance(rag)

        # Final statistics
        print("\n" + "="*80)
        print("FINAL STATISTICS")
        print("="*80)
        stats = rag.get_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")

        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED")
        print("="*80)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
