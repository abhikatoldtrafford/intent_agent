"""
RAG Service Initialization and Demo

Initializes the RAG service by:
1. Loading documentation from data/docs/
2. Chunking with semantic splitter
3. Embedding with OpenAI
4. Building FAISS + BM25 indices
5. Caching for fast reuse

This script can be run standalone or imported for initialization.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file (override existing)
load_dotenv(override=True)

from services.rag_service import RAGService, RAGEvaluator


def log(message: str, level: str = "INFO"):
    """Enhanced logging with timestamps and levels."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    symbols = {
        "INFO": "ℹ️",
        "SUCCESS": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "PROGRESS": "🔄",
        "STEP": "➡️"
    }
    symbol = symbols.get(level, "•")
    print(f"[{timestamp}] {symbol} {level}: {message}")
    sys.stdout.flush()


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)
    sys.stdout.flush()


def initialize_rag(force_rebuild: bool = False) -> RAGService:
    """
    Initialize RAG service with error handling and detailed logging.

    Args:
        force_rebuild: Force rebuild even if cache exists

    Returns:
        Initialized RAGService instance

    Raises:
        ValueError: If OpenAI API key not set
        FileNotFoundError: If docs directory doesn't exist
    """
    log("Starting RAG service initialization...", "PROGRESS")

    # Check prerequisites
    log("Checking prerequisites...", "STEP")

    if not os.getenv("OPENAI_API_KEY"):
        log("OpenAI API key not found in environment", "ERROR")
        raise ValueError("OPENAI_API_KEY environment variable not set")

    log("✓ OpenAI API key found", "SUCCESS")

    docs_path = Path("data/docs")
    if not docs_path.exists():
        log(f"Documentation directory not found: {docs_path}", "ERROR")
        raise FileNotFoundError(f"Documentation directory not found: {docs_path}")

    log(f"✓ Documentation directory found: {docs_path}", "SUCCESS")

    # Count markdown files
    md_files = list(docs_path.glob("*.md"))
    if not md_files:
        log(f"No markdown files found in {docs_path}", "ERROR")
        raise FileNotFoundError(f"No markdown files found in {docs_path}")

    log(f"✓ Found {len(md_files)} documentation files", "SUCCESS")
    for i, file in enumerate(md_files, 1):
        size_kb = file.stat().st_size / 1024
        log(f"  [{i}] {file.name} ({size_kb:.1f} KB)", "INFO")

    # Check cache status
    embeddings_path = Path("data/embeddings")
    cache_file = embeddings_path / "rag_cache.pkl"
    faiss_index = embeddings_path / "faiss_index.bin"

    if cache_file.exists() and faiss_index.exists() and not force_rebuild:
        cache_age = time.time() - cache_file.stat().st_mtime
        log(f"✓ Cache found (age: {cache_age/60:.1f} minutes)", "INFO")
        log("Will load from cache for fast initialization", "INFO")
    else:
        if force_rebuild:
            log("Force rebuild requested - will rebuild from scratch", "WARNING")
        else:
            log("No cache found - will build new index (may take 1-2 minutes)", "INFO")

    # Initialize service
    log("Initializing RAG service instance...", "STEP")
    start_time = time.time()

    try:
        rag = RAGService(
            docs_path="data/docs",
            embeddings_path="data/embeddings",
            use_semantic_chunking=True
        )
        log(f"✓ RAG service instance created ({(time.time()-start_time)*1000:.0f}ms)", "SUCCESS")
    except Exception as e:
        log(f"Failed to create RAG service: {e}", "ERROR")
        raise

    # Build/load index
    log("Building/loading RAG index...", "STEP")
    index_start = time.time()

    try:
        rag.initialize(force_rebuild=force_rebuild)
        index_time = time.time() - index_start
        log(f"✓ Index ready ({index_time:.2f}s)", "SUCCESS")
    except Exception as e:
        log(f"Failed to build index: {e}", "ERROR")
        raise

    total_time = time.time() - start_time
    log(f"✓ RAG initialization complete ({total_time:.2f}s total)", "SUCCESS")

    return rag


def demo_queries(rag: RAGService):
    """Run demo queries to test RAG service with detailed output."""
    test_queries = [
        "How do I fix high latency?",
        "What is the deployment process?",
        "How do I monitor errors?",
        "What is the system architecture?",
    ]

    print_header("RUNNING TEST QUERIES")
    log(f"Testing RAG search with {len(test_queries)} queries", "INFO")

    for idx, query in enumerate(test_queries, 1):
        print(f"\n{'─'*80}")
        log(f"Query {idx}/{len(test_queries)}: {query}", "STEP")

        query_start = time.time()

        try:
            results = rag.search(query, top_k=2, method="hybrid")
            query_time = (time.time() - query_start) * 1000

            log(f"✓ Search completed in {query_time:.0f}ms", "SUCCESS")
            log(f"Found {len(results)} results", "INFO")

            for i, result in enumerate(results, 1):
                print(f"\n  Result {i}:")
                log(f"    File: {result.metadata.filename}", "INFO")
                log(f"    Score: {result.score:.4f}", "INFO")
                if result.metadata.doc_section:
                    log(f"    Section: {result.metadata.doc_section}", "INFO")
                log(f"    Chunk ID: {result.metadata.chunk_id}", "INFO")
                print(f"    Preview: {result.content[:150]}...")

        except Exception as e:
            log(f"Query failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()


def show_detailed_stats(rag: RAGService):
    """Display detailed statistics about the RAG service."""
    print_header("RAG SERVICE STATISTICS")

    try:
        stats = rag.get_stats()

        log(f"Total Documents: {stats.get('total_documents', 0)}", "INFO")
        log(f"Total Chunks: {stats.get('total_chunks', 0)}", "INFO")
        log(f"Embedding Model: {stats.get('embedding_model', 'N/A')}", "INFO")
        log(f"Embedding Dimensions: {stats.get('embedding_dimensions', 0)}", "INFO")

        if stats.get('avg_chunk_size'):
            log(f"Average Chunk Size: {stats['avg_chunk_size']:.0f} chars", "INFO")

        if stats.get('cache_size_mb'):
            log(f"Cache Size: {stats['cache_size_mb']:.2f} MB", "INFO")

        if stats.get('index_type'):
            log(f"Index Type: {stats['index_type']}", "INFO")

    except Exception as e:
        log(f"Failed to get stats: {e}", "WARNING")


def main():
    """Main entry point for demo."""
    print_header("RAG SERVICE INITIALIZATION DEMO")

    log("Intent-Routed Agent POC - RAG Service", "INFO")
    log(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "INFO")

    try:
        # Initialize RAG
        rag = initialize_rag(force_rebuild=False)

        # Show detailed stats
        show_detailed_stats(rag)

        # Run demo queries
        demo_queries(rag)

        # Success message
        print_header("✅ RAG SERVICE READY")

        log("RAG service is fully initialized and operational", "SUCCESS")
        log("Cache has been saved for fast reuse", "INFO")

        print("\n📚 You can now run the agent:")
        print("  • python demo/demo_agent.py")
        print("  • python test/test_agent.py")
        print("  • python main.py  # Interactive CLI")
        print("  • streamlit run streamlit_app.py  # Web UI")

        print("\n🔍 RAG Features Available:")
        print("  • Hybrid search (Vector + BM25)")
        print("  • Semantic chunking with metadata")
        print("  • Context window retrieval")
        print("  • Multi-document coverage")

        print("\n" + "="*80 + "\n")

    except ValueError as e:
        log(str(e), "ERROR")
        print("\n❌ Configuration Error")
        print("\n⚙️  Setup required:")
        print("  export OPENAI_API_KEY='sk-...'")
        print("\nOr create a .env file:")
        print("  OPENAI_API_KEY=sk-...")
        sys.exit(1)

    except FileNotFoundError as e:
        log(str(e), "ERROR")
        print("\n❌ File Error")
        print("\n📁 Make sure the data/docs directory exists with markdown files")
        sys.exit(1)

    except Exception as e:
        log(f"Initialization failed: {e}", "ERROR")
        print("\n❌ Initialization Error")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
