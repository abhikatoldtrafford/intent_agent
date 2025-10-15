#!/usr/bin/env python3
"""
Start the Complete Agent System

This script:
1. Checks prerequisites (OpenAI key, database, docs)
2. Initializes RAG service (if needed)
3. Starts the FastAPI metrics server

Single entry point for the entire system.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_prerequisites():
    """Check all prerequisites and return status."""
    print("\n🔍 Checking Prerequisites...")
    print("-" * 60)

    checks = {}

    # Check OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    checks["OpenAI API Key"] = bool(api_key)
    print(f"  {'✓' if checks['OpenAI API Key'] else '✗'} OpenAI API Key: {'Set' if api_key else 'NOT SET'}")

    # Check database
    db_path = Path("data/metrics.db")
    checks["Database"] = db_path.exists()
    if checks["Database"]:
        size_mb = db_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Database: Found ({size_mb:.2f} MB)")
    else:
        print(f"  ✗ Database: NOT FOUND")

    # Check documentation
    docs_path = Path("data/docs")
    md_files = list(docs_path.glob("*.md")) if docs_path.exists() else []
    checks["Documentation"] = len(md_files) >= 3
    print(f"  {'✓' if checks['Documentation'] else '✗'} Documentation: {len(md_files)} markdown files")

    # Check RAG embeddings
    embeddings_cache = Path("data/embeddings/rag_cache.pkl")
    faiss_index = Path("data/embeddings/faiss_index.bin")
    checks["RAG Embeddings"] = embeddings_cache.exists() and faiss_index.exists()
    print(f"  {'✓' if checks['RAG Embeddings'] else '✗'} RAG Embeddings: {'Initialized' if checks['RAG Embeddings'] else 'NOT INITIALIZED'}")

    return checks


def initialize_rag():
    """Initialize RAG service."""
    print("\n📚 Initializing RAG Service...")
    print("-" * 60)

    try:
        from demo.demo_rag import initialize_rag as init_rag_func
        rag = init_rag_func(force_rebuild=False)

        stats = rag.get_stats()
        print(f"  ✓ RAG initialized successfully")
        print(f"    - Documents: {stats['total_documents']}")
        print(f"    - Chunks: {stats['total_chunks']}")
        print(f"    - Embedding model: {stats['embedding_model']}")
        return True

    except Exception as e:
        print(f"  ✗ RAG initialization failed: {e}")
        return False


def start_api_server():
    """Start the FastAPI server."""
    print("\n🚀 Starting Metrics API Server...")
    print("-" * 60)
    print("\n  Server URL: http://127.0.0.1:8001")
    print("  API Docs:   http://127.0.0.1:8001/docs")
    print("  Alternative: http://127.0.0.1:8001/redoc")
    print("\n  Press Ctrl+C to stop the server")
    print("="*80 + "\n")

    try:
        # Run uvicorn
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "services.api_service:app",
            "--host", "127.0.0.1",
            "--port", "8001",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped gracefully.")
    except Exception as e:
        print(f"\n✗ Error starting server: {e}")
        print("\nMake sure required packages are installed:")
        print("  pip install -r requirements.txt")
        sys.exit(1)


def main():
    """Main entry point."""
    print("="*80)
    print("INTENT-ROUTED AGENT - SYSTEM STARTUP")
    print("="*80)

    # Check prerequisites
    checks = check_prerequisites()

    # Handle missing prerequisites
    if not checks.get("OpenAI API Key"):
        print("\n⚠️  CRITICAL: OpenAI API key not set!")
        print("\nSet it with:")
        print("  export OPENAI_API_KEY='sk-...'")
        print("\nOr create a .env file with:")
        print("  OPENAI_API_KEY=sk-...")
        sys.exit(1)

    if not checks.get("Database"):
        print("\n⚠️  WARNING: Database not found")
        print("  Some features may not work correctly")
        print("  Run: python test/validate_db_service.py")
        print()

    if not checks.get("Documentation"):
        print("\n⚠️  WARNING: Documentation files not found")
        print("  RAG service requires markdown files in data/docs/")
        print()

    # Initialize RAG if needed
    if not checks.get("RAG Embeddings"):
        print("\n⚙️  RAG embeddings not found - initializing now...")
        if checks.get("Documentation"):
            success = initialize_rag()
            if not success:
                print("\n⚠️  RAG initialization failed - continuing anyway")
                print("  You can initialize manually later: python demo/demo_rag.py")
        else:
            print("  ✗ Cannot initialize RAG without documentation files")
    else:
        print("\n✓ All services ready!")

    # Start API server
    start_api_server()


if __name__ == "__main__":
    main()
