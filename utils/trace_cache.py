"""
Trace Cache Manager

Caches execution traces from multiple sources:
- Agent executions (from run_agent)
- LangSmith traces (via API)
- OpenTelemetry traces
- Demo/test executions

Cache lifetime: 24 hours
Auto-populates on Streamlit app startup
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# Cache configuration
CACHE_DIR = Path(__file__).parent.parent / "data" / "trace_cache"
CACHE_LIFETIME_HOURS = 24
LANGSMITH_CACHE_FILE = CACHE_DIR / "langsmith_traces.json"
OTEL_CACHE_FILE = CACHE_DIR / "otel_traces.json"
AGENT_CACHE_FILE = CACHE_DIR / "agent_executions.json"
DEMO_CACHE_FILE = CACHE_DIR / "demo_executions.json"
TEST_CACHE_FILE = CACHE_DIR / "test_executions.json"


def ensure_cache_dir():
    """Ensure cache directory exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def is_cache_valid(cache_file: Path) -> bool:
    """Check if cache file exists and is within lifetime."""
    if not cache_file.exists():
        return False

    # Check file modification time
    mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
    age = datetime.now() - mtime

    if age > timedelta(hours=CACHE_LIFETIME_HOURS):
        logger.info(f"Cache expired: {cache_file.name} (age: {age.total_seconds()/3600:.1f}h)")
        return False

    logger.info(f"Cache valid: {cache_file.name} (age: {age.total_seconds()/3600:.1f}h)")
    return True


def load_cache(cache_file: Path) -> Optional[List[Dict[str, Any]]]:
    """Load cache from file if valid."""
    if not is_cache_valid(cache_file):
        return None

    try:
        with open(cache_file, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} items from cache: {cache_file.name}")
        return data
    except Exception as e:
        logger.error(f"Error loading cache {cache_file.name}: {e}")
        return None


def save_cache(cache_file: Path, data: List[Dict[str, Any]]):
    """Save cache to file."""
    ensure_cache_dir()

    try:
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved {len(data)} items to cache: {cache_file.name}")
    except Exception as e:
        logger.error(f"Error saving cache {cache_file.name}: {e}")


def append_to_cache(cache_file: Path, new_data: Dict[str, Any]):
    """Append a new item to cache."""
    # Load existing cache
    existing = load_cache(cache_file) or []

    # Add timestamp if not present
    if 'cached_at' not in new_data:
        new_data['cached_at'] = datetime.now().isoformat()

    # Append new data
    existing.append(new_data)

    # Save updated cache
    save_cache(cache_file, existing)


def get_langsmith_traces() -> List[Dict[str, Any]]:
    """
    Get LangSmith traces from cache or API.

    Returns:
        List of LangSmith trace objects
    """
    # Try cache first
    cached = load_cache(LANGSMITH_CACHE_FILE)
    if cached:
        return cached

    # Fetch from LangSmith API
    traces = fetch_langsmith_traces()

    if traces:
        save_cache(LANGSMITH_CACHE_FILE, traces)

    return traces or []


def fetch_langsmith_traces() -> Optional[List[Dict[str, Any]]]:
    """
    Fetch traces from LangSmith using Python SDK.

    Uses LangSmith Python SDK to retrieve recent traces for the project.
    """
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    project_name = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT", "intent-agent-poc")

    if not api_key:
        logger.warning("LANGSMITH_API_KEY not set - cannot fetch traces")
        return None

    try:
        # Use LangSmith SDK instead of REST API (more reliable)
        from langsmith import Client

        client = Client(api_key=api_key)

        # Get runs from last 24 hours
        start_time = datetime.now() - timedelta(hours=24)

        logger.info(f"Fetching LangSmith traces for project: {project_name}")

        # List runs using SDK
        runs = list(client.list_runs(
            project_name=project_name,
            start_time=start_time,
            is_root=True,  # Only root runs
            limit=100
        ))

        logger.info(f"Fetched {len(runs)} traces from LangSmith")

        # Transform to our format
        traces = []
        for run in runs:
            trace = {
                "id": str(run.id),
                "name": run.name,
                "run_type": run.run_type,
                "start_time": run.start_time.isoformat() if run.start_time else None,
                "end_time": run.end_time.isoformat() if run.end_time else None,
                "status": run.status if hasattr(run, 'status') else None,
                "inputs": run.inputs or {},
                "outputs": run.outputs or {},
                "error": run.error,
                "trace_id": str(run.trace_id) if run.trace_id else None,
                "parent_run_id": str(run.parent_run_id) if run.parent_run_id else None,
                "execution_order": run.execution_order if hasattr(run, 'execution_order') else None,
                "session_id": str(run.session_id) if hasattr(run, 'session_id') else None,
                "url": run.url if hasattr(run, 'url') else f"https://smith.langchain.com/o/{run.id}/r",
                "fetched_at": datetime.now().isoformat()
            }
            traces.append(trace)

        return traces

    except ImportError:
        logger.error("langsmith package not installed - run: pip install langsmith")
        return None
    except Exception as e:
        logger.error(f"Error fetching LangSmith traces: {e}")
        return None


def get_agent_executions() -> List[Dict[str, Any]]:
    """Get agent execution traces from cache or recent runs."""
    # Try multiple sources
    cached = load_cache(AGENT_CACHE_FILE)
    if cached:
        return cached

    # Check for demo executions
    demo_file = Path(__file__).parent.parent / "data" / "demo_executions.json"
    if demo_file.exists():
        try:
            with open(demo_file, 'r') as f:
                demo_data = json.load(f)
                logger.info(f"Loaded {len(demo_data)} demo executions")
                # Cache them
                save_cache(AGENT_CACHE_FILE, demo_data)
                return demo_data
        except Exception as e:
            logger.error(f"Error loading demo executions: {e}")

    # Check for test executions
    test_file = Path(__file__).parent.parent / "data" / "test_executions.json"
    if test_file.exists():
        try:
            with open(test_file, 'r') as f:
                test_data = json.load(f)
                logger.info(f"Loaded {len(test_data)} test executions")
                return test_data
        except Exception as e:
            logger.error(f"Error loading test executions: {e}")

    return []


def cache_agent_execution(result: Dict[str, Any], query: str):
    """Cache an agent execution result."""
    execution = {
        "query": query,
        "result": result,
        "timestamp": datetime.now().isoformat(),
        "cached_at": datetime.now().isoformat()
    }

    append_to_cache(AGENT_CACHE_FILE, execution)


def get_all_cached_traces() -> Dict[str, Any]:
    """
    Get all cached traces from all sources.

    Returns:
        Dictionary with traces from all sources:
        {
            "langsmith": [...],
            "agent_executions": [...],
            "demo_executions": [...],
            "test_executions": [...],
            "cache_status": {...}
        }
    """
    return {
        "langsmith": get_langsmith_traces(),
        "agent_executions": get_agent_executions(),
        "demo_executions": load_cache(DEMO_CACHE_FILE) or [],
        "test_executions": load_cache(TEST_CACHE_FILE) or [],
        "cache_status": get_cache_status()
    }


def get_cache_status() -> Dict[str, Any]:
    """Get status of all caches."""
    status = {}

    for name, cache_file in [
        ("langsmith", LANGSMITH_CACHE_FILE),
        ("agent", AGENT_CACHE_FILE),
        ("demo", DEMO_CACHE_FILE),
        ("test", TEST_CACHE_FILE)
    ]:
        if cache_file.exists():
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            age = datetime.now() - mtime
            size = cache_file.stat().st_size

            status[name] = {
                "exists": True,
                "valid": is_cache_valid(cache_file),
                "age_hours": age.total_seconds() / 3600,
                "size_kb": size / 1024,
                "last_modified": mtime.isoformat()
            }
        else:
            status[name] = {
                "exists": False,
                "valid": False
            }

    return status


def clear_cache(cache_name: Optional[str] = None):
    """Clear cache files."""
    if cache_name:
        # Clear specific cache
        cache_files = {
            "langsmith": LANGSMITH_CACHE_FILE,
            "agent": AGENT_CACHE_FILE,
            "demo": DEMO_CACHE_FILE,
            "test": TEST_CACHE_FILE
        }

        cache_file = cache_files.get(cache_name)
        if cache_file and cache_file.exists():
            cache_file.unlink()
            logger.info(f"Cleared cache: {cache_name}")
    else:
        # Clear all caches
        for cache_file in [LANGSMITH_CACHE_FILE, AGENT_CACHE_FILE, DEMO_CACHE_FILE, TEST_CACHE_FILE]:
            if cache_file.exists():
                cache_file.unlink()
        logger.info("Cleared all caches")


def refresh_langsmith_cache():
    """Force refresh LangSmith cache from API."""
    logger.info("Refreshing LangSmith cache...")
    traces = fetch_langsmith_traces()

    if traces:
        save_cache(LANGSMITH_CACHE_FILE, traces)
        logger.info(f"Refreshed LangSmith cache with {len(traces)} traces")
        return True
    else:
        logger.error("Failed to refresh LangSmith cache")
        return False


def auto_populate_traces():
    """
    Auto-populate traces from all available sources.

    Called on Streamlit app startup to load cached traces.
    Returns summary of loaded traces.
    """
    logger.info("Auto-populating traces from cache...")

    summary = {
        "langsmith_traces": 0,
        "agent_executions": 0,
        "demo_executions": 0,
        "test_executions": 0,
        "total": 0
    }

    # Load LangSmith traces
    langsmith = get_langsmith_traces()
    summary["langsmith_traces"] = len(langsmith)

    # Load agent executions
    agent = get_agent_executions()
    summary["agent_executions"] = len(agent)

    # Load demo executions
    demo = load_cache(DEMO_CACHE_FILE)
    if demo:
        summary["demo_executions"] = len(demo)

    # Load test executions
    test = load_cache(TEST_CACHE_FILE)
    if test:
        summary["test_executions"] = len(test)

    summary["total"] = sum([
        summary["langsmith_traces"],
        summary["agent_executions"],
        summary["demo_executions"],
        summary["test_executions"]
    ])

    logger.info(f"Auto-populated {summary['total']} traces from cache")
    return summary
