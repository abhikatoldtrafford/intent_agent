"""
Agent Demonstration Script

Demonstrates the intent-routed agent with various query types:
1. Metrics lookup queries
2. Knowledge base queries
3. Calculation queries
4. Mixed queries requiring multiple tools
5. Historical data queries

Prerequisites:
- API server running: python start_api_server.py
- RAG initialized: python demo/demo_rag.py
- Database created: python test/validate_db_service.py
- OPENAI_API_KEY environment variable set

This version runs non-interactively for automated demos.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import time
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import run_agent, get_graph_visualization

# Track all agent executions for observability
DEMO_EXECUTIONS = []
DEMO_RESULTS_FILE = Path(__file__).parent.parent / "data" / "demo_executions.json"


def log(message: str, level: str = "INFO"):
    """Enhanced logging with timestamps and levels."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    symbols = {
        "INFO": "ℹ️",
        "SUCCESS": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "PROGRESS": "🔄",
        "STEP": "➡️",
        "QUERY": "❓",
        "ANSWER": "💡"
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


def print_subheader(title: str):
    """Print a formatted subheader."""
    print("\n" + "─"*80)
    print(f"  {title}")
    print("─"*80)
    sys.stdout.flush()


def check_prerequisites():
    """Check and log all prerequisites."""
    log("Checking prerequisites...", "STEP")

    checks = {
        "OpenAI API Key": bool(os.getenv("OPENAI_API_KEY")),
        "RAG Service": os.path.exists("data/embeddings/rag_cache.pkl"),
        "Database": os.path.exists("data/metrics.db"),
        "API Server": False
    }

    # Check API server
    try:
        import requests
        response = requests.get("http://127.0.0.1:8001/health", timeout=2)
        checks["API Server"] = response.status_code == 200
    except:
        pass

    all_ok = all(checks.values())

    for check, status in checks.items():
        if status:
            log(f"✓ {check}", "SUCCESS")
        else:
            log(f"✗ {check}", "WARNING")

    if not all_ok:
        print_header("⚠️  MISSING PREREQUISITES")
        print("\n⚙️  Setup steps:")
        if not checks["OpenAI API Key"]:
            print("  1. Set OpenAI key: export OPENAI_API_KEY=your_key")
        if not checks["RAG Service"]:
            print("  2. Initialize RAG: python demo/demo_rag.py")
        if not checks["Database"]:
            print("  3. Create database: python test/validate_db_service.py")
        if not checks["API Server"]:
            print("  4. Start API server: python start_api_server.py")
        print("\nContinuing with demo (some queries may fail)...\n")

    return all_ok


def run_demo_query(query: str, description: str = "", section: str = ""):
    """Run a demo query and display results with comprehensive logging."""
    if section:
        print_subheader(section)

    print(f"\n{'─'*80}")
    log(query, "QUERY")
    if description:
        log(f"Expected: {description}", "INFO")

    query_start = time.time()

    try:
        # Run the agent
        log("Starting agent execution...", "PROGRESS")
        result = run_agent(query, verbose=False)

        query_time = time.time() - query_start

        # Log key information
        log(f"✓ Agent completed in {query_time:.2f}s", "SUCCESS")

        # Display intent and confidence
        intent = result.get('intent', 'unknown')
        confidence = result.get('confidence', 0)
        log(f"Intent classified as: {intent} (confidence: {confidence:.2f})", "INFO")

        # Display tools used
        tools = result.get('tools_executed', [])
        if tools:
            log(f"Tools executed: {', '.join(tools)}", "INFO")
        else:
            log("No tools executed", "WARNING")

        # Display processing time breakdown
        total_ms = result.get('total_duration_ms', 0)
        log(f"Total processing time: {total_ms:.0f}ms", "INFO")

        # Display node durations if available
        node_durations = result.get('node_durations', {})
        if node_durations:
            log("Node execution times:", "INFO")
            for node, duration in node_durations.items():
                print(f"    • {node}: {duration:.0f}ms")

        # Display findings if any
        findings = result.get('findings', [])
        if findings:
            log(f"Findings ({len(findings)}):", "INFO")
            for i, finding in enumerate(findings, 1):
                print(f"    {i}. {finding}")

        # Display recommendations if any
        recommendations = result.get('recommendations', [])
        if recommendations:
            log(f"Recommendations ({len(recommendations)}):", "INFO")
            for i, rec in enumerate(recommendations, 1):
                print(f"    {i}. {rec}")

        # Display answer
        answer = result.get("final_answer", "No answer generated")
        log("Agent answer:", "ANSWER")
        print(f"\n{answer}\n")

        # Display trace summary
        trace = result.get("trace", [])
        if trace:
            log(f"Trace events captured: {len(trace)}", "INFO")

        # Check for errors
        tool_errors = result.get('tool_errors', {})
        if tool_errors:
            log(f"Tool errors encountered: {len(tool_errors)}", "WARNING")
            for tool_name, error in tool_errors.items():
                print(f"    • {tool_name}: {error}")

        # Display data quality if available
        data_quality = result.get('data_quality', {})
        if data_quality:
            completeness = data_quality.get('completeness', 0)
            consistency = data_quality.get('consistency', 0)
            log(f"Data quality: completeness={completeness:.0%}, consistency={consistency:.0%}", "INFO")

        # Save execution for observability
        DEMO_EXECUTIONS.append({
            "query": query,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })

        return True

    except Exception as e:
        query_time = time.time() - query_start
        log(f"Query failed after {query_time:.2f}s: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all demo queries non-interactively."""
    print_header("INTENT-ROUTED AGENT DEMONSTRATION")

    log("Intent-Routed Agent POC - Comprehensive Demo", "INFO")
    log(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "INFO")

    # Check prerequisites
    all_ok = check_prerequisites()

    # Show graph visualization
    print_header("AGENT WORKFLOW GRAPH")
    try:
        graph_viz = get_graph_visualization()
        print(graph_viz)
        log("✓ Workflow graph generated", "SUCCESS")
    except Exception as e:
        log(f"Could not generate graph: {e}", "WARNING")

    # Demo queries organized by intent type
    print_header("RUNNING DEMO QUERIES")
    log("Executing queries across all intent types", "INFO")

    success_count = 0
    total_count = 0

    # 1. Metrics Lookup Queries
    print_header("SECTION 1: METRICS LOOKUP QUERIES")
    log("Testing REST API tool and real-time metrics", "INFO")

    queries = [
        ("What is the current latency for api-gateway?",
         "Simple metrics query - should use REST API tool"),
        ("Show me error rates for auth-service",
         "Error metrics query"),
        ("Is the business-logic service healthy?",
         "Health check query"),
    ]

    for query, desc in queries:
        total_count += 1
        if run_demo_query(query, desc):
            success_count += 1
        time.sleep(0.5)  # Brief pause between queries

    # 2. Knowledge Lookup Queries
    print_header("SECTION 2: KNOWLEDGE BASE QUERIES")
    log("Testing RAG tool and documentation search", "INFO")

    queries = [
        ("How do I configure API rate limiting?",
         "Documentation query - should use RAG tool"),
        ("What are the best practices for deployment?",
         "Best practices knowledge query"),
        ("How do I troubleshoot high error rates?",
         "Troubleshooting documentation query"),
    ]

    for query, desc in queries:
        total_count += 1
        if run_demo_query(query, desc):
            success_count += 1
        time.sleep(0.5)

    # 3. Calculation Queries
    print_header("SECTION 3: CALCULATION QUERIES")
    log("Testing calculator tool", "INFO")

    queries = [
        ("Calculate the average of 150, 200, and 250",
         "Simple calculation - should use calculator tool"),
        ("If latency is 95ms and threshold is 100ms, is it within limits?",
         "Comparison calculation"),
    ]

    for query, desc in queries:
        total_count += 1
        if run_demo_query(query, desc):
            success_count += 1
        time.sleep(0.5)

    # 4. Historical Data Queries
    print_header("SECTION 4: HISTORICAL DATA QUERIES")
    log("Testing SQL database tool", "INFO")

    queries = [
        ("What was the average CPU usage for api-gateway over the past week?",
         "Historical query - should use SQL database"),
        ("Compare memory usage between api-gateway and auth-service",
         "Comparison query using database"),
    ]

    for query, desc in queries:
        total_count += 1
        if run_demo_query(query, desc):
            success_count += 1
        time.sleep(0.5)

    # 5. Mixed Queries
    print_header("SECTION 5: MIXED QUERIES (Multiple Tools)")
    log("Testing multi-tool coordination", "INFO")

    queries = [
        ("What is the latency for api-gateway and how can I improve it?",
         "Mixed query - needs metrics API and knowledge base"),
        ("Show me the error rate and explain how to reduce errors",
         "Mixed query - needs metrics and documentation"),
    ]

    for query, desc in queries:
        total_count += 1
        if run_demo_query(query, desc):
            success_count += 1
        time.sleep(0.5)

    # 6. Complex Multi-Source Queries (NEW SECTION)
    print_header("SECTION 6: COMPLEX MULTI-SOURCE QUERIES")
    log("Testing advanced query patterns with intelligent routing", "INFO")

    queries = [
        ("Show me services where CPU exceeded 75% in the last 4 days and rank by frequency",
         "Complex historical query - SQL database with filtering and aggregation"),
        ("Compare current throughput across all services and recommend optimization strategies",
         "Complex mixed query - current metrics + knowledge base recommendations"),
        ("What percentage of requests for payment-service resulted in errors over the past week?",
         "Complex calculation query - historical data + percentage computation"),
        ("Identify services with degraded status patterns in the last 72 hours and explain common causes",
         "Complex diagnostic query - historical analysis + troubleshooting knowledge"),
        ("Show correlation between memory spikes and error rates for business-logic service",
         "Complex analysis query - multi-metric correlation from historical data"),
    ]

    for query, desc in queries:
        total_count += 1
        if run_demo_query(query, desc):
            success_count += 1
        time.sleep(0.5)

    # Save demo executions for observability
    try:
        DEMO_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(DEMO_RESULTS_FILE, 'w') as f:
            json.dump(DEMO_EXECUTIONS, f, indent=2, default=str)
        log(f"✓ Saved {len(DEMO_EXECUTIONS)} executions to {DEMO_RESULTS_FILE}", "SUCCESS")
    except Exception as e:
        log(f"Warning: Could not save demo executions: {e}", "WARNING")

    # Final Summary
    print_header("DEMO COMPLETE")

    success_rate = (success_count / total_count * 100) if total_count > 0 else 0
    log(f"Successfully completed {success_count}/{total_count} queries ({success_rate:.1f}%)", "INFO")

    print("\n✨ Demo Features Demonstrated:")
    print("  • Intent classification using OpenAI GPT-4o-mini")
    print("  • Automatic tool selection based on intent")
    print("  • Multi-tool execution and aggregation")
    print("  • Inference and threshold checking")
    print("  • Confidence scoring and feedback loop")
    print("  • Complete execution trace")

    print("\n🔧 Tools Tested:")
    print("  • REST API tool (real-time metrics)")
    print("  • RAG tool (documentation search)")
    print("  • SQL database tool (historical data)")
    print("  • Calculator tool (computations)")

    print("\n📊 Workflow Nodes:")
    print("  • classify_intent - Intent classification")
    print("  • select_tools - Tool selection")
    print("  • execute_tools - Tool execution")
    print("  • aggregate_results - Data aggregation")
    print("  • perform_inference - Analysis and insights")
    print("  • check_feedback - Confidence evaluation")
    print("  • format_response - Response formatting")

    print("\n📚 Next Steps:")
    print("  • Run tests: python test/test_agent.py")
    print("  • Try interactive CLI: python main.py")
    print("  • Try web UI: streamlit run streamlit_app.py")
    print("  • Read docs: agent/README.md")

    print("\n" + "="*80 + "\n")

    if success_count < total_count:
        log(f"Some queries failed - check prerequisites", "WARNING")
        sys.exit(1)
    else:
        log("All queries completed successfully!", "SUCCESS")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        log(f"Demo failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)
