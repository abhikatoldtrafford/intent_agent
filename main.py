#!/usr/bin/env python3
"""
Interactive CLI for Intent-Routed Agent

Provides a command-line interface for querying the agent with natural language.

Features:
- Interactive query loop
- Trace visualization toggle
- Graph visualization
- Prerequisite checking
- Colored output (if supported)
- Command history

Usage:
    python main.py

Commands:
    help   - Show help message
    trace  - Toggle trace display
    graph  - Show workflow graph
    clear  - Clear screen
    quit   - Exit (or Ctrl+C)
"""

# Load environment variables from .env file (override system env)
from dotenv import load_dotenv
load_dotenv(override=True)

import os
import sys
from pathlib import Path
from typing import Dict, Any

try:
    from agent import run_agent, get_graph_visualization
    from agent.state import (
        INTENT_METRICS_LOOKUP,
        INTENT_KNOWLEDGE_LOOKUP,
        INTENT_CALCULATION,
        INTENT_MIXED,
        INTENT_CLARIFICATION,
        INTENT_UNKNOWN
    )
except ImportError as e:
    print(f"Error importing agent: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


# ANSI color codes (for terminals that support it)
class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def disable():
        """Disable colors (for Windows or non-ANSI terminals)."""
        Colors.HEADER = ''
        Colors.OKBLUE = ''
        Colors.OKCYAN = ''
        Colors.OKGREEN = ''
        Colors.WARNING = ''
        Colors.FAIL = ''
        Colors.ENDC = ''
        Colors.BOLD = ''
        Colors.UNDERLINE = ''


# Disable colors on Windows or if not a TTY
if os.name == 'nt' or not sys.stdout.isatty():
    Colors.disable()


def print_header(text: str):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}\n")


def print_section(text: str):
    """Print a subsection."""
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}{text}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'-'*60}{Colors.ENDC}")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.OKGREEN}✓{Colors.ENDC} {text}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.FAIL}✗{Colors.ENDC} {text}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.WARNING}⚠{Colors.ENDC} {text}")


def check_prerequisites() -> Dict[str, bool]:
    """Check all prerequisites and return status."""
    print_section("Checking Prerequisites")

    checks = {}

    # Check OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    checks["OpenAI API Key"] = bool(api_key)
    if checks["OpenAI API Key"]:
        print_success(f"OpenAI API Key: Set")
    else:
        print_error("OpenAI API Key: NOT SET")

    # Check database
    db_path = Path("data/metrics.db")
    checks["Database"] = db_path.exists()
    if checks["Database"]:
        size_mb = db_path.stat().st_size / (1024 * 1024)
        print_success(f"Database: Found ({size_mb:.2f} MB)")
    else:
        print_error("Database: NOT FOUND")

    # Check documentation
    docs_path = Path("data/docs")
    md_files = list(docs_path.glob("*.md")) if docs_path.exists() else []
    checks["Documentation"] = len(md_files) >= 3
    if checks["Documentation"]:
        print_success(f"Documentation: {len(md_files)} markdown files")
    else:
        print_error(f"Documentation: Only {len(md_files)} files found")

    # Check RAG embeddings
    embeddings_cache = Path("data/embeddings/rag_cache.pkl")
    faiss_index = Path("data/embeddings/faiss_index.bin")
    checks["RAG Embeddings"] = embeddings_cache.exists() and faiss_index.exists()
    if checks["RAG Embeddings"]:
        print_success("RAG Embeddings: Initialized")
    else:
        print_error("RAG Embeddings: NOT INITIALIZED")

    # Check API server
    try:
        import requests
        response = requests.get("http://127.0.0.1:8001/health", timeout=2)
        checks["API Server"] = response.status_code == 200
        if checks["API Server"]:
            print_success("API Server: Running at http://127.0.0.1:8001")
        else:
            print_error(f"API Server: Unhealthy (status {response.status_code})")
    except Exception:
        checks["API Server"] = False
        print_error("API Server: NOT RUNNING")

    # Print warnings for missing prerequisites
    if not all(checks.values()):
        print()
        print_warning("Some prerequisites are missing!")
        print("\nSetup steps:")
        if not checks["OpenAI API Key"]:
            print("  1. Set OpenAI key: export OPENAI_API_KEY='sk-...'")
        if not checks["RAG Embeddings"]:
            print("  2. Initialize RAG: python demo/demo_rag.py")
        if not checks["API Server"]:
            print("  3. Start API server: python start_api_server.py (in another terminal)")
        if not checks["Database"]:
            print("  4. Create database: python test/validate_db_service.py")

    return checks


def show_help():
    """Display help message."""
    help_text = """
{bold}Available Commands:{endc}

  {cyan}help{endc}   - Show this help message
  {cyan}trace{endc}  - Toggle trace display on/off
  {cyan}graph{endc}  - Show agent workflow graph visualization
  {cyan}clear{endc}  - Clear the screen
  {cyan}quit{endc}   - Exit the CLI (or press Ctrl+C)
  {cyan}exit{endc}   - Same as quit

{bold}Example Queries:{endc}

  {green}Metrics Lookup:{endc}
    • What is the latency for api-gateway?
    • Show me error rates for auth-service
    • Is business-logic healthy?

  {green}Knowledge Base:{endc}
    • How do I configure API rate limiting?
    • What are deployment best practices?
    • How do I troubleshoot high latency?

  {green}Calculations:{endc}
    • Calculate (150 + 200) / 2
    • What is 95 compared to 100?
    • Compute the average of 10, 20, and 30

  {green}Historical Data:{endc}
    • What was the average CPU usage for api-gateway last week?
    • Compare memory usage between services

  {green}Mixed Queries:{endc}
    • What is the latency and how do I improve it?
    • Show error rates and explain how to reduce them

{bold}Tips:{endc}
  • Be specific about service names and metrics
  • Use "current" or "historical" to control data source
  • Enable trace to see detailed execution steps
""".format(
        bold=Colors.BOLD,
        endc=Colors.ENDC,
        cyan=Colors.OKCYAN,
        green=Colors.OKGREEN
    )
    print(help_text)


def format_intent_display(intent: str) -> str:
    """Format intent for display with color."""
    intent_colors = {
        INTENT_METRICS_LOOKUP: Colors.OKBLUE,
        INTENT_KNOWLEDGE_LOOKUP: Colors.OKGREEN,
        INTENT_CALCULATION: Colors.OKCYAN,
        INTENT_MIXED: Colors.WARNING,
        INTENT_CLARIFICATION: Colors.WARNING,
        INTENT_UNKNOWN: Colors.FAIL
    }
    color = intent_colors.get(intent, Colors.ENDC)
    return f"{color}{intent}{Colors.ENDC}"


def display_result(result: Dict[str, Any], show_trace: bool = False):
    """Display agent result in a formatted way."""
    # Result header
    print(f"\n{Colors.BOLD}{Colors.OKGREEN}{'─'*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKGREEN}RESULT{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKGREEN}{'─'*80}{Colors.ENDC}\n")

    # Main answer
    print(result.get("final_answer", "No answer generated"))

    # Metadata
    print(f"\n{Colors.BOLD}{Colors.OKGREEN}{'─'*80}{Colors.ENDC}")

    intent_display = format_intent_display(result.get('intent', 'unknown'))
    confidence = result.get('confidence', 0)

    # Confidence color
    if confidence >= 0.8:
        conf_color = Colors.OKGREEN
    elif confidence >= 0.6:
        conf_color = Colors.WARNING
    else:
        conf_color = Colors.FAIL

    print(f"{Colors.BOLD}Intent:{Colors.ENDC} {intent_display} | "
          f"{Colors.BOLD}Confidence:{Colors.ENDC} {conf_color}{confidence:.2f}{Colors.ENDC} | "
          f"{Colors.BOLD}Time:{Colors.ENDC} {result.get('total_duration_ms', 0):.0f}ms")

    tools_executed = result.get('tools_executed', [])
    if tools_executed:
        tools_str = ", ".join(tools_executed)
        print(f"{Colors.BOLD}Tools Used:{Colors.ENDC} {Colors.OKCYAN}{tools_str}{Colors.ENDC}")

    print(f"{Colors.BOLD}{Colors.OKGREEN}{'─'*80}{Colors.ENDC}\n")

    # Show trace if enabled
    if show_trace:
        print_section("Execution Trace")
        trace_events = result.get('trace', [])

        for i, event in enumerate(trace_events, 1):
            node = event.get('node', '')
            event_type = event.get('event_type', '')
            timestamp = event.get('timestamp', '')

            print(f"{Colors.BOLD}{i}.{Colors.ENDC} [{Colors.OKCYAN}{node}{Colors.ENDC}] {event_type}")

            # Show interesting data
            data = event.get('data', {})
            if isinstance(data, dict) and data:
                for key, value in list(data.items())[:3]:  # Show first 3 items
                    if isinstance(value, (str, int, float, bool)):
                        print(f"   {Colors.BOLD}{key}:{Colors.ENDC} {value}")

        # Node durations
        print_section("Node Durations")
        for node, duration in result.get('node_durations', {}).items():
            print(f"  {Colors.OKCYAN}{node}{Colors.ENDC}: {duration:.0f}ms")


def main():
    """Main CLI loop."""
    print_header("INTENT-ROUTED AGENT - INTERACTIVE CLI")

    # Check prerequisites
    checks = check_prerequisites()

    if not checks.get("OpenAI API Key"):
        print()
        print_error("Cannot run without OpenAI API key!")
        print("Set it with: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    if not all(checks.values()):
        print()
        proceed = input(f"{Colors.WARNING}Some prerequisites missing. Continue anyway? (y/n):{Colors.ENDC} ")
        if proceed.lower() != 'y':
            sys.exit(1)

    # Ready message
    print()
    print_success("Agent ready! Type your queries below.")
    print(f"Type {Colors.BOLD}help{Colors.ENDC} for commands or {Colors.BOLD}quit{Colors.ENDC} to exit.")

    show_trace = False
    query_count = 0

    # Main loop
    while True:
        try:
            # Prompt
            prompt_text = f"\n{Colors.BOLD}{Colors.OKBLUE}[{query_count}]>{Colors.ENDC} "
            query = input(prompt_text).strip()

            if not query:
                continue

            # Handle commands
            if query.lower() in ['quit', 'exit', 'q']:
                print(f"\n{Colors.OKGREEN}Goodbye!{Colors.ENDC}\n")
                break

            if query.lower() == 'help':
                show_help()
                continue

            if query.lower() == 'trace':
                show_trace = not show_trace
                status = f"{Colors.OKGREEN}ON{Colors.ENDC}" if show_trace else f"{Colors.FAIL}OFF{Colors.ENDC}"
                print(f"Trace display: {status}")
                continue

            if query.lower() == 'graph':
                print(get_graph_visualization())
                continue

            if query.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                continue

            # Run agent
            print(f"\n{Colors.OKCYAN}Processing...{Colors.ENDC}")

            result = run_agent(query, verbose=False)
            display_result(result, show_trace=show_trace)

            query_count += 1

        except KeyboardInterrupt:
            print(f"\n\n{Colors.WARNING}Interrupted.{Colors.ENDC} Type {Colors.BOLD}quit{Colors.ENDC} to exit or press Ctrl+C again.\n")
            try:
                # Give user a chance to quit gracefully
                sys.stdin.readline()
            except KeyboardInterrupt:
                print(f"\n{Colors.OKGREEN}Goodbye!{Colors.ENDC}\n")
                break

        except Exception as e:
            print()
            print_error(f"Error: {e}")
            print(f"\nIf this persists, check that:")
            print(f"  1. API server is running: {Colors.BOLD}python start_api_server.py{Colors.ENDC}")
            print(f"  2. RAG is initialized: {Colors.BOLD}python demo/demo_rag.py{Colors.ENDC}")
            print(f"  3. OpenAI API key is valid")
            print()

            # Show traceback in debug mode
            if os.getenv("DEBUG"):
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
