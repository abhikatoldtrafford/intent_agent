"""
LangGraph Workflow Definition

Defines the complete agent workflow as a state graph with nodes and edges.

Workflow:
    START
      ↓
    classify_intent
      ↓
    select_tools
      ↓
    execute_tools
      ↓
    aggregate_results
      ↓
    perform_inference
      ↓
    check_feedback
      ↓ (conditional)
    [if feedback_needed=True]  →  select_tools (retry)
    [if feedback_needed=False] →  format_response
      ↓
    END
"""

from typing import Literal
from langgraph.graph import StateGraph, END
from agent.state import AgentState, create_initial_state
from agent.nodes import (
    classify_intent,
    select_tools,
    execute_tools,
    aggregate_results,
    perform_inference,
    check_feedback,
    format_response
)


def should_retry(state: AgentState) -> Literal["retry", "respond"]:
    """
    Conditional edge function to determine if we should retry or respond.

    If feedback_needed is True, retry by going back to tool selection.
    Otherwise, proceed to format response.

    Args:
        state: Current agent state

    Returns:
        "retry" if feedback needed, "respond" otherwise
    """
    feedback_needed = state.get("feedback_needed", False)

    if feedback_needed:
        # Check if we have a clarification question
        if state.get("clarification_question"):
            # In a full implementation, we'd ask the user here
            # For now, we'll just proceed with what we have
            return "respond"
        else:
            # Retry with different approach
            return "retry"
    else:
        return "respond"


def create_agent_graph() -> StateGraph:
    """
    Create the LangGraph workflow.

    Returns:
        Compiled StateGraph ready to invoke
    """
    # Initialize graph with state schema
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("select_tools", select_tools)
    workflow.add_node("execute_tools", execute_tools)
    workflow.add_node("aggregate_results", aggregate_results)
    workflow.add_node("perform_inference", perform_inference)
    workflow.add_node("check_feedback", check_feedback)
    workflow.add_node("format_response", format_response)

    # Set entry point
    workflow.set_entry_point("classify_intent")

    # Add edges (transitions between nodes)
    workflow.add_edge("classify_intent", "select_tools")
    workflow.add_edge("select_tools", "execute_tools")
    workflow.add_edge("execute_tools", "aggregate_results")
    workflow.add_edge("aggregate_results", "perform_inference")
    workflow.add_edge("perform_inference", "check_feedback")

    # Conditional edge: retry or respond
    workflow.add_conditional_edges(
        "check_feedback",
        should_retry,
        {
            "retry": "select_tools",      # Go back to tool selection
            "respond": "format_response"  # Proceed to response
        }
    )

    # Final edge to END
    workflow.add_edge("format_response", END)

    # Compile the graph
    return workflow.compile()


# Create the compiled graph (singleton)
agent_graph = create_agent_graph()


def run_agent(query: str, session_id: str = None, verbose: bool = False) -> AgentState:
    """
    Run the agent with a user query.

    Args:
        query: User's question/query
        session_id: Optional session identifier
        verbose: If True, print trace events

    Returns:
        Final AgentState with answer and trace

    Example:
        >>> result = run_agent("What is the latency for api-gateway?")
        >>> print(result["final_answer"])
    """
    # Create initial state
    initial_state = create_initial_state(query, session_id)

    # Run the graph
    final_state = agent_graph.invoke(initial_state)

    # Print trace if verbose
    if verbose:
        print("\n" + "="*80)
        print("AGENT EXECUTION TRACE")
        print("="*80)

        trace_events = final_state.get("trace", [])
        for event in trace_events:
            timestamp = event.get("timestamp", "")
            node = event.get("node", "")
            event_type = event.get("event_type", "")
            data = event.get("data", {})

            print(f"[{timestamp}] {node} - {event_type}")
            if data:
                for key, value in data.items():
                    print(f"  {key}: {value}")

        # Print Orchestration Decisions
        orchestration_log = final_state.get("orchestration_log", [])
        if orchestration_log:
            print("\n" + "="*80)
            print("🎯 ORCHESTRATION DECISIONS")
            print("="*80)
            for i, decision in enumerate(orchestration_log, 1):
                stage = decision.get('stage', 'unknown')
                intent = decision.get('intent', 'unknown')
                decision_text = decision.get('decision', 'N/A')
                reasoning = decision.get('reasoning', 'No reasoning')
                retry_iter = decision.get('retry_iteration', 0)

                badge = "🟢" if retry_iter == 0 else "🟡" if retry_iter == 1 else "🔴"
                print(f"\n{badge} Decision {i} (Iteration {retry_iter}):")
                print(f"  Stage: {stage}")
                print(f"  Intent: {intent}")
                print(f"  Decision: {decision_text}")
                print(f"  Reasoning: {reasoning}")

        # Print Feedback Loop Iterations
        feedback_iterations = final_state.get("feedback_iterations", [])
        if feedback_iterations:
            print("\n" + "="*80)
            print("🔁 FEEDBACK LOOP ITERATIONS")
            print("="*80)
            for i, iteration in enumerate(feedback_iterations, 1):
                iter_num = iteration.get('iteration', i)
                reason = iteration.get('reason', 'unknown')
                confidence = iteration.get('confidence_at_retry', 0)
                fallback_tools = iteration.get('fallback_tools', [])

                conf_badge = "🟢" if confidence >= 0.8 else "🟡" if confidence >= 0.6 else "🔴"
                print(f"\n🔁 Retry Iteration {iter_num}:")
                print(f"  Reason: {reason}")
                print(f"  Confidence: {conf_badge} {confidence:.2f}")
                print(f"  Fallback Tools: {', '.join(fallback_tools) if fallback_tools else 'None'}")

        print("="*80 + "\n")

    return final_state


def stream_agent(query: str, session_id: str = None):
    """
    Stream the agent execution with intermediate states.

    Args:
        query: User's question/query
        session_id: Optional session identifier

    Yields:
        AgentState after each node execution

    Example:
        >>> for state in stream_agent("Show me error metrics"):
        >>>     print(f"Node: {state.get('current_node')}")
        >>>     print(f"Confidence: {state.get('confidence')}")
    """
    initial_state = create_initial_state(query, session_id)

    # Stream the graph execution
    for state in agent_graph.stream(initial_state):
        yield state


async def arun_agent(query: str, session_id: str = None, verbose: bool = False) -> AgentState:
    """
    Async version of run_agent.

    Args:
        query: User's question/query
        session_id: Optional session identifier
        verbose: If True, print trace events

    Returns:
        Final AgentState with answer and trace

    Example:
        >>> result = await arun_agent("What is the latency for api-gateway?")
        >>> print(result["final_answer"])
    """
    initial_state = create_initial_state(query, session_id)

    # Run the graph asynchronously
    final_state = await agent_graph.ainvoke(initial_state)

    if verbose:
        print("\n" + "="*80)
        print("AGENT EXECUTION TRACE")
        print("="*80)

        trace_events = final_state.get("trace", [])
        for event in trace_events:
            timestamp = event.get("timestamp", "")
            node = event.get("node", "")
            event_type = event.get("event_type", "")
            data = event.get("data", {})

            print(f"[{timestamp}] {node} - {event_type}")
            if data:
                for key, value in data.items():
                    print(f"  {key}: {value}")
        print("="*80 + "\n")

    return final_state


def get_graph_visualization() -> str:
    """
    Get a visual representation of the graph.

    Returns:
        ASCII diagram of the workflow
    """
    return """
Agent Workflow Graph:

    ┌─────────────────┐
    │     START       │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ classify_intent │  ← Determine query intent
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  select_tools   │  ← Choose appropriate tools
    └────────┬────────┘
             │         ◄────────────────┐
             ▼                          │
    ┌─────────────────┐                │
    │ execute_tools   │  ← Run tools    │ (retry loop)
    └────────┬────────┘                │
             │                          │
             ▼                          │
    ┌─────────────────┐                │
    │aggregate_results│  ← Combine data │
    └────────┬────────┘                │
             │                          │
             ▼                          │
    ┌─────────────────┐                │
    │perform_inference│  ← Analyze      │
    └────────┬────────┘                │
             │                          │
             ▼                          │
    ┌─────────────────┐                │
    │ check_feedback  │  ← Confidence?  │
    └────────┬────────┘                │
             │                          │
             ▼                          │
         (decision)                     │
          /      \\                      │
   feedback?    no feedback            │
        /           \\                   │
      yes            no                 │
       │              │                 │
       └──────────────┘                 │
             │                          │
             ▼                          │
    ┌─────────────────┐                │
    │ format_response │  ← Create answer
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │      END        │
    └─────────────────┘

🎯 ORCHESTRATION FEATURES:

  ORCHESTRATION LOGGING (select_tools):
    - Logs every tool selection decision with full reasoning
    - Tracks retry iteration (0, 1, 2)
    - Shows WHY tools were chosen, not just WHAT tools
    - Stored in state["orchestration_log"]

  INTELLIGENT FALLBACK (aggregate_results):
    - Detects when API returns empty → suggests SQL database
    - Detects when SQL returns empty → suggests API
    - Prevents wasted retries by suggesting RIGHT alternative
    - Stored in state["fallback_tools_suggested"]

  FEEDBACK LOOP (check_feedback):
    - Evaluates confidence (HIGH ≥0.8, MEDIUM ≥0.6, LOW <0.6)
    - Triggers retry for LOW confidence (max 2 retries)
    - Logs each retry with reason and suggested fallback tools
    - Stored in state["feedback_iterations"]

Nodes:
  1. classify_intent    - Classify query intent using OpenAI
  2. select_tools       - 🎯 Select tools + LOG ORCHESTRATION DECISION
  3. execute_tools      - Execute selected tools and collect outputs
  4. aggregate_results  - 🔄 Combine outputs + SUGGEST FALLBACKS
  5. perform_inference  - Analyze data and make inferences
  6. check_feedback     - 🔁 Evaluate confidence + LOG FEEDBACK ITERATION
  7. format_response    - Create final formatted answer with trace

Edges:
  - Sequential flow through nodes 1-6
  - Conditional edge at check_feedback:
    * If feedback_needed=True → retry (back to select_tools)
    * If feedback_needed=False → respond (to format_response)
  - Max retries: 2 (prevents infinite loops)

State:
  - AgentState (TypedDict) passed through all nodes
  - Updated by each node with new information
  - Final state contains:
    ✅ final_answer
    ✅ orchestration_log
    ✅ feedback_iterations
    ✅ complete trace
"""


if __name__ == "__main__":
    # Quick test
    print(get_graph_visualization())

    print("\nTesting agent with sample query...\n")

    # Test query
    result = run_agent(
        "What is the current latency for api-gateway?",
        verbose=True
    )

    print("\n" + "="*80)
    print("FINAL ANSWER")
    print("="*80)
    print(result["final_answer"])
    print("="*80)
