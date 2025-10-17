"""
Trace Display Utilities

Provides comprehensive trace visualization for agent execution.
Displays detailed execution flow with timing, decisions, and data.
"""

from typing import Dict, Any, List
from datetime import datetime


def format_duration(ms: float) -> str:
    """Format duration in human-readable format."""
    if ms < 1:
        return f"{ms*1000:.1f}μs"
    elif ms < 1000:
        return f"{ms:.1f}ms"
    else:
        return f"{ms/1000:.2f}s"


def format_timestamp(timestamp_str: str) -> str:
    """Format timestamp for display."""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime("%H:%M:%S.%f")[:-3]
    except:
        return timestamp_str


def display_trace_event(event: Dict[str, Any], index: int):
    """Display a single trace event with details."""
    node = event.get('node', 'unknown')
    event_type = event.get('event_type', 'unknown')
    timestamp = event.get('timestamp', '')
    data = event.get('data', {})

    # Format timestamp
    time_str = format_timestamp(timestamp) if timestamp else "N/A"

    # Event header
    print(f"\n{'─'*80}")
    print(f"{index}. [{node}] {event_type} @ {time_str}")

    # Display relevant data based on node
    if node == 'classify_intent':
        if event_type == 'start':
            query = data.get('query', 'N/A')
            print(f"   📝 Query: \"{query}\"")
        elif event_type == 'complete':
            intent = data.get('intent', 'N/A')
            confidence = data.get('confidence', 0)
            reasoning = data.get('reasoning', 'N/A')
            print(f"   🎯 Intent: {intent}")
            print(f"   📊 Confidence: {confidence:.2f}")
            if reasoning and reasoning != 'N/A':
                print(f"   💭 Reasoning: {reasoning[:150]}...")

    elif node == 'select_tools':
        if event_type == 'complete':
            tools = data.get('tools_selected', [])
            print(f"   🔧 Tools Selected: {', '.join(tools) if tools else 'none'}")

            # Show orchestration decisions
            orchestration = data.get('orchestration_log', [])
            if orchestration:
                latest = orchestration[-1] if orchestration else {}
                decision = latest.get('decision', 'N/A')
                reasoning = latest.get('reasoning', 'N/A')
                print(f"   🧠 Decision: {decision}")
                if reasoning and reasoning != 'N/A':
                    print(f"   💡 Reasoning: {reasoning}")

    elif node == 'execute_tools':
        if event_type == 'start':
            tools = data.get('tools_to_execute', [])
            print(f"   ⚙️  Executing: {', '.join(tools) if tools else 'none'}")
        elif event_type == 'complete':
            tools_executed = data.get('tools_executed', [])
            tool_results_count = len(data.get('tool_results', {}))
            print(f"   ✅ Executed: {', '.join(tools_executed) if tools_executed else 'none'}")
            print(f"   📦 Results: {tool_results_count} tool(s) returned data")

            # Show tool errors if any
            tool_errors = data.get('tool_errors', {})
            if tool_errors:
                print(f"   ❌ Errors: {len(tool_errors)} tool(s) failed")
                for tool_name, error in tool_errors.items():
                    print(f"      • {tool_name}: {str(error)[:100]}")

    elif node == 'aggregate_results':
        if event_type == 'complete':
            aggregated_data = data.get('aggregated_data', {})
            findings = data.get('findings', [])
            data_quality = data.get('data_quality', {})

            print(f"   📊 Aggregated data fields: {', '.join(aggregated_data.keys()) if aggregated_data else 'none'}")
            if findings:
                print(f"   🔍 Findings: {len(findings)}")
                for i, finding in enumerate(findings[:3], 1):
                    print(f"      {i}. {finding[:100]}")

            if data_quality:
                completeness = data_quality.get('completeness', 1.0)
                consistency = data_quality.get('consistency', 1.0)
                print(f"   📈 Data Quality: completeness={completeness:.0%}, consistency={consistency:.0%}")

    elif node == 'perform_inference':
        if event_type == 'complete':
            recommendations = data.get('recommendations', [])
            inference_summary = data.get('inference_summary', '')

            if recommendations:
                print(f"   💡 Recommendations: {len(recommendations)}")
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"      {i}. {rec[:100]}")

            if inference_summary:
                print(f"   🧠 Inference: {inference_summary[:150]}...")

    elif node == 'check_feedback':
        if event_type == 'complete':
            feedback_needed = data.get('feedback_needed', False)
            retry_count = data.get('retry_count', 0)
            confidence = data.get('confidence', 0)
            retry_reason = data.get('retry_reason', '')

            print(f"   🔄 Feedback Needed: {feedback_needed}")
            print(f"   🔁 Retry Count: {retry_count}")
            print(f"   📊 Confidence: {confidence:.2f}")

            if retry_reason:
                print(f"   💭 Retry Reason: {retry_reason}")

            # Show feedback iterations
            feedback_iterations = data.get('feedback_iterations', [])
            if feedback_iterations:
                print(f"   📝 Feedback Iterations: {len(feedback_iterations)}")
                for i, iteration in enumerate(feedback_iterations, 1):
                    reason = iteration.get('reason', 'N/A')
                    action = iteration.get('action', 'N/A')
                    print(f"      {i}. {reason} → {action}")

    elif node == 'format_response':
        if event_type == 'complete':
            final_answer = data.get('final_answer', '')
            answer_preview = final_answer[:150] + "..." if len(final_answer) > 150 else final_answer
            print(f"   📝 Final Answer: {answer_preview}")

    # Show duration if available
    duration = event.get('duration_ms')
    if duration is not None:
        print(f"   ⏱️  Duration: {format_duration(duration)}")


def display_execution_trace(result: Dict[str, Any], show_all: bool = True):
    """
    Display complete execution trace from agent result.

    Args:
        result: Agent execution result containing trace data
        show_all: If True, show all events; if False, only show key events
    """
    trace_events = result.get('trace', [])

    if not trace_events:
        print("\n⚠️  No trace events found in result!")
        return

    print("\n" + "="*80)
    print("EXECUTION TRACE")
    print("="*80)

    # Filter events if not showing all
    if not show_all:
        # Only show 'complete' events for key nodes
        key_nodes = ['classify_intent', 'select_tools', 'execute_tools',
                     'aggregate_results', 'perform_inference', 'check_feedback',
                     'format_response']
        trace_events = [
            e for e in trace_events
            if e.get('node') in key_nodes and e.get('event_type') == 'complete'
        ]

    # Display each event
    for i, event in enumerate(trace_events, 1):
        display_trace_event(event, i)

    print("\n" + "="*80)

    # Summary statistics
    print("\nTRACE SUMMARY:")
    nodes_visited = set(e.get('node') for e in result.get('trace', []))
    print(f"  • Total events: {len(result.get('trace', []))}")
    print(f"  • Nodes visited: {len(nodes_visited)}")
    print(f"  • Total duration: {result.get('total_duration_ms', 0):.0f}ms")

    # Node durations
    node_durations = result.get('node_durations', {})
    if node_durations:
        print(f"\n  Node Execution Times:")
        for node, duration in sorted(node_durations.items(), key=lambda x: x[1], reverse=True):
            print(f"    • {node}: {format_duration(duration)}")

    # Orchestration decisions
    orchestration_log = result.get('orchestration_log', [])
    if orchestration_log:
        print(f"\n  Orchestration Decisions: {len(orchestration_log)}")
        for i, decision in enumerate(orchestration_log, 1):
            stage = decision.get('stage', 'N/A')
            dec = decision.get('decision', 'N/A')
            iteration = decision.get('retry_iteration', 0)
            print(f"    {i}. [{stage}] {dec} (iteration {iteration})")

    # Feedback iterations
    feedback_iterations = result.get('feedback_iterations', [])
    if feedback_iterations:
        print(f"\n  Feedback Loop Iterations: {len(feedback_iterations)}")
        for i, iteration in enumerate(feedback_iterations, 1):
            reason = iteration.get('reason', 'N/A')
            action = iteration.get('action', 'N/A')
            print(f"    {i}. {reason} → {action}")

    print("="*80 + "\n")


def display_trace_summary(result: Dict[str, Any]):
    """Display a compact summary of execution trace."""
    print("\n" + "="*80)
    print("TRACE SUMMARY")
    print("="*80)

    # Basic info
    query = result.get('query', 'N/A')
    intent = result.get('intent', 'N/A')
    confidence = result.get('confidence', 0)
    total_ms = result.get('total_duration_ms', 0)

    print(f"\n📝 Query: {query}")
    print(f"🎯 Intent: {intent} (confidence: {confidence:.2f})")
    print(f"⏱️  Total Time: {format_duration(total_ms)}")

    # Tools
    tools_executed = result.get('tools_executed', [])
    if tools_executed:
        print(f"🔧 Tools Used: {', '.join(tools_executed)}")
    else:
        print(f"🔧 Tools Used: none")

    # Trace events
    trace_events = result.get('trace', [])
    nodes_visited = set(e.get('node') for e in trace_events)
    print(f"📊 Events: {len(trace_events)} events across {len(nodes_visited)} nodes")

    # Orchestration
    orchestration_log = result.get('orchestration_log', [])
    if orchestration_log:
        print(f"🧠 Orchestration: {len(orchestration_log)} decision(s)")

    # Feedback
    feedback_iterations = result.get('feedback_iterations', [])
    retry_count = result.get('retry_count', 0)
    if feedback_iterations:
        print(f"🔄 Feedback: {len(feedback_iterations)} iteration(s), {retry_count} retries")

    # Data quality
    data_quality = result.get('data_quality', {})
    if data_quality:
        completeness = data_quality.get('completeness', 1.0)
        consistency = data_quality.get('consistency', 1.0)
        print(f"📈 Data Quality: {completeness:.0%} complete, {consistency:.0%} consistent")

    # Findings and recommendations
    findings = result.get('findings', [])
    recommendations = result.get('recommendations', [])
    if findings:
        print(f"🔍 Findings: {len(findings)}")
    if recommendations:
        print(f"💡 Recommendations: {len(recommendations)}")

    print("="*80 + "\n")


def display_orchestration_log(result: Dict[str, Any]):
    """Display detailed orchestration decision log."""
    orchestration_log = result.get('orchestration_log', [])

    if not orchestration_log:
        print("\n⚠️  No orchestration decisions found!")
        return

    print("\n" + "="*80)
    print("ORCHESTRATION DECISION LOG")
    print("="*80)

    for i, decision in enumerate(orchestration_log, 1):
        stage = decision.get('stage', 'N/A')
        intent = decision.get('intent', 'N/A')
        dec = decision.get('decision', 'N/A')
        reasoning = decision.get('reasoning', 'N/A')
        iteration = decision.get('retry_iteration', 0)
        timestamp = decision.get('timestamp', 'N/A')

        print(f"\n{i}. Decision at {stage} (iteration {iteration})")
        print(f"   🎯 Intent: {intent}")
        print(f"   🧠 Decision: {dec}")
        print(f"   💡 Reasoning: {reasoning}")
        print(f"   ⏰ Timestamp: {format_timestamp(timestamp)}")

    print("\n" + "="*80 + "\n")


def display_feedback_iterations(result: Dict[str, Any]):
    """Display detailed feedback iteration log."""
    feedback_iterations = result.get('feedback_iterations', [])

    if not feedback_iterations:
        print("\n✅ No feedback iterations (query completed successfully on first try)")
        return

    print("\n" + "="*80)
    print("FEEDBACK LOOP ITERATIONS")
    print("="*80)

    for i, iteration in enumerate(feedback_iterations, 1):
        iteration_num = iteration.get('iteration', i)
        reason = iteration.get('reason', 'N/A')
        action = iteration.get('action', 'N/A')
        confidence = iteration.get('confidence_at_feedback', 0)
        retry_confidence = iteration.get('confidence_at_retry', 0)
        fallback_tools = iteration.get('fallback_tools', [])
        timestamp = iteration.get('timestamp', 'N/A')

        print(f"\n🔄 Iteration {iteration_num}")
        print(f"   📊 Confidence: {confidence:.2f}")
        print(f"   💭 Reason: {reason}")
        print(f"   ⚡ Action: {action}")

        if fallback_tools:
            print(f"   🔧 Fallback Tools: {', '.join(fallback_tools)}")

        if retry_confidence > 0:
            print(f"   🔁 Retry Confidence: {retry_confidence:.2f}")

        print(f"   ⏰ Timestamp: {format_timestamp(timestamp)}")

    print("\n" + "="*80 + "\n")
