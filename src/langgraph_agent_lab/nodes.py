"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, ApprovalDecision, make_event


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── implement ALL nodes below ────────────────────────


class _ClassificationResult(BaseModel):
    route: Literal["simple", "tool", "missing_info", "risky", "error"] = Field(
        description="The single best-matching intent route for the query."
    )


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM.

    *** MUST use a real LLM call — keyword-only heuristics will lose points. ***

    Use .with_structured_output() or equivalent to get reliable enum classification.
    The LLM should classify into one of: simple, tool, missing_info, risky, error.

    Hints:
    - See llm.py for the get_llm() helper
    - Use Pydantic model or TypedDict with .with_structured_output()
    - Set risk_level to "high" for risky routes, "low" otherwise
    - Priority guide: risky > tool > missing_info > error > simple

    Return: {"route": str, "risk_level": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    llm = get_llm().with_structured_output(_ClassificationResult)
    prompt = (
        "You are an intent classifier for a support-ticket agent. Classify the user query "
        "into exactly one route.\n\n"
        "Routes, in priority order (check top to bottom, stop at the first match):\n"
        "1. risky — the user is asking us to PERFORM an action with real side effects on their "
        "account or money: refunds, deletions, cancellations, sending emails on their behalf\n"
        "2. tool — the user wants us to LOOK UP information we need a system/database for: "
        "order status, tracking, account lookups, search queries\n"
        "3. missing_info — the query is too vague or incomplete to know what the user wants "
        "(e.g. 'can you fix it?', 'help me' with no detail)\n"
        "4. error — the user is reporting a system failure: timeout, crash, service unavailable\n"
        "5. simple — a general how-to or informational question that we can answer directly "
        "from general knowledge, with no lookup and no side-effecting action\n\n"
        "Examples:\n"
        '- "How do I reset my password?" → simple (how-to question, no lookup, no action)\n'
        '- "Please lookup order status for order 12345" → tool (needs a database lookup)\n'
        '- "Refund this customer and send confirmation email" → risky (side-effecting action)\n'
        '- "Can you fix it?" → missing_info (no detail about what "it" is)\n'
        '- "Timeout failure while processing request" → error (system failure report)\n\n'
        f"User query: {query}\n\n"
        "Pick exactly one route."
    )
    result: _ClassificationResult = llm.invoke(prompt)
    route = result.route.strip().lower()
    risk_level = "high" if route == "risky" else "low"
    return {
        "route": route,
        "risk_level": risk_level,
        "events": [
            make_event("classify", "completed", f"classified as '{route}'", risk_level=risk_level)
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call.

    Simulate transient failures for error-route scenarios to test retry loops.

    Requirements:
    - Read current attempt count from state
    - If route is "error" and attempt < 2: return error result (string containing "ERROR")
    - Otherwise: return a mock success result string
    - Append result to tool_results list

    Return: {"tool_results": [result_string], "events": [make_event(...)]}
    """
    attempt = state.get("attempt", 0)
    route = state.get("route", "")
    if route == "error" and attempt < 2:
        result = f"ERROR: transient tool failure on attempt {attempt}"
        return {
            "tool_results": [result],
            "events": [make_event("tool", "error", result, attempt=attempt)],
        }
    query = state.get("query", "")
    result = f"tool_success: resolved lookup for '{query[:60]}'"
    return {
        "tool_results": [result],
        "events": [make_event("tool", "completed", result, attempt=attempt)],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Check whether the latest tool result is satisfactory or needs retry.

    SHOULD use LLM-as-judge for bonus points. Heuristic (e.g., check for "ERROR" substring)
    is acceptable for base score.

    Requirements:
    - Read the latest entry from tool_results
    - Set evaluation_result to "needs_retry" or "success"
    - This field drives route_after_evaluate conditional edge

    Note: You may need to add 'evaluation_result' to AgentState if not present.

    Return: {"evaluation_result": str, "events": [make_event(...)]}
    """
    tool_results = state.get("tool_results", [])
    latest = tool_results[-1] if tool_results else ""
    evaluation_result = "needs_retry" if "ERROR" in latest else "success"
    return {
        "evaluation_result": evaluation_result,
        "events": [make_event("evaluate", "completed", f"evaluation={evaluation_result}")],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM.

    *** MUST use a real LLM call — hardcoded strings will lose points. ***

    The LLM should generate a helpful response grounded in available context:
    - tool_results (if any)
    - approval decision (if risky route)
    - original query

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    tool_results = state.get("tool_results", [])
    approval = state.get("approval")

    context_parts = [f"User query: {query}"]
    if tool_results:
        context_parts.append("Tool results:\n" + "\n".join(tool_results))
    if approval:
        context_parts.append(f"Approval decision: {approval}")
    context = "\n\n".join(context_parts)

    llm = get_llm()
    prompt = (
        "You are a helpful support agent. Write a concise, friendly response to the user's "
        "query. Ground your answer only in the context below — do not invent facts that are "
        "not present in it.\n\n"
        f"{context}\n\n"
        "Response:"
    )
    response = llm.invoke(prompt)
    final_answer = response.content if hasattr(response, "content") else str(response)
    return {
        "final_answer": final_answer,
        "events": [make_event("answer", "completed", "answer generated")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Generate a specific clarification question based on the vague/incomplete query.

    Note: You may need to add 'pending_question' to AgentState if not present.

    Return: {"pending_question": str, "final_answer": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    llm = get_llm()
    prompt = (
        "The following support-ticket query is too vague or incomplete to act on. "
        "Write one short, specific clarification question that asks the user for the "
        "missing detail needed to help them.\n\n"
        f"Query: {query}\n\n"
        "Clarification question:"
    )
    response = llm.invoke(prompt)
    question = response.content if hasattr(response, "content") else str(response)
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "clarification requested")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval.

    Describe the proposed action and why it requires approval.

    Note: You may need to add 'proposed_action' to AgentState if not present.

    Return: {"proposed_action": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    proposed_action = (
        f"Perform requested action for query: '{query}' (side-effecting, needs approval)"
    )
    return {
        "proposed_action": proposed_action,
        "events": [make_event("risky_action", "completed", proposed_action)],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default behavior: mock approval (approved=True) so tests and CI run offline.
    Extension: if env LANGGRAPH_INTERRUPT=true, use langgraph.types.interrupt() for real HITL.

    Return: {"approval": {"approved": bool, "reviewer": str, "comment": str}, "events": [make_event(...)]}
    """
    decision = ApprovalDecision(
        approved=True, reviewer="mock-reviewer", comment="auto-approved by lab default"
    )
    approval = decision.model_dump()
    return {
        "approval": approval,
        "events": [make_event("approval", "completed", "approval decision recorded", **approval)],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt.

    Increment the attempt counter and log the transient failure.

    Requirements:
    - Read current attempt from state, increment by 1
    - Add an error message to errors list
    - Return updated attempt count

    Return: {"attempt": int, "errors": [str], "events": [make_event(...)]}
    """
    attempt = state.get("attempt", 0) + 1
    tool_results = state.get("tool_results", [])
    latest = tool_results[-1] if tool_results else "unknown failure"
    error_message = f"attempt {attempt}: {latest}"
    return {
        "attempt": attempt,
        "errors": [error_message],
        "events": [make_event("retry", "completed", error_message, attempt=attempt)],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded.

    This is the third layer: retry → fallback → dead letter.
    Log the failure and set a final_answer explaining that the request could not be completed.

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    attempt = state.get("attempt", 0)
    final_answer = (
        "We were unable to complete this request after multiple attempts "
        f"(attempt {attempt}). It has been escalated to a human agent for follow-up."
    )
    return {
        "final_answer": final_answer,
        "events": [make_event("dead_letter", "failed", final_answer, attempt=attempt)],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END.

    Return: {"events": [make_event("finalize", "completed", "workflow finished")]}
    """
    return {"events": [make_event("finalize", "completed", "workflow finished")]}
