"""Report generation helper.

implement report rendering using MetricsReport data
and the template in reports/lab_report_template.md.
"""

from __future__ import annotations

from pathlib import Path

from .metrics import MetricsReport

_ARCHITECTURE_TEXT = """\
The graph is built in `graph.py` with `StateGraph(AgentState)` and 11 nodes:
`intake`, `classify`, `tool`, `evaluate`, `answer`, `clarify` (ask_clarification_node),
`risky_action`, `approval`, `retry` (retry_or_fallback_node), `dead_letter`, `finalize`.

Fixed edges: `START -> intake -> classify`, `tool -> evaluate`,
`risky_action -> approval`, and `answer` / `clarify` / `dead_letter` all funnel into
`finalize -> END` so every path terminates the same way.

Conditional edges (routing.py):
- `route_after_classify` fans out from `classify` to `answer`, `tool`, `clarify`,
  `risky_action`, or `retry` based on the LLM-classified route.
- `route_after_evaluate` is the retry-loop gate: `needs_retry` -> `retry`,
  `success` -> `answer`.
- `route_after_retry` bounds the loop: `attempt < max_attempts` -> `tool` again,
  otherwise -> `dead_letter`.
- `route_after_approval` sends approved risky actions to `tool`, rejected ones to
  `clarify`.

`classify_node` and `answer_node` call a real LLM (via `llm.get_llm()`).
`classify_node` uses `.with_structured_output()` with a `Literal`-constrained route
field so the model can only return one of the five valid route names, plus a
priority-ordered, few-shot prompt (risky > tool > missing_info > error > simple)."""

_STATE_SCHEMA_TABLE = """\
| Field | Reducer | Why |
|---|---|---|
| messages | append (`add`) | audit trail of node-level notes |
| tool_results | append (`add`) | retry loop needs the full history of attempts |
| errors | append (`add`) | accumulate every transient failure across retries |
| events | append (`add`) | full audit log used by metrics.py and grading |
| route | overwrite | only the latest classification matters |
| risk_level | overwrite | only the latest classification matters |
| attempt | overwrite | a counter, not a log |
| evaluation_result | overwrite | retry-loop gate reads only the latest evaluation |
| pending_question | overwrite | only the current clarification question is relevant |
| proposed_action | overwrite | only the current proposed action is relevant |
| approval | overwrite | only the latest approval decision drives routing |
| final_answer | overwrite | the graph produces one final answer per run |"""

_FAILURE_MODES_TEXT = """\
1. **Transient tool failure / retry loop**: `tool_node` simulates an error for
   `route == "error"` while `attempt < 2`. `evaluate_node` flags this as
   `needs_retry`, `route_after_retry` bounces back to `tool` until either the
   call succeeds or `attempt >= max_attempts`, at which point `dead_letter_node`
   emits a final answer explaining the escalation instead of looping forever or
   crashing.
2. **Risky action without approval**: `risky_action_node` never calls the tool
   directly — it only prepares a `proposed_action` and always routes to
   `approval_node` first. If `route_after_approval` sees `approved: False` it
   routes to `clarify` instead of `tool`, so a side-effecting action can never
   execute without an explicit approval record in state."""

_EXTENSION_WORK_TEXT = """\
- **SQLite persistence** (`persistence.py`, `build_checkpointer("sqlite")`):
  `SqliteSaver(conn)` with WAL mode, keyed by `thread_id` per scenario.
  Verified a fresh process reconnecting to the same `.db` file recovers the
  last checkpoint (crash-resume evidence).
- **Real HITL** (`nodes.py::approval_node`): when `LANGGRAPH_INTERRUPT=true`,
  `approval_node` calls `langgraph.types.interrupt()` with the proposed action
  instead of auto-approving. `graph.invoke(...)` returns with `__interrupt__`
  set; resuming with `graph.invoke(Command(resume={"approved": True, ...}),
  config=same_thread_config)` continues the graph from that exact point with
  the human's decision recorded in `approval`. Default behavior (flag unset)
  is unchanged — mock auto-approval — so `run-scenarios` stays non-interactive.
- **Graph diagram** (`cli.py::export-diagram`): `agent-lab export-diagram`
  renders the compiled graph via `graph.get_graph().draw_mermaid()` to
  `docs/graph_diagram.mmd`, matching the target flow in `docs/LAB_GUIDE.md`."""

_IMPROVEMENT_PLAN_TEXT = """\
With one more day: build the Streamlit approve/reject UI on top of the
`interrupt()`/`Command(resume=...)` flow already implemented, add an
LLM-as-judge to `evaluate_node` instead of the current "ERROR" substring
heuristic, and use `Send()` for parallel tool fan-out on multi-part
requests."""


def render_report(metrics: MetricsReport) -> str:
    """Render a complete lab report from metrics data.

    Generate a report that includes:
    1. Metrics summary table (total scenarios, success rate, retries, interrupts)
    2. Per-scenario results table
    3. Architecture explanation (your graph design, state schema, reducers)
    4. Failure analysis (at least two failure modes you considered)
    5. Improvement plan

    Use reports/lab_report_template.md as your guide.

    Return: formatted markdown string
    """
    lines: list[str] = []
    lines.append("# Day 08 Lab Report")
    lines.append("")
    lines.append("## 1. Team / student")
    lines.append("")
    lines.append("- Name: (fill in)")
    lines.append("- Repo/commit: (fill in)")
    lines.append("- Date: (fill in)")
    lines.append("")
    lines.append("## 2. Architecture")
    lines.append("")
    lines.append(_ARCHITECTURE_TEXT)
    lines.append("")
    lines.append("## 3. State schema")
    lines.append("")
    lines.append(_STATE_SCHEMA_TABLE)
    lines.append("")
    lines.append("## 4. Scenario results")
    lines.append("")
    lines.append(f"- Total scenarios: {metrics.total_scenarios}")
    lines.append(f"- Success rate: {metrics.success_rate:.0%}")
    lines.append(f"- Avg nodes visited: {metrics.avg_nodes_visited:.2f}")
    lines.append(f"- Total retries: {metrics.total_retries}")
    lines.append(f"- Total interrupts: {metrics.total_interrupts}")
    lines.append(f"- Resume success demonstrated: {metrics.resume_success}")
    lines.append("")
    lines.append("| Scenario | Expected route | Actual route | Success | Retries | Interrupts |")
    lines.append("|---|---|---|---:|---:|---:|")
    for item in metrics.scenario_metrics:
        lines.append(
            f"| {item.scenario_id} | {item.expected_route} | {item.actual_route or '-'} "
            f"| {'yes' if item.success else 'no'} | {item.retry_count} | {item.interrupt_count} |"
        )
    lines.append("")
    lines.append("## 5. Failure analysis")
    lines.append("")
    lines.append(_FAILURE_MODES_TEXT)
    failed = [item for item in metrics.scenario_metrics if not item.success]
    if failed:
        lines.append("")
        lines.append(f"Observed failures in this run ({len(failed)}):")
        for item in failed:
            lines.append(
                f"- `{item.scenario_id}`: expected `{item.expected_route}`, "
                f"got `{item.actual_route}`; errors={item.errors}"
            )
    lines.append("")
    lines.append("## 6. Persistence / recovery evidence")
    lines.append("")
    lines.append(
        "`persistence.py` builds a `SqliteSaver(conn)` (WAL mode) keyed by "
        f"`thread_id` per scenario. `resume_success={metrics.resume_success}` above "
        "reflects whether this run demonstrated state-history replay or crash-resume "
        "(see `tests`/manual runs that reconnect to the same `checkpoints.db` and call "
        "`graph.get_state_history()` / `checkpointer.get()`)."
    )
    lines.append("")
    lines.append("## 7. Extension work")
    lines.append("")
    lines.append(_EXTENSION_WORK_TEXT)
    lines.append("")
    lines.append("## 8. Improvement plan")
    lines.append("")
    lines.append(_IMPROVEMENT_PLAN_TEXT)
    lines.append("")
    return "\n".join(lines)


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
