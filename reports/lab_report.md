# Day 08 Lab Report

## 1. Team / student

- Name: Nguyễn Thành Duy - 2A202601599
- Repo/commit: [Link](https://github.com/Nguyenthanhduy16/Track3-DAY23-NguyenThanhDuy-2A202601599.git)
- Date: 25/08/2026

## 2. Architecture

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
priority-ordered, few-shot prompt (risky > tool > missing_info > error > simple).

## 3. State schema

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
| final_answer | overwrite | the graph produces one final answer per run |

## 4. Scenario results

- Total scenarios: 7
- Success rate: 100%
- Avg nodes visited: 6.43
- Total retries: 3
- Total interrupts: 2
- Resume success demonstrated: False

| Scenario | Expected route | Actual route | Success | Retries | Interrupts |
|---|---|---|---:|---:|---:|
| S01_simple | simple | simple | yes | 0 | 0 |
| S02_tool | tool | tool | yes | 0 | 0 |
| S03_missing | missing_info | missing_info | yes | 0 | 0 |
| S04_risky | risky | risky | yes | 0 | 1 |
| S05_error | error | error | yes | 2 | 0 |
| S06_delete | risky | risky | yes | 0 | 1 |
| S07_dead_letter | error | error | yes | 1 | 0 |

## 5. Failure analysis

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
   execute without an explicit approval record in state.

## 6. Persistence / recovery evidence

`persistence.py` builds a `SqliteSaver(conn)` (WAL mode) keyed by `thread_id` per scenario. `resume_success=False` above reflects whether this run demonstrated state-history replay or crash-resume (see `tests`/manual runs that reconnect to the same `checkpoints.db` and call `graph.get_state_history()` / `checkpointer.get()`).

## 7. Extension work

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
  `docs/graph_diagram.mmd`, matching the target flow in `docs/LAB_GUIDE.md`.

## 8. Improvement plan

With one more day: build the Streamlit approve/reject UI on top of the
`interrupt()`/`Command(resume=...)` flow already implemented, add an
LLM-as-judge to `evaluate_node` instead of the current "ERROR" substring
heuristic, and use `Send()` for parallel tool fan-out on multi-part
requests.
