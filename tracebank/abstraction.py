"""The abstraction function lambda: raw agent message log  ->  trace sigma.

Section 3.2. tau-bench logs are already structured, so lambda is a projection of
the message list onto the activity alphabet A.
"""
from __future__ import annotations

import ast
import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional

from tracebank.events import Event, Trace

# Tools that mutate environment state (everything else is a read / lookup).
WRITE_TOOLS = {
    # retail
    "cancel_pending_order", "exchange_delivered_order_items",
    "modify_pending_order_address", "modify_pending_order_items",
    "modify_pending_order_payment", "modify_user_address",
    "return_delivered_order_items",
    # airline
    "book_reservation", "cancel_reservation", "send_certificate",
    "update_reservation_baggages", "update_reservation_flights",
    "update_reservation_passengers",
    # both
    "transfer_to_human_agents",
}
TERMINATE_TOOLS = {"transfer_to_human_agents"}

# Label constants
RESPOND = "respond"
USER = "user"
END = "end_conversation"
THINK = "think"

GRANULARITIES = ("tool", "tool_status", "rw", "role", "calls", "agent",
                 "tool_result", "status", "rw_status")

# Result-status taxonomy, mined from the 14,285 tool results in the logged
# corpora (see E19).  The patterns cover 99.94% of them; the rest fall to "err".
#
#   ok        a normal result                                       91.8%
#   empty     a well-formed but EMPTY result -- `[]` from a flight    1.7%
#             search.  Not an error, so the older `tool_status`
#             granularity calls this a success; it is really the
#             moment a good run pivots.
#   notfound  a lookup missed (user / order / product / payment)      4.3%
#   policy    the action is forbidden ("non-pending order cannot be   0.8%
#             modified") -- the agent tried something off-policy
#   payment   the money does not add up / balance too low             0.5%
#   invalid   the request is malformed ("should match", "should be")  0.4%
#   avail     the environment said no (no seats, flight not on date)  0.4%
STATUS_OK, STATUS_EMPTY = "ok", "empty"
_EMPTY_BODIES = ("[]", "{}", "null", "None")
_ERR_PATTERNS = (
    ("notfound", ("not found",)),
    ("policy", ("cannot be", "not allowed")),
    ("avail", ("not available", "not enough seats")),
    ("payment", ("payment amount", "gift card balance", "insufficient",
                 "does not add up", "not enough balance")),
    ("invalid", ("should be", "should match")),
)


def result_status(content: str) -> str:
    """Classify one tool result into the status alphabet above.

    An empty string means `think`, which returns nothing and is a no-op rather
    than an empty *result*, so it is "ok".
    """
    s = str(content).strip()
    low = s.lower()
    if not low.startswith("error"):
        return STATUS_EMPTY if s in _EMPTY_BODIES else STATUS_OK
    for name, pats in _ERR_PATTERNS:
        if any(p in low for p in pats):
            return name
    return "err"


# tau2's own status taxonomy.  Re-mined on the 60,142 tool results in the 12
# logged tau2 corpora (HANDOFF_TAU2.md Sec 3.4 requires this -- E19's
# `empty_rate` is built entirely on the taxonomy, so carrying tau-bench's over
# unchanged would make E19 meaningless here).  Coverage 99.97%, against
# tau-bench's 99.94%.  Two differences from the tau-bench taxonomy:
#
#   * tau2 marks errors with an explicit `error` boolean on the tool message,
#     so the "does the text start with Error" heuristic is not needed (it does
#     agree on 1,264 of 1,264 errors, which is a useful cross-check).
#   * `wrongtool` is new and is a tau2 phenomenon: in the dual-control telecom
#     domain the agent invokes a *user-side* device tool it does not hold, and
#     the environment answers "Tool 'x' not found".  Under tau-bench's patterns
#     this would land in `notfound` alongside a genuine failed lookup, which is
#     a different thing entirely -- hence its own class, tested first.
STATUS_WRONGTOOL = "wrongtool"
_TAU2_ERR_PATTERNS = (
    ("wrongtool", ("not found.", "unknown id format or type")),
    ("payment", ("payment amount", "gift card balance", "insufficient",
                 "does not add up", "not enough balance", "awaiting payment",
                 "payment method")),
    ("notfound", ("not found",)),
    ("avail", ("not available", "not enough seats", "must be suspended")),
    ("policy", ("cannot be", "not allowed", "cannot use", "is not permitted")),
    ("invalid", ("should be", "should match", "must be positive",
                 "required positional", "unexpected keyword argument",
                 "does not match", "missing", "invalid", "must be")),
)


def result_status_tau2(content: str, is_error: bool) -> str:
    """Classify one tau2 tool result, using its explicit `error` flag."""
    s = str(content).strip()
    if not is_error:
        return STATUS_EMPTY if s in _EMPTY_BODIES or not s else STATUS_OK
    low = s.lower()
    for name, pats in _TAU2_ERR_PATTERNS:
        if any(p in low for p in pats):
            return name
    return "err"


def _parse_tool_calls(raw: Any) -> List[Dict[str, Any]]:
    """Normalise a tool_calls field from either benchmark to a common shape.

    tau-bench:  {"id": .., "function": {"name": .., "arguments": ..}}
    tau2:       {"id": .., "name": .., "arguments": .., "requestor": ..}

    The old version filtered on `c.get("function")`, which drops *every* tau2
    call -- turning each one into a `respond` without erroring.  Returns a list
    of {id, name, args, requestor}; tau-bench calls are always requestor
    "assistant", which is what it means for a benchmark with no user tools.
    """
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
    if isinstance(raw, dict):
        raw = [raw]
    out: List[Dict[str, Any]] = []
    for c in raw:
        if not isinstance(c, dict):
            continue
        fn = c.get("function")
        if isinstance(fn, dict):                      # tau-bench
            out.append({"id": c.get("id"),
                        "name": fn.get("name", "unknown_tool"),
                        "args": fn.get("arguments"),
                        "requestor": "assistant"})
        elif c.get("name"):                           # tau2
            out.append({"id": c.get("id"), "name": c["name"],
                        "args": c.get("arguments"),
                        "requestor": c.get("requestor") or "assistant"})
    return out


def _parse_args(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            v = json.loads(raw)
            return v if isinstance(v, dict) else {"_": v}
        except json.JSONDecodeError:
            return {"_": raw}
    return {}


def _args_hash(name: str, args: Dict[str, Any]) -> str:
    blob = name + "|" + json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


USER_TOOL = "user_tool"          # role-granularity label for a user-side call


def _rw(name: str, spec: Any = None) -> str:
    """read / write / unknown for one tool name.

    With no spec this is the tau-bench table, unchanged.  With a tau2 ToolSpec
    it is that domain's extracted `@is_tool(ToolType...)` annotations, and a
    name outside the table returns "unknown" rather than silently defaulting to
    "read" -- HANDOFF_TAU2.md Sec 3.4.
    """
    if spec is not None:
        return spec.rw(name)
    if name == THINK:
        return "read"
    return "write" if name in WRITE_TOOLS else "read"


def _relabel(name: str, granularity: str, is_error: bool,
             status: str = STATUS_OK, spec: Any = None,
             by_user: bool = False) -> str:
    """Map a raw event name into the granularity's alphabet.

    `by_user` marks a tool call issued by the *user* simulator, which only
    happens in tau2's dual-control telecom domain.  It is kept distinct in the
    two actor-sensitive granularities (`role`, `rw`/`rw_status`) because the
    whole point of dual control is that the two actors act on the same
    environment; collapsing them would erase the phenomenon.  Airline and
    retail have no user tools, so every label there is unchanged from tau-bench.
    """
    if granularity == "tool":
        return name
    if granularity == "tool_status":
        # the original binary variant, kept unchanged so its null result stays
        # reproducible; `tool_result` below is the richer one
        return f"{name}!err" if is_error else name
    if granularity == "rw":
        if name in (RESPOND, USER, END):
            return name
        base = _rw(name, spec)
        return f"u:{base}" if by_user else base
    if granularity == "role":
        if name in (RESPOND, USER, END):
            return name
        return USER_TOOL if by_user else "tool"
    if granularity == "tool_result":
        if name in (RESPOND, USER, END):
            return name
        return name if status == STATUS_OK else f"{name}!{status}"
    if granularity == "status":
        # drop what was called, keep only how it went
        if name in (RESPOND, USER, END):
            return name
        return status
    if granularity == "rw_status":
        if name in (RESPOND, USER, END):
            return name
        base = _rw(name, spec)
        if by_user:
            base = f"u:{base}"
        return base if status == STATUS_OK else f"{base}!{status}"
    raise ValueError(f"unknown granularity {granularity!r}")


def _is_error(msg: Dict[str, Any], content: str) -> bool:
    """tau2 states it outright; tau-bench leaves it in the text."""
    flag = msg.get("error")
    if isinstance(flag, bool):
        return flag
    return content.strip().lower().startswith("error")


def project(messages: Iterable[Dict[str, Any]], granularity: str = "tool",
            spec: Any = None) -> List[Event]:
    """Project a tau-bench or tau2 message list onto A.

    One event per agent decision, user turn, or user-side tool call:
      assistant + tool_call  -> the tool's name (one event per call: tau-bench
                                never has more than one, tau2 has up to 12)
      assistant + content    -> "respond"
      user + tool_call       -> the tool's name, marked as the user's (tau2
                                telecom only -- its dual-control device actions)
      user turn              -> "user", or "end_conversation" on ###STOP###
      system prompt          -> dropped (it is context, not an activity)
      tool result            -> dropped, but its ok/error status is folded back
                                onto the call that produced it, matched by id.

    Two granularities drop events rather than relabel them:
      "calls"  keeps only tool-call events (drops respond / user /
               end_conversation) -- the bare sequence of things done to the env,
               by either actor.
      "agent"  keeps everything the agent itself emits, i.e. drops the user
               turns AND the user's own tool calls, but keeps "respond".
    Both stay prefix-stable, because the keep/drop test is per event.

    `spec` is a tau2 ToolSpec (tracebank.tau2.spec_for) and selects that
    domain's read/write table and tau2's status taxonomy.  None = tau-bench.
    """
    events: List[Event] = []
    pending: Dict[Any, int] = {}       # call id -> index of its event
    fifo: List[int] = []               # fallback when a call carries no id
    raw_names: List[str] = []          # pre-relabel names, kept in attr
    by_user: List[bool] = []

    def add_call(call: Dict[str, Any], user_side: bool) -> None:
        name = call.get("name") or "unknown_tool"
        args = _parse_args(call.get("args"))
        attr = {
            "raw": name, "kind": USER_TOOL if user_side else "tool",
            "args_hash": _args_hash(name, args),
            "error": False, "status": STATUS_OK,
            "write": _rw(name, spec) == "write",
            "terminal": name in (spec.terminate if spec is not None
                                 else TERMINATE_TOOLS),
        }
        if user_side:                      # absent on tau-bench, whose attr
            attr["by_user"] = True         # dicts stay byte-identical
        events.append(Event(act=name, pos=len(events), attr=attr))
        raw_names.append(name)
        by_user.append(user_side)
        idx = len(events) - 1
        if call.get("id") is not None:
            pending[call["id"]] = idx
        fifo.append(idx)

    for msg in messages:
        role = msg.get("role")
        if role == "system":
            continue
        if role == "tool":
            content = str(msg.get("content", ""))
            cid = msg.get("id", msg.get("tool_call_id"))
            idx = pending.pop(cid, None) if cid is not None else None
            if idx is None and fifo:
                idx = fifo[0]
            if idx is not None:
                fifo[:] = [i for i in fifo if i != idx]
                is_err = _is_error(msg, content)
                e = events[idx]
                e.attr["error"] = is_err
                e.attr["status"] = (result_status_tau2(content, is_err)
                                    if spec is not None else result_status(content))
                e.attr["obs_len"] = len(content)
            continue
        if role == "user":
            calls = _parse_tool_calls(msg.get("tool_calls"))
            if calls:
                for call in calls:
                    add_call(call, user_side=True)
                continue
            content = str(msg.get("content", ""))
            stop = "###STOP###" in content
            name = END if stop else USER
            events.append(Event(act=name, pos=len(events),
                                attr={"raw": name, "kind": "user", "len": len(content)}))
            raw_names.append(name)
            by_user.append(False)
            continue
        if role == "assistant":
            calls = _parse_tool_calls(msg.get("tool_calls"))
            if calls:
                for call in calls:
                    add_call(call, user_side=False)
            else:
                content = msg.get("content")
                content = "" if content in (None, "None") else str(content)
                events.append(Event(act=RESPOND, pos=len(events),
                                    attr={"raw": RESPOND, "kind": "respond", "len": len(content)}))
                raw_names.append(RESPOND)
                by_user.append(False)
            continue
    if granularity in ("calls", "agent"):
        # calls: drop respond + the user turns;  agent: drop the user turns and
        # the user's own tool calls
        drop = (RESPOND, USER, END) if granularity == "calls" else (USER, END)
        keep = [i for i, e in enumerate(events)
                if raw_names[i] not in drop
                and not (granularity == "agent" and by_user[i])]
        return [Event(act=events[i].act, pos=j, ts=events[i].ts, attr=events[i].attr)
                for j, i in enumerate(keep)]
    if granularity != "tool":
        events = [Event(act=_relabel(raw_names[i], granularity,
                                     bool(e.attr.get("error")),
                                     e.attr.get("status", STATUS_OK),
                                     spec=spec, by_user=by_user[i]),
                        pos=i, ts=e.ts, attr=e.attr)
                  for i, e in enumerate(events)]
    return events


def trace_from_run(run: Dict[str, Any], domain: str, model: str,
                   granularity: str = "tool") -> Trace:
    """Build a Trace from one tau-bench EnvRunResult dict."""
    info = run.get("info") or {}
    task = info.get("task") or {}
    events = project(run.get("traj") or [], granularity=granularity)
    rid = f"{model}:{domain}:t{run.get('task_id')}:r{run.get('trial')}"
    return Trace(
        run_id=rid, events=events, reward=float(run.get("reward", 0.0)),
        task_id=run.get("task_id"), trial=run.get("trial"),
        domain=domain, model=model, instruction=task.get("instruction"),
        meta={"user_id": task.get("user_id"),
              "n_gt_actions": len(task.get("actions") or []),
              "granularity": granularity},
    )


def trace_from_simulation(sim: Dict[str, Any], domain: str, model: str,
                          granularity: str = "tool", spec: Any = None,
                          task: Optional[Dict[str, Any]] = None) -> Trace:
    """Build a Trace from one tau2 simulation record.

    The tau2 analogue of `trace_from_run`.  Field map, all verified against the
    logged corpus: traj -> messages, reward -> reward_info.reward (binary 0/1
    in all 26 result files, so `success = reward >= 1.0` is unchanged),
    task_id -> task_id (a *string* here, not an int -- only ever used as a
    grouping key), trial -> trial.
    """
    from tracebank.tau2 import spec_for
    if spec is None:
        spec = spec_for(domain)
    events = project(sim.get("messages") or [], granularity=granularity, spec=spec)
    rid = f"{model}:{domain}:t{sim.get('task_id')}:r{sim.get('trial')}"
    ri = sim.get("reward_info") or {}
    task = task or {}
    instr = ((task.get("user_scenario") or {}).get("instructions") or {})
    return Trace(
        run_id=rid, events=events, reward=float(ri.get("reward", 0.0) or 0.0),
        task_id=sim.get("task_id"), trial=sim.get("trial"),
        domain=domain, model=model,
        instruction=instr.get("task_instructions") if isinstance(instr, dict) else None,
        meta={"granularity": granularity,
              "termination_reason": sim.get("termination_reason"),
              "duration": sim.get("duration"),
              "agent_cost": sim.get("agent_cost"),
              "reward_basis": ri.get("reward_basis"),
              "benchmark": "tau2"},
    )
