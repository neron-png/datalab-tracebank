"""tau2-bench support: the corpus loader and the per-domain tool registry.

tau2 differs from tau-bench in four ways that matter to lambda.  Each one is a
silent corruption if unhandled, so each is named here:

1. **Result-file shape.**  tau-bench ships a flat list of `EnvRunResult`;
   tau2 ships `{"info": ..., "tasks": [...], "simulations": [...]}` and calls
   the message log `messages`, not `traj`.
2. **Tool-call shape.**  tau-bench nests the call under `"function"`; tau2 puts
   `name` / `arguments` at the top level.  The old parser filtered on
   `c.get("function")`, so on tau2 it returns [] for *every* call and every tool
   call silently becomes a `respond`.
3. **Parallel calls.**  tau-bench has exactly one tool call per assistant
   message (14,285 of 14,285); tau2 has up to 12.  Results are matched back by
   call id, which both benchmarks carry.
4. **Dual control.**  In telecom the *user* also calls tools -- 30 of its 43
   tools are user-side device actions, and in the logged corpus the user issues
   more calls than the agent does.  These are environment activity and must not
   be folded into the single `user` label.

`reward` lives at `reward_info.reward` and is binary 0/1 across all 26 logged
result files, so `success = reward >= 1.0` carries over unchanged.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLTYPES = os.path.join(ROOT, "data", "tau2_tooltypes.json")

# Where the logged tau2 result files live.  Override with TAU2_RESULTS.
TAU2_RESULTS = os.environ.get(
    "TAU2_RESULTS",
    os.path.join(ROOT, "vendor", "tau2-experiments", "data", "tau2", "results", "final"))

DOMAINS = ("airline", "retail", "telecom")


# --------------------------------------------------------------------------- #
# Tool registry
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ToolSpec:
    """Which tools mutate state, which end the conversation, which exist.

    `known` is the whole registered alphabet for the domain.  A name outside it
    is *not* defaulted to "read" -- it is labelled "unknown", so a registry gap
    shows up in the numbers instead of hiding inside the read class.
    """
    domain: str
    write: frozenset
    read: frozenset
    terminate: frozenset
    user_side: frozenset

    @property
    def known(self) -> frozenset:
        return self.write | self.read

    def rw(self, name: str) -> str:
        if name in self.write:
            return "write"
        if name in self.read:
            return "read"
        return "unknown"


# GENERIC tools (`calculate`, `transfer_to_human_agents`) touch no domain state,
# so they group with the reads; `transfer_to_human_agents` is additionally the
# one terminating tool, in all three domains.
_GENERIC_IS_READ = True
TERMINATE = {"transfer_to_human_agents"}

_SPECS: Dict[str, ToolSpec] = {}


def _load_table() -> Dict[str, Dict[str, Dict[str, str]]]:
    if not os.path.exists(TOOLTYPES):
        raise FileNotFoundError(
            f"{TOOLTYPES} missing -- run scripts/extract_tau2_tooltypes.py")
    with open(TOOLTYPES) as f:
        return json.load(f)["domains"]


def spec_for(domain: str) -> ToolSpec:
    """The ToolSpec for a tau2 domain, built from the extracted annotations."""
    if domain in _SPECS:
        return _SPECS[domain]
    table = _load_table()
    if domain not in table:
        raise KeyError(f"no tau2 tool table for domain {domain!r}; "
                       f"have {sorted(table)}")
    tools = table[domain]
    write, read, user = set(), set(), set()
    for name, meta in tools.items():
        tt = meta["type"]
        if tt == "WRITE":
            write.add(name)
        elif tt in ("READ", "THINK") or (tt == "GENERIC" and _GENERIC_IS_READ):
            read.add(name)
        else:
            read.add(name)
        if meta["side"] == "user":
            user.add(name)
    spec = ToolSpec(domain=domain, write=frozenset(write), read=frozenset(read),
                    terminate=frozenset(TERMINATE & (write | read)),
                    user_side=frozenset(user))
    _SPECS[domain] = spec
    return spec


# --------------------------------------------------------------------------- #
# Corpus loading
# --------------------------------------------------------------------------- #

def result_path(fname: str) -> str:
    return fname if os.path.isabs(fname) else os.path.join(TAU2_RESULTS, fname)


def load_simulations(fname: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return (simulations, info) from one tau2 result file."""
    path = result_path(fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing -- set TAU2_RESULTS to the tau2 results dir")
    with open(path) as f:
        blob = json.load(f)
    return blob["simulations"], blob.get("info", {})


def audit_tool_coverage(sims: Iterable[Dict[str, Any]], spec: ToolSpec) -> Dict[str, Any]:
    """Check every tool name in the corpus is registered as read xor write.

    HANDOFF_TAU2.md Sec 3.4 asks for this to fail loudly rather than default.
    The one legitimate way a name can be missing is a *hallucinated* call: in
    telecom the agent sometimes invokes a user-side tool it does not hold, and
    the environment answers "Tool 'x' not found".  Those are agent behaviour,
    not a registry gap, so the audit tolerates a name only when every one of its
    invocations errored.  A name that ever executed cleanly and is not in the
    table is a real gap and raises.
    """
    seen: Dict[str, Dict[str, int]] = {}
    for s in sims:
        pend: Dict[str, str] = {}
        for m in s.get("messages", []):
            for c in (m.get("tool_calls") or []):
                nm = c.get("name")
                pend[c.get("id")] = nm
                d = seen.setdefault(nm, {"calls": 0, "ok": 0, "err": 0})
                d["calls"] += 1
            if m.get("role") == "tool":
                nm = pend.get(m.get("id"))
                if nm is None:
                    continue
                seen[nm]["err" if m.get("error") else "ok"] += 1
    unknown = {n: v for n, v in seen.items() if spec.rw(n) == "unknown"}
    gaps = {n: v for n, v in unknown.items() if v["ok"] > 0}
    if gaps:
        raise AssertionError(
            f"[{spec.domain}] tool names outside the read/write table that "
            f"executed successfully -- the registry is stale, not a "
            f"hallucination: {gaps}")
    return {
        "n_tool_names": len(seen),
        "registered": sorted(n for n in seen if spec.rw(n) != "unknown"),
        "hallucinated": {n: v["calls"] for n, v in sorted(unknown.items())},
        "n_write": sum(1 for n in seen if spec.rw(n) == "write"),
        "n_read": sum(1 for n in seen if spec.rw(n) == "read"),
    }
