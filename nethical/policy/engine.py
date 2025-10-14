from __future__ import annotations

import re
from typing import Dict, Any, Optional, Iterable, Tuple, Union, List, Mapping

import yaml

from nethical.hooks.interfaces import Region


class PolicyError(Exception):
    """Raised when the policy file or evaluation encounters an unrecoverable error."""


class PolicyEngine:
    """
    Region-aware, YAML-driven policy evaluation engine.

    Enhancements:
    - Deep merge region overlays on top of base config
    - Rich boolean condition DSL with nested all/any/not
    - Backwards compatible string atoms like: "manipulation.override_attempt == true"
    - Operators: ==, !=, >, >=, <, <=, in, contains, startswith, endswith, matches (regex), exists, missing
    - List-aware evaluation (any match semantics) and index addressing via path segments (e.g., items[0].field)
    - Rule metadata: id, priority, enabled, halt_on_match
    - Action fields: decision (or effect), add_disclaimer (str | [str]), escalate (bool | str), tags ([str])
    - Outcomes include: decisions, disclaimers, escalations, tags, matched_rules, final_decision
    - Deterministic final_decision with deny-overrides strategy (configurable)
    """

    def __init__(self, rules: Dict[str, Any], region: Region):
        self.rules = rules or {}
        self.region = region

        # Global config knobs with sensible defaults
        defaults = self.rules.get("defaults", {}) or {}
        self.default_decision: str = str(defaults.get("decision", "RESTRICT")).upper()
        self.deny_overrides: bool = bool(defaults.get("deny_overrides", True))
        self.strict: bool = bool(defaults.get("strict", False))  # if True, unknown ops raise

    @staticmethod
    def load(path: str, region: Region) -> "PolicyEngine":
        """
        Load YAML, apply region overlays (deep merge), and instantiate engine.
        File structure example:
          defaults:
            decision: ALLOW
            deny_overrides: true
          rules:
          - id: anti-manipulation
            enabled: true
            priority: 50
            when:
              any:
              - "manipulation.override_attempt == true"
              - { exists: "manipulation.vector" }
            action:
              decision: DENY
              add_disclaimer: "Manipulative content detected"
              escalate: true
              tags: ["manipulation"]
          region_overlays:
            EU:
              defaults:
                decision: RESTRICT
        """
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = yaml.safe_load(f) or {}
            except Exception as e:
                raise PolicyError(f"Failed to parse policy YAML: {e}") from e

        overlays = (data.get("region_overlays") or {}).get(region.value) or {}
        merged = _deep_merge_dicts(dict(data), overlays)
        return PolicyEngine(merged, region)

    def evaluate(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate input facts against rules.

        Returns:
          {
            "decisions": [str],
            "final_decision": str,
            "disclaimers": [str],
            "escalations": [str],
            "tags": [str],
            "matched_rules": [
              {"id": str, "priority": int, "decision": str}
            ]
          }
        """
        # Prepare outcomes
        outcomes = {
            "decisions": [],       # all rule decisions in order of application
            "disclaimers": [],     # unique disclaimers
            "escalations": [],     # rule ids or targets that escalated
            "tags": [],            # collected tags from actions
            "matched_rules": [],   # structured trace of matched rules
        }

        # Facts are augmented with region for convenience
        eval_facts = dict(facts or {})
        eval_facts.setdefault("region", self.region.value)

        rules: List[Mapping[str, Any]] = self.rules.get("rules") or []
        # Sort by priority (higher first), then by defined order to ensure determinism
        rules_sorted = sorted(
            enumerate(rules),
            key=lambda it: (int(it[1].get("priority", 0)), -it[0]),
            reverse=True,
        )
        halted = False

        for _, rule in rules_sorted:
            if halted:
                break
            if not _is_rule_enabled(rule):
                continue

            cond = rule.get("when", {})
            if self._eval_condition(cond, eval_facts):
                action = rule.get("action", {}) or {}
                decision = _normalize_decision(action.get("decision") or action.get("effect") or "RESTRICT")
                outcomes["decisions"].append(decision)

                # Disclaimers: allow str or list[str]
                disclaimers = action.get("add_disclaimer")
                if isinstance(disclaimers, str):
                    _append_unique(outcomes["disclaimers"], disclaimers)
                elif isinstance(disclaimers, (list, tuple, set)):
                    for d in disclaimers:
                        if isinstance(d, str):
                            _append_unique(outcomes["disclaimers"], d)

                # Escalations: bool or string target. If bool true, use rule id; if string, use string.
                esc = action.get("escalate")
                if isinstance(esc, bool) and esc:
                    rid = rule.get("id")
                    if rid:
                        _append_unique(outcomes["escalations"], str(rid))
                elif isinstance(esc, str) and esc.strip():
                    _append_unique(outcomes["escalations"], esc.strip())

                # Tags
                tags = action.get("tags") or []
                if isinstance(tags, str):
                    tags = [tags]
                for t in tags:
                    if isinstance(t, str) and t:
                        _append_unique(outcomes["tags"], t)

                # Record matched rule
                outcomes["matched_rules"].append({
                    "id": str(rule.get("id") or ""),
                    "priority": int(rule.get("priority", 0)),
                    "decision": decision,
                })

                # Allow rules to mutate local evaluation context (non-persistent outside the call)
                if isinstance(action.get("set"), dict):
                    for k, v in action["set"].items():
                        eval_facts[k] = v

                # Halt if requested
                if bool(rule.get("halt_on_match") or action.get("halt")):
                    halted = True

        # Compute final decision
        outcomes["final_decision"] = self._compute_final_decision(outcomes["decisions"])
        return outcomes

    # --------------------------
    # Condition evaluation
    # --------------------------

    def _eval_condition(self, cond: Any, facts: Dict[str, Any]) -> bool:
        """
        Evaluate a condition which can be:
        - dict with 'all', 'any', 'not'
        - string atom: "path op value"
        - atom dicts:
          - {"exists": "path"} | {"missing": "path"}
          - {"op": "==", "left": "path", "right": value}
          - {"==": ["leftPath", value]}  # operator key style
          - {"in": ["leftPath", ["a","b"]]}
          - {"contains": ["leftPath", "needle"]}
          - {"startswith": ["leftPath", "prefix"]}
          - {"endswith": ["leftPath", "suffix"]}
          - {"matches": ["leftPath", "^re$"]} or {"matches": {"left": "p", "pattern": "^re$", "flags": "i"}}
        - list: treated as 'all' of contained conditions
        - bool: returned as-is
        """
        if cond is None:
            return False
        # boolean literal
        if isinstance(cond, bool):
            return cond
        # list: all must be true
        if isinstance(cond, list):
            return all(self._eval_condition(c, facts) for c in cond)
        # string atom
        if isinstance(cond, str):
            return self._eval_string_atom(cond, facts)

        if isinstance(cond, dict):
            # Logical combinators
            if "all" in cond:
                items = cond.get("all") or []
                return all(self._eval_condition(c, facts) for c in items)
            if "any" in cond:
                items = cond.get("any") or []
                return any(self._eval_condition(c, facts) for c in items)
            if "not" in cond:
                return not self._eval_condition(cond.get("not"), facts)

            # Atom dictionary formats
            # exists / missing
            if "exists" in cond:
                path = cond.get("exists")
                return self._path_exists(path, facts)
            if "missing" in cond:
                path = cond.get("missing")
                return not self._path_exists(path, facts)

            # Generic {"op": "...", "left": "...", "right": ...}
            if "op" in cond and "left" in cond:
                return self._eval_op(cond["op"], cond["left"], cond.get("right", None), facts)

            # Operator-as-key mapping e.g., {"==": ["a.b", 1]}
            if len(cond) == 1:
                op_key, args = next(iter(cond.items()))
                if isinstance(args, (list, tuple)):
                    left, right = (args + [None, None])[:2]
                    return self._eval_op(op_key, left, right, facts)

        # Unknown shapes
        if self.strict:
            raise PolicyError(f"Unsupported condition structure: {cond!r}")
        return False

    def _eval_string_atom(self, expr: str, facts: Dict[str, Any]) -> bool:
        """
        Backwards-compatible support for 'path op value' with simple tokenization.
        Supported ops: ==, !=, >=, <=, >, <, in
        """
        try:
            # Attempt to split into three tokens; handle quoted right side
            m = re.match(r'^\s*(\S+)\s*(==|!=|>=|<=|>|<|in)\s*(.+?)\s*$', expr)
            if not m:
                return False
            left_path, op, right_raw = m.group(1), m.group(2), m.group(3)
            right_val = _parse_literal(right_raw)
            return self._eval_op(op, left_path, right_val, facts)
        except Exception:
            if self.strict:
                raise
            return False

    def _eval_op(self, op: str, left_path: str, right_val: Any, facts: Dict[str, Any]) -> bool:
        op = str(op).lower().strip()
        left_val = self._get_fact(left_path, facts)

        # exists / missing could come through here but we already support them separately
        if op in ("==", "equals", "eq"):
            return _eq_any(left_val, right_val)
        if op in ("!=", "ne", "neq"):
            return not _eq_any(left_val, right_val)
        if op in (">", "gt"):
            return _cmp_any(left_val, right_val, lambda a, b: a > b)
        if op in (">=", "ge", "gte"):
            return _cmp_any(left_val, right_val, lambda a, b: a >= b)
        if op in ("<", "lt"):
            return _cmp_any(left_val, right_val, lambda a, b: a < b)
        if op in ("<=", "le", "lte"):
            return _cmp_any(left_val, right_val, lambda a, b: a <= b)
        if op in ("in", "one_of"):
            return _in_any(left_val, right_val)
        if op == "contains":
            return _contains_any(left_val, right_val)
        if op == "startswith":
            return _starts_any(left_val, right_val)
        if op == "endswith":
            return _ends_any(left_val, right_val)
        if op == "matches":
            # right may be pattern string or a mapping with keys: pattern, flags
            pattern, flags = _extract_regex(right_val)
            return _matches_any(left_val, pattern, flags)

        if self.strict:
            raise PolicyError(f"Unknown operator: {op}")
        return False

    def _compute_final_decision(self, decisions: List[str]) -> str:
        """
        Default deny-overrides: if any decision is DENY/RESTRICT, choose that. Else ALLOW if present. Else default.
        """
        if not decisions:
            return _normalize_decision(self.default_decision)
        dnorm = [_normalize_decision(d) for d in decisions]
        if self.deny_overrides:
            for deny_symbol in ("DENY", "RESTRICT", "BLOCK"):
                if deny_symbol in dnorm:
                    return "DENY"
        # Otherwise, prefer ALLOW if present; else last decision as fallback
        if "ALLOW" in dnorm:
            return "ALLOW"
        return dnorm[-1]

    # --------------------------
    # Fact access helpers
    # --------------------------

    def _path_exists(self, path: str, facts: Dict[str, Any]) -> bool:
        try:
            val = self._get_fact(path, facts)
            if isinstance(val, list):
                return any(v is not None for v in val)
            return val is not None
        except Exception:
            return False

    def _get_fact(self, path: str, facts: Dict[str, Any]) -> Any:
        """
        Get nested value from facts using dot notation with optional list index:
          - "a.b.c"
          - "items[0].name"
        Returns None if any segment is missing.

        If intermediate segment resolves to a list but no index provided, the value is returned as-is (list).
        Operators will handle list-aware comparisons (any-match semantics).
        """
        if not path:
            return None
        cur: Any = facts
        for segment in _split_path(path):
            key, idx = _parse_segment(segment)
            if isinstance(cur, Mapping):
                cur = cur.get(key, None)
            else:
                return None
            if idx is not None:
                if isinstance(cur, list):
                    if 0 <= idx < len(cur):
                        cur = cur[idx]
                    else:
                        return None
                else:
                    return None
        return cur


# --------------------------
# Utility functions
# --------------------------

def _deep_merge_dicts(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge overlay into base. Lists are replaced by overlay lists (no deep-merge for lists).
    """
    if not isinstance(base, dict) or not isinstance(overlay, dict):
        return overlay
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def _is_rule_enabled(rule: Mapping[str, Any]) -> bool:
    enabled = rule.get("enabled")
    return enabled is None or bool(enabled)  # default enabled


def _append_unique(arr: List[Any], item: Any) -> None:
    if item not in arr:
        arr.append(item)


def _normalize_decision(decision: Any) -> str:
    if decision is None:
        return "RESTRICT"
    d = str(decision).strip().upper()
    # normalize synonyms
    if d in ("BLOCK", "RESTRICT", "DENY"):
        return "DENY"
    if d in ("ALLOW", "PERMIT"):
        return "ALLOW"
    return d or "RESTRICT"


def _parse_literal(token: str) -> Any:
    """
    Parse a right-hand literal from string expression:
    - quoted strings: "text with spaces" or 'single quoted'
    - booleans: true/false (case-insensitive)
    - null: null / none
    - numbers: int/float
    - fallback: raw string (unquoted)
    """
    s = token.strip()
    # quoted
    if (len(s) >= 2) and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1]
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    # number?
    try:
        if re.match(r"^[+-]?\d+$", s):
            return int(s)
        if re.match(r"^[+-]?\d+\.\d+$", s):
            return float(s)
    except Exception:
        pass
    return s


def _split_path(path: str) -> List[str]:
    # Simple split by dot not inside brackets
    parts: List[str] = []
    buf = []
    bracket = 0
    for ch in path:
        if ch == '.' and bracket == 0:
            parts.append(''.join(buf))
            buf = []
        else:
            if ch == '[':
                bracket += 1
            elif ch == ']':
                bracket = max(0, bracket - 1)
            buf.append(ch)
    if buf:
        parts.append(''.join(buf))
    return parts


def _parse_segment(seg: str) -> Tuple[str, Optional[int]]:
    """
    Parse "key[0]" or "key" -> (key, index or None)
    Wildcards are not supported; index must be integer if provided.
    """
    m = re.match(r"^([^\[\]]+)(?:\[(\d+)\])?$", seg)
    if not m:
        return seg, None
    key = m.group(1)
    idx = int(m.group(2)) if m.group(2) is not None else None
    return key, idx


def _to_list(val: Any) -> List[Any]:
    if isinstance(val, list):
        return val
    return [val]


def _coerce_number(val: Any) -> Optional[float]:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return None
    return None


def _eq_any(left: Any, right: Any) -> bool:
    # If left is list, any element equals right
    if isinstance(left, list):
        return any(_eq_any(el, right) for el in left)
    return left == right


def _cmp_any(left: Any, right: Any, cmp) -> bool:
    # If left is list, any element satisfies comparison
    if isinstance(left, list):
        return any(_cmp_any(el, right, cmp) for el in left)
    a = _coerce_number(left)
    b = _coerce_number(right)
    if a is None or b is None:
        return False
    return cmp(a, b)


def _in_any(left: Any, right: Any) -> bool:
    # left in right-collection or substring check
    if isinstance(left, list):
        return any(_in_any(el, right) for el in left)
    if isinstance(right, (list, tuple, set)):
        return left in right
    if isinstance(right, str) and isinstance(left, str):
        return left in right
    return False


def _contains_any(left: Any, right: Any) -> bool:
    # right is element/substring looked up inside left
    if isinstance(left, list):
        return right in left
    if isinstance(left, str) and isinstance(right, str):
        return right in left
    return False


def _starts_any(left: Any, right: Any) -> bool:
    if isinstance(left, list):
        return any(_starts_any(el, right) for el in left)
    if isinstance(left, str) and isinstance(right, str):
        return left.startswith(right)
    return False


def _ends_any(left: Any, right: Any) -> bool:
    if isinstance(left, list):
        return any(_ends_any(el, right) for el in left)
    if isinstance(left, str) and isinstance(right, str):
        return left.endswith(right)
    return False


def _extract_regex(spec: Any) -> Tuple[str, int]:
    """
    Accepts:
      - pattern string
      - {"pattern": "...", "flags": "i"} or {"left": "...", "pattern": "...", "flags": "im"}
    """
    flags = 0
    if isinstance(spec, str):
        pattern = spec
    elif isinstance(spec, dict):
        pattern = spec.get("pattern") or ""
        flag_str = str(spec.get("flags") or "")
        for ch in flag_str:
            if ch == "i":
                flags |= re.IGNORECASE
            elif ch == "m":
                flags |= re.MULTILINE
            elif ch == "s":
                flags |= re.DOTALL
    else:
        pattern = str(spec or "")
    return pattern, flags


def _matches_any(left: Any, pattern: str, flags: int = 0) -> bool:
    if not pattern:
        return False
    try:
        rx = re.compile(pattern, flags=flags)
    except re.error:
        return False
    if isinstance(left, list):
        return any(_matches_any(el, pattern, flags) for el in left)
    if isinstance(left, str):
        return rx.search(left) is not None
    return False
