from __future__ import annotations
from typing import Dict, Any, Optional
import yaml
from nethical.hooks.interfaces import Region

class PolicyEngine:
    def __init__(self, rules: Dict[str, Any], region: Region):
        self.rules = rules
        self.region = region

    @staticmethod
    def load(path: str, region: Region) -> "PolicyEngine":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        # Region-aware overlays
        overlays = (data.get("region_overlays") or {}).get(region.value) or {}
        merged = {**data, **overlays}
        return PolicyEngine(merged, region)

    def evaluate(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        # Placeholder for full DSL interpreter – apply simple “when/action” rules
        outcomes = {"decisions": [], "disclaimers": [], "escalations": []}
        for rule in self.rules.get("rules", []):
            if self._match(rule.get("when", {}), facts):
                action = rule.get("action", {})
                outcomes["decisions"].append(action.get("decision", "RESTRICT"))
                if d := action.get("add_disclaimer"):
                    outcomes["disclaimers"].append(d)
                if action.get("escalate"):
                    outcomes["escalations"].append(rule.get("id"))
        return outcomes

    def _match(self, cond: Dict[str, Any], facts: Dict[str, Any]) -> bool:
        # Very minimal matcher; replace with proper AST evaluator
        any_conds = cond.get("any")
        all_conds = cond.get("all")
        def eval_atom(expr: str) -> bool:
            # expr example: 'manipulation.override_attempt == true'
            try:
                left, op, right = expr.split()
                val = self._get_fact(left, facts)
                if right.lower() in ("true","false"):
                    rhs = right.lower() == "true"
                else:
                    try:
                        rhs = float(right)
                    except:
                        rhs = right.strip('"').strip("'")
                if op == "==": return val == rhs
                if op == ">=": return float(val) >= float(rhs)
                if op == "<=": return float(val) <= float(rhs)
                if op == ">":  return float(val) > float(rhs)
                if op == "<":  return float(val) < float(rhs)
                return False
            except Exception:
                return False
        if any_conds and any(eval_atom(x) for x in any_conds): return True
        if all_conds and all(eval_atom(x) for x in all_conds): return True
        return False

    def _get_fact(self, path: str, facts: Dict[str, Any]):
        cur = facts
        for p in path.split("."):
            if isinstance(cur, dict):
                cur = cur.get(p)
            else:
                return None
        return cur
