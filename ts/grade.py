import json
import re
from collections import Counter

from .data import nname

CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)
STD_RE = re.compile(r"[ ,./\-_*^]")


def parse_output(text):
    text = text or ""
    if "<tool_call>" not in text:
        return {"attempted": False, "parse_ok": True, "calls": []}
    calls, ok = [], True
    for blob in CALL_RE.findall(text):
        try:
            obj = json.loads(blob.strip())
            args = obj.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(obj.get("name"), str) or not isinstance(args, dict):
                raise ValueError
            calls.append({"name": obj["name"], "arguments": args})
        except Exception:
            ok = False
    if not calls:
        ok = False
    return {"attempted": True, "parse_ok": ok, "calls": calls}


def _std(s):
    return STD_RE.sub("", str(s)).lower().replace("'", '"')


def _num(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.strip())
        except ValueError:
            return None
    return None


def scalar_eq(m, a):
    if isinstance(a, bool) or isinstance(m, bool):
        return m == a
    na, nm = _num(a), _num(m)
    if na is not None and nm is not None and (isinstance(a, (int, float)) or isinstance(m, (int, float))):
        return abs(na - nm) <= 1e-6 * max(1.0, abs(na))
    if isinstance(a, str) or isinstance(m, str):
        return _std(m) == _std(a)
    return m == a


def list_eq(m, a):
    return len(m) == len(a) and all(val_eq(x, y) for x, y in zip(m, a))


def val_eq(m, a):
    if isinstance(a, dict):
        if not isinstance(m, dict):
            return False
        for k, av in a.items():
            accs = av if isinstance(av, list) else [av]
            if k not in m:
                if "" in accs or av == "":
                    continue
                return False
            mk = m[k]
            vals = [x for x in accs if x != ""]
            if not vals:
                continue
            if not (any(val_eq(mk, x) for x in vals) or (isinstance(av, list) and isinstance(mk, list) and list_eq(mk, av))):
                return False
        return all(k in a for k in m)
    if isinstance(a, list):
        return isinstance(m, list) and list_eq(m, a)
    if isinstance(m, (list, dict)):
        return False
    return scalar_eq(m, a)


def call_match(call, gold):
    gname, gparams = next(iter(gold.items()))
    if nname(call["name"]) != nname(gname):
        return False
    args = call["arguments"]
    for k, accs in gparams.items():
        accs = accs if isinstance(accs, list) else [accs]
        if k not in args:
            if "" in accs:
                continue
            return False
        vals = [x for x in accs if x != ""]
        if vals and not any(val_eq(args[k], x) for x in vals):
            return False
    return all(k in gparams for k in args)


def calls_match(calls, golds):
    if len(calls) != len(golds):
        return False
    used = [False] * len(golds)

    def rec(i):
        if i == len(calls):
            return True
        for j, g in enumerate(golds):
            if not used[j] and call_match(calls[i], g):
                used[j] = True
                if rec(i + 1):
                    return True
                used[j] = False
        return False

    return rec(0)


def grade(ex, text):
    p = parse_output(text)
    names = {nname(t["name"]) for t in ex["tools"]}
    halluc = sum(nname(c["name"]) not in names for c in p["calls"])
    if not ex["gold"]:
        correct = not p["attempted"]
    else:
        correct = p["attempted"] and p["parse_ok"] and calls_match(p["calls"], ex["gold"])
    return {"attempted": p["attempted"], "parse_ok": p["parse_ok"], "n_calls": len(p["calls"]), "halluc": halluc, "correct": bool(correct)}


def reward(gold, tool_names, text):
    p = parse_output(text)
    if p["attempted"] and not p["parse_ok"]:
        return -0.5
    names = {nname(n) for n in tool_names}
    if not gold:
        r = 0.0 if p["attempted"] else 1.0
    else:
        if not p["attempted"]:
            return 0.0
        gn = Counter(nname(next(iter(g))) for g in gold)
        cn = Counter(nname(c["name"]) for c in p["calls"])
        inter = sum((gn & cn).values())
        f1 = 2 * inter / (sum(gn.values()) + sum(cn.values()))
        r = 0.2 + 0.3 * f1 + 0.5 * float(calls_match(p["calls"], gold))
    if p["calls"]:
        halluc = sum(nname(c["name"]) not in names for c in p["calls"])
        r -= 0.5 * halluc / len(p["calls"])
    return r
