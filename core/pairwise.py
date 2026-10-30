"""Every dub against every other dub. Winner per metric uses the tie tolerances."""
from itertools import combinations


def better(key: str, a, b, th: dict) -> str:
    """Return 'a', 'b' or 'tie' for one metric."""
    if a is None or b is None:
        return "n/a"
    m = th["metrics"][key]
    tol = th["tie_tolerance"].get(key, 0)
    d = m["direction"]
    if d == "band":
        mid = (m["low"] + m["high"]) / 2
        a2, b2 = abs(a - mid), abs(b - mid)
        lower_better = True
        a, b = a2, b2
    elif d in ("max", "max_abs"):
        lower_better = True
        if d == "max_abs":
            a, b = abs(a), abs(b)
    else:
        lower_better = False
    if abs(a - b) <= tol:
        return "tie"
    if lower_better:
        return "a" if a < b else "b"
    return "a" if a > b else "b"


def compare_all(outputs: dict[str, dict], th: dict) -> dict:
    """outputs: output_id -> metric values. Returns pair results and a win matrix."""
    ids = list(outputs)
    pairs = {}
    wins = {i: {j: 0 for j in ids} for i in ids}
    for a, b in combinations(ids, 2):
        per_metric = {}
        score_a = score_b = 0
        for k in th["metrics"]:
            r = better(k, outputs[a].get(k), outputs[b].get(k), th)
            if r == "a":
                score_a += 1
                wins[a][b] += 1
                per_metric[k] = a
            elif r == "b":
                score_b += 1
                wins[b][a] += 1
                per_metric[k] = b
            else:
                per_metric[k] = r
        pairs[f"{a} vs {b}"] = {"a": a, "b": b, "wins_a": score_a, "wins_b": score_b,
                                "per_metric": per_metric}
    return {"pairs": pairs, "win_matrix": wins}
