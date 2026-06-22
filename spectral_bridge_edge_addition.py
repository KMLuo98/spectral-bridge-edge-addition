"""Spectral bridge edge addition for complex-network resilience.

Single-file implementation of SBIA and centrality baselines.  The input network
is supplied from the command line.  For adjacency matrices the script detects
whether the network is directed and/or weighted automatically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.linalg as la

ALGORITHMS = ["SBIA", "Deg", "Bet", "Clos", "PR"]
COLORS = {"SBIA": "#084081", "Deg": "#2c7fb8", "Bet": "#4292c6", "Clos": "#6baed6", "PR": "#9ecae1"}
MARKERS = {"SBIA": "o", "Deg": "^", "Bet": "s", "Clos": "D", "PR": "v"}


def read_rows(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append([x.strip() for x in line.split(",")] if "," in line else line.split())
    return rows


def try_matrix(rows):
    try:
        a = np.asarray([[float(x) for x in r] for r in rows], dtype=float)
    except ValueError:
        return None
    return a if a.ndim == 2 and a.shape[0] == a.shape[1] and a.shape[0] > 0 else None


def load_network(path: Path, fmt="auto"):
    rows = read_rows(path)
    a = try_matrix(rows)
    labels = None
    source_format = "matrix"
    if fmt == "matrix" or (fmt == "auto" and a is not None):
        if a is None:
            raise ValueError("matrix input must be square numeric data")
        np.fill_diagonal(a, 0.0)
        labels = [str(i) for i in range(a.shape[0])]
    else:
        source_format = "edgelist"
        labels, index, edges = [], {}, []
        def idx(x):
            if x not in index:
                index[x] = len(labels)
                labels.append(x)
            return index[x]
        for r in rows:
            if len(r) < 2:
                raise ValueError(f"bad edge-list row: {r}")
            u, v = idx(r[0]), idx(r[1])
            w = float(r[2]) if len(r) > 2 else 1.0
            edges.append((u, v, w))
        a = np.zeros((len(labels), len(labels)), dtype=float)
        for u, v, w in edges:
            if u != v:
                a[u, v] = w
    directed = not np.allclose(a, a.T, atol=1e-10)
    if source_format == "edgelist" and not directed:
        a = np.maximum(a, a.T)
    weights = a[np.abs(a) > 1e-12]
    weighted = bool(weights.size and not np.allclose(weights, np.ones_like(weights), atol=1e-10))
    return a, labels, directed, weighted, source_format


def laplacian(a):
    return np.diag(a.sum(axis=1)) - a


def candidates(a, directed):
    n = a.shape[0]
    if directed:
        return [(i, j) for i in range(n) for j in range(n) if i != j and a[i, j] == 0]
    return [(i, j) for i in range(n) for j in range(i + 1, n) if a[i, j] == 0]


def add_edge(a, edge, directed, weight):
    i, j = edge
    a[i, j] = weight
    if not directed:
        a[j, i] = weight


def mu_f(a, directed):
    lmat = laplacian(a)
    if not directed:
        vals = np.sort(np.real(la.eigvalsh(lmat)))
        return float(vals[1])
    vals = la.eigvals(lmat)
    vals = [z for z in vals if abs(z) > 1e-8 and z.real > 1e-8]
    return 0.0 if not vals else float(min(vals, key=lambda z: (z.real, abs(z.imag))).real)


def sbia_scores(a, directed):
    lmat = laplacian(a)
    if not directed:
        vals, vecs = la.eigh(lmat)
        v = np.real(vecs[:, np.argsort(vals)[1]])
        return {(i, j): float((v[i] - v[j]) ** 2) for i, j in candidates(a, False)}
    vals, left, right = la.eig(lmat, left=True, right=True)
    valid = [k for k, z in enumerate(vals) if abs(z) > 1e-8 and z.real > 1e-8]
    if not valid:
        return {e: 0.0 for e in candidates(a, True)}
    k = min(valid, key=lambda q: (vals[q].real, abs(vals[q].imag)))
    u, v = left[:, k], right[:, k]
    denom = np.vdot(u, v)
    if abs(denom) < 1e-12:
        denom = 1.0
    return {(i, j): float(np.real(np.conj(u[i]) * (v[i] - v[j]) / denom)) for i, j in candidates(a, True)}


def nx_graph(a, directed):
    g = nx.DiGraph() if directed else nx.Graph()
    g.add_nodes_from(range(a.shape[0]))
    rows, cols = np.where(a > 0)
    for i, j in zip(rows, cols):
        if i == j:
            continue
        if directed:
            g.add_edge(int(i), int(j), weight=float(a[i, j]))
        elif i < j:
            g.add_edge(int(i), int(j), weight=float(a[i, j]))
    return g


def node_scores(a, directed, metric):
    g = nx_graph(a, directed)
    if metric == "Deg":
        if directed:
            scale = max(1, 2 * (len(g) - 1))
            return {n: float(g.in_degree(n) + g.out_degree(n)) / scale for n in g.nodes()}
        return nx.degree_centrality(g)
    if metric == "Bet":
        return nx.betweenness_centrality(g, normalized=True, weight=None)
    if metric == "Clos":
        return nx.closeness_centrality(g)
    if metric == "PR":
        return nx.pagerank(g, alpha=0.85, weight=None, max_iter=2000, tol=1e-12)
    raise ValueError(metric)


def pick(scores):
    return max(scores, key=lambda e: (scores[e], -e[0], -e[1]))


def run_strategy(a0, directed, algorithm, add_weight=1.0, budget=None, variant=None):
    a = a0.copy()
    mu0 = mu_f(a, directed)
    if mu0 <= 1e-12:
        raise RuntimeError("initial network has zero generalized Fiedler value")
    total = len(candidates(a, directed))
    steps = total if budget is None else min(int(budget), total)
    rows = [{"step": 0, "x": 0.0, "mu_f": mu0, "ri": 1.0}]
    edges = []
    for step in range(1, steps + 1):
        cands = candidates(a, directed)
        if not cands:
            break
        if algorithm == "SBIA":
            scores = sbia_scores(a, directed)
        else:
            ns = node_scores(a, directed, algorithm)
            if variant == "sum":
                scores = {(i, j): ns[i] + ns[j] for i, j in cands}
            elif variant == "diff":
                scores = {(i, j): abs(ns[i] - ns[j]) for i, j in cands}
            else:
                raise ValueError("baseline variant must be sum or diff")
        edge = pick(scores)
        add_edge(a, edge, directed, add_weight)
        edges.append(edge)
        mu = mu_f(a, directed)
        denom = total if budget is None else steps
        rows.append({"step": step, "x": step / denom, "mu_f": mu, "ri": mu / mu0})
    frame = pd.DataFrame(rows)
    auc = float(np.trapz(frame["ri"].to_numpy(), frame["x"].to_numpy()))
    return frame, auc, edges


def evaluate(a, labels, directed, budget, add_weight):
    curves, aucs, edge_rows = [], [], []
    curve, auc, edges = run_strategy(a, directed, "SBIA", add_weight, budget)
    curve["algorithm"], curve["variant"] = "SBIA", "spectral"
    curves.append(curve)
    aucs.append({"algorithm": "SBIA", "variant": "spectral", "auc": auc})
    for s, (i, j) in enumerate(edges, 1):
        edge_rows.append({"algorithm": "SBIA", "variant": "spectral", "step": s, "source": labels[i], "target": labels[j]})
    for alg in ["Deg", "Bet", "Clos", "PR"]:
        best = None
        for var in ["sum", "diff"]:
            curve, auc, edges = run_strategy(a, directed, alg, add_weight, budget, var)
            if best is None or auc > best[0]:
                best = (auc, var, curve, edges)
        auc, var, curve, edges = best
        curve["algorithm"], curve["variant"] = alg, var
        curves.append(curve)
        aucs.append({"algorithm": alg, "variant": var, "auc": auc})
        for s, (i, j) in enumerate(edges, 1):
            edge_rows.append({"algorithm": alg, "variant": var, "step": s, "source": labels[i], "target": labels[j]})
    return pd.concat(curves, ignore_index=True), pd.DataFrame(aucs), pd.DataFrame(edge_rows)


def plot_summary(curves, aucs, out):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for alg in ALGORITHMS:
        sub = curves[curves.algorithm == alg]
        axes[0].plot(sub.x, sub.ri, label=alg, color=COLORS[alg], marker=MARKERS[alg], markevery=max(1, len(sub)//10), linewidth=2.2 if alg == "SBIA" else 1.7, markersize=4)
    axes[0].set_xlabel(r"$f_{\mathrm{EA}}$")
    axes[0].set_ylabel("RI")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    ordered = aucs.set_index("algorithm").loc[ALGORITHMS].reset_index()
    bars = axes[1].barh(np.arange(len(ordered)), ordered.auc, color=[COLORS[a] for a in ordered.algorithm])
    axes[1].set_yticks(np.arange(len(ordered)), ordered.algorithm)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("AUC")
    axes[1].grid(axis="x", alpha=0.25)
    for bar, val in zip(bars, ordered.auc):
        axes[1].text(val, bar.get_y() + bar.get_height() / 2, f" {val:.3g}", va="center")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def connected_graph(kind, n, seed):
    for off in range(10000):
        s = seed + off
        if kind == "BA":
            g = nx.barabasi_albert_graph(n, 2, seed=s)
        elif kind == "SW":
            g = nx.watts_strogatz_graph(n, 4, 0.15, seed=s)
        elif kind == "ER":
            g = nx.erdos_renyi_graph(n, 0.10, seed=s)
        else:
            raise ValueError(kind)
        if nx.is_connected(g):
            return g
    raise RuntimeError(kind)


def undirected_adjacency(g, weighted, seed):
    rng = np.random.default_rng(seed)
    a = np.zeros((g.number_of_nodes(), g.number_of_nodes()))
    for i, j in g.edges():
        w = float(rng.uniform(0.5, 1.5)) if weighted else 1.0
        a[i, j] = a[j, i] = w
    return a


def directed_adjacency(g, seed):
    rng = np.random.default_rng(seed)
    n = g.number_of_nodes()
    a = np.zeros((n, n))
    for i in range(n):
        a[i, (i + 1) % n] = 1.0
    for i, j in g.edges():
        if a[i, j] or a[j, i]:
            continue
        if rng.random() < 0.5:
            a[i, j] = 1.0
        else:
            a[j, i] = 1.0
    return a


def write_examples(out_dir):
    specs = [("fig4_undirected_unweighted", False, False), ("fig5_directed_unweighted", True, False), ("fig6_undirected_weighted", False, True)]
    for fig_idx, (folder, directed, weighted) in enumerate(specs):
        folder_path = Path(out_dir) / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        for net_idx, kind in enumerate(["BA", "SW", "ER"]):
            seed = 42 + 1000 * net_idx + 10000 * fig_idx
            g = connected_graph(kind, 40, seed)
            a = directed_adjacency(g, seed + 123) if directed else undirected_adjacency(g, weighted, seed + 456)
            np.savetxt(folder_path / f"{kind}.csv", a, delimiter=",", fmt="%.10g")


def parse_budget(text):
    if text is None or str(text).lower() == "full":
        return None
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("budget must be positive or full")
    return value


def main():
    p = argparse.ArgumentParser(description="Run SBIA and centrality baselines on an input network.")
    p.add_argument("--input", type=Path, help="Adjacency matrix or edge-list file")
    p.add_argument("--format", choices=["auto", "matrix", "edgelist"], default="auto")
    p.add_argument("--output-dir", type=Path, default=Path("sbia_output"))
    p.add_argument("--budget", type=parse_budget, default=None, help="Number of additions, or full")
    p.add_argument("--add-weight", type=float, default=1.0)
    p.add_argument("--write-examples", type=Path, help="Write Fig. 4--6 example inputs and exit")
    args = p.parse_args()
    if args.write_examples:
        write_examples(args.write_examples)
        print(f"Wrote example inputs to {args.write_examples.resolve()}")
        return
    if args.input is None:
        p.error("--input is required unless --write-examples is used")
    a, labels, directed, weighted, source_format = load_network(args.input, args.format)
    curves, aucs, edges = evaluate(a, labels, directed, args.budget, args.add_weight)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curves.to_csv(args.output_dir / "curves.csv", index=False)
    aucs.to_csv(args.output_dir / "auc.csv", index=False)
    edges.to_csv(args.output_dir / "selected_edges.csv", index=False)
    plot_summary(curves, aucs, args.output_dir / "summary.png")
    meta = {"input": str(args.input), "source_format": source_format, "nodes": len(labels), "directed": directed, "weighted": weighted, "budget": args.budget, "add_weight": args.add_weight}
    (args.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Detected network: nodes={len(labels)}, directed={directed}, weighted={weighted}")
    print(aucs.sort_values("auc", ascending=False).to_string(index=False))
    print(f"Wrote outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
