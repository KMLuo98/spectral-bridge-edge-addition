"""Spectral bridge edge addition for complex-network resilience.

This single-file script evaluates the spectral bridge iterative addition (SBIA)
rule and four centrality-based baselines on an input network.  The network is
passed as a command-line argument.  For adjacency-matrix inputs, the script
automatically detects whether the network is directed and/or weighted.

Examples
--------
Run on an adjacency matrix:

    python spectral_bridge_edge_addition.py --input network.txt --output-dir results/network

Run a synthetic example corresponding to Figs. 4--6:

    python spectral_bridge_edge_addition.py --example fig4 --network BA --output-dir results/fig4_BA
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.linalg as la


ALGORITHMS = ["SBIA", "Deg", "Bet", "Clos", "PR"]
COLORS = {
    "SBIA": "#084081",
    "Deg": "#2c7fb8",
    "Bet": "#4292c6",
    "Clos": "#6baed6",
    "PR": "#9ecae1",
}
MARKERS = {"SBIA": "o", "Deg": "^", "Bet": "s", "Clos": "D", "PR": "v"}


@dataclass
class NetworkInput:
    adjacency: np.ndarray
    labels: List[str]
    directed: bool
    weighted: bool
    source_format: str


def _split_fields(line: str) -> List[str]:
    if "," in line:
        return [part.strip() for part in line.split(",") if part.strip()]
    return line.split()


def _as_float_matrix(rows: Sequence[Sequence[str]]) -> Optional[np.ndarray]:
    try:
        matrix = np.asarray([[float(x) for x in row] for row in rows], dtype=float)
    except ValueError:
        return None
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        return None
    return matrix


def _nonzero_weights(adjacency: np.ndarray) -> np.ndarray:
    return adjacency[np.abs(adjacency) > 1e-12]


def detect_weighted(adjacency: np.ndarray) -> bool:
    weights = _nonzero_weights(adjacency)
    if weights.size == 0:
        return False
    return not np.allclose(weights, np.ones_like(weights), atol=1e-10)


def detect_directed(adjacency: np.ndarray) -> bool:
    return not np.allclose(adjacency, adjacency.T, atol=1e-10)


def load_network(path: Path, input_format: str = "auto") -> NetworkInput:
    """Load a network from an adjacency matrix or an edge list.

    Adjacency matrices are interpreted in the usual convention
    ``A[i, j] = weight of edge i -> j``.  Matrix inputs allow exact automatic
    detection of directedness and weights.  Edge-list inputs are intrinsically
    ambiguous when undirected edges are listed once; for unambiguous automatic
    detection, prefer adjacency matrices or include reciprocal directed edges.
    """
    text_lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not text_lines:
        raise ValueError(f"empty network file: {path}")
    rows = [_split_fields(line) for line in text_lines]

    matrix = _as_float_matrix(rows)
    if input_format == "matrix" or (input_format == "auto" and matrix is not None):
        if matrix is None:
            raise ValueError("input was requested as matrix but is not square numeric data")
        np.fill_diagonal(matrix, 0.0)
        labels = [str(i) for i in range(matrix.shape[0])]
        return NetworkInput(
            adjacency=matrix,
            labels=labels,
            directed=detect_directed(matrix),
            weighted=detect_weighted(matrix),
            source_format="matrix",
        )

    if input_format not in {"auto", "edgelist"}:
        raise ValueError(f"unknown input format: {input_format}")
    return load_edgelist(rows)


def load_edgelist(rows: Sequence[Sequence[str]]) -> NetworkInput:
    labels: List[str] = []
    label_to_idx: Dict[str, int] = {}
    edge_rows: List[Tuple[str, str, float]] = []

    def node_index(label: str) -> int:
        if label not in label_to_idx:
            label_to_idx[label] = len(labels)
            labels.append(label)
        return label_to_idx[label]

    for row in rows:
        if len(row) < 2:
            raise ValueError(f"edge-list row has fewer than two fields: {row}")
        weight = float(row[2]) if len(row) >= 3 else 1.0
        edge_rows.append((row[0], row[1], weight))
        node_index(row[0])
        node_index(row[1])

    adjacency = np.zeros((len(labels), len(labels)), dtype=float)
    seen: Dict[Tuple[int, int], float] = {}
    for src_label, dst_label, weight in edge_rows:
        src = label_to_idx[src_label]
        dst = label_to_idx[dst_label]
        if src == dst:
            continue
        adjacency[src, dst] = weight
        seen[(src, dst)] = weight

    directed = False
    for (src, dst), weight in seen.items():
        reverse = seen.get((dst, src))
        if reverse is None or not math.isclose(weight, reverse, rel_tol=1e-10, abs_tol=1e-10):
            directed = True
            break
    if not directed:
        adjacency = np.maximum(adjacency, adjacency.T)

    return NetworkInput(
        adjacency=adjacency,
        labels=labels,
        directed=directed,
        weighted=detect_weighted(adjacency),
        source_format="edgelist",
    )


def laplacian(adjacency: np.ndarray) -> np.ndarray:
    """Return the out-degree Laplacian for A[i, j] = edge i -> j."""
    return np.diag(adjacency.sum(axis=1)) - adjacency


def candidate_edges(adjacency: np.ndarray, directed: bool) -> List[Tuple[int, int]]:
    n = adjacency.shape[0]
    if directed:
        return [(i, j) for i in range(n) for j in range(n) if i != j and adjacency[i, j] == 0]
    return [(i, j) for i in range(n) for j in range(i + 1, n) if adjacency[i, j] == 0]


def add_edge(adjacency: np.ndarray, edge: Tuple[int, int], directed: bool, weight: float) -> None:
    src, dst = edge
    adjacency[src, dst] = weight
    if not directed:
        adjacency[dst, src] = weight


def generalized_fiedler_value(adjacency: np.ndarray, directed: bool) -> float:
    lmat = laplacian(adjacency)
    if not directed:
        vals = la.eigvalsh(lmat)
        vals = np.sort(np.real(vals))
        return float(vals[1])
    vals = la.eigvals(lmat)
    candidates = [z for z in vals if abs(z) > 1e-8 and z.real > 1e-8]
    if not candidates:
        return 0.0
    return float(min(candidates, key=lambda z: (z.real, abs(z.imag))).real)


def sbia_scores(adjacency: np.ndarray, directed: bool) -> Dict[Tuple[int, int], float]:
    lmat = laplacian(adjacency)
    if not directed:
        vals, vecs = la.eigh(lmat)
        order = np.argsort(vals)
        fiedler_vec = np.real(vecs[:, order[1]])
        return {
            (src, dst): float((fiedler_vec[src] - fiedler_vec[dst]) ** 2)
            for src, dst in candidate_edges(adjacency, directed=False)
        }

    vals, left, right = la.eig(lmat, left=True, right=True)
    valid = [idx for idx, val in enumerate(vals) if abs(val) > 1e-8 and val.real > 1e-8]
    if not valid:
        return {edge: 0.0 for edge in candidate_edges(adjacency, directed=True)}
    idx = min(valid, key=lambda k: (vals[k].real, abs(vals[k].imag)))
    u = left[:, idx]
    v = right[:, idx]
    denom = np.vdot(u, v)
    if abs(denom) < 1e-12:
        denom = 1.0
    scores: Dict[Tuple[int, int], float] = {}
    for src, dst in candidate_edges(adjacency, directed=True):
        scores[(src, dst)] = float(np.real(np.conj(u[src]) * (v[src] - v[dst]) / denom))
    return scores


def graph_for_centrality(adjacency: np.ndarray, directed: bool) -> nx.Graph:
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(range(adjacency.shape[0]))
    rows, cols = np.where(adjacency > 0)
    for src, dst in zip(rows, cols):
        if src == dst:
            continue
        if directed:
            graph.add_edge(int(src), int(dst), weight=float(adjacency[src, dst]))
        elif src < dst:
            graph.add_edge(int(src), int(dst), weight=float(adjacency[src, dst]))
    return graph


def node_scores(adjacency: np.ndarray, directed: bool, metric: str) -> Dict[int, float]:
    graph = graph_for_centrality(adjacency, directed)
    if metric == "Deg":
        if directed:
            scale = max(1, 2 * (len(graph) - 1))
            return {
                node: float(graph.in_degree(node) + graph.out_degree(node)) / scale
                for node in graph.nodes()
            }
        return nx.degree_centrality(graph)
    if metric == "Bet":
        return nx.betweenness_centrality(graph, normalized=True, weight=None)
    if metric == "Clos":
        return nx.closeness_centrality(graph)
    if metric == "PR":
        return nx.pagerank(graph, alpha=0.85, weight=None, max_iter=2000, tol=1e-12)
    raise ValueError(f"unknown metric: {metric}")


def select_edge(scores: Dict[Tuple[int, int], float]) -> Tuple[int, int]:
    return max(scores, key=lambda edge: (scores[edge], -edge[0], -edge[1]))


def run_strategy(
    initial_adjacency: np.ndarray,
    *,
    directed: bool,
    add_weight: float,
    budget: Optional[int],
    algorithm: str,
    endpoint_mode: Optional[str] = None,
) -> Tuple[pd.DataFrame, float, List[Tuple[int, int]]]:
    adjacency = initial_adjacency.copy()
    mu0 = generalized_fiedler_value(adjacency, directed)
    if mu0 <= 1e-12:
        raise RuntimeError("initial network has zero generalized Fiedler value")

    total_candidates = len(candidate_edges(adjacency, directed))
    steps = total_candidates if budget is None else min(budget, total_candidates)
    rows = [{"step": 0, "x": 0.0, "mu_f": mu0, "ri": 1.0}]
    selected: List[Tuple[int, int]] = []

    for step in range(1, steps + 1):
        cands = candidate_edges(adjacency, directed)
        if not cands:
            break
        if algorithm == "SBIA":
            scores = sbia_scores(adjacency, directed)
        else:
            scores_by_node = node_scores(adjacency, directed, algorithm)
            if endpoint_mode == "sum":
                scores = {(src, dst): scores_by_node[src] + scores_by_node[dst] for src, dst in cands}
            elif endpoint_mode == "diff":
                scores = {(src, dst): abs(scores_by_node[src] - scores_by_node[dst]) for src, dst in cands}
            else:
                raise ValueError("endpoint_mode must be 'sum' or 'diff' for baseline algorithms")
        edge = select_edge(scores)
        add_edge(adjacency, edge, directed, add_weight)
        selected.append(edge)
        mu = generalized_fiedler_value(adjacency, directed)
        denominator = total_candidates if budget is None else steps
        rows.append({"step": step, "x": step / denominator, "mu_f": mu, "ri": mu / mu0})

    frame = pd.DataFrame(rows)
    auc = float(np.trapz(frame["ri"].to_numpy(), frame["x"].to_numpy()))
    return frame, auc, selected


def evaluate_network(
    network: NetworkInput,
    *,
    budget: Optional[int],
    add_weight: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    curve_frames: List[pd.DataFrame] = []
    auc_rows: List[dict] = []
    edge_rows: List[dict] = []

    sbia_curve, sbia_auc, sbia_edges = run_strategy(
        network.adjacency,
        directed=network.directed,
        add_weight=add_weight,
        budget=budget,
        algorithm="SBIA",
    )
    sbia_curve["algorithm"] = "SBIA"
    sbia_curve["variant"] = "spectral"
    curve_frames.append(sbia_curve)
    auc_rows.append({"algorithm": "SBIA", "variant": "spectral", "auc": sbia_auc})
    for step, (src, dst) in enumerate(sbia_edges, start=1):
        edge_rows.append(
            {
                "algorithm": "SBIA",
                "variant": "spectral",
                "step": step,
                "source": network.labels[src],
                "target": network.labels[dst],
            }
        )

    for algorithm in ["Deg", "Bet", "Clos", "PR"]:
        best = None
        for variant in ["sum", "diff"]:
            curve, auc, edges = run_strategy(
                network.adjacency,
                directed=network.directed,
                add_weight=add_weight,
                budget=budget,
                algorithm=algorithm,
                endpoint_mode=variant,
            )
            candidate = (auc, variant, curve, edges)
            if best is None or candidate[0] > best[0]:
                best = candidate
        assert best is not None
        auc, variant, curve, edges = best
        curve["algorithm"] = algorithm
        curve["variant"] = variant
        curve_frames.append(curve)
        auc_rows.append({"algorithm": algorithm, "variant": variant, "auc": auc})
        for step, (src, dst) in enumerate(edges, start=1):
            edge_rows.append(
                {
                    "algorithm": algorithm,
                    "variant": variant,
                    "step": step,
                    "source": network.labels[src],
                    "target": network.labels[dst],
                }
            )

    curves = pd.concat(curve_frames, ignore_index=True)
    aucs = pd.DataFrame(auc_rows)
    selected_edges = pd.DataFrame(edge_rows)
    return curves, aucs, selected_edges


def plot_curves(curves: pd.DataFrame, aucs: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    ax_curve, ax_bar = axes
    for algorithm in ALGORITHMS:
        sub = curves[curves["algorithm"] == algorithm]
        ax_curve.plot(
            sub["x"],
            sub["ri"],
            label=algorithm,
            color=COLORS[algorithm],
            marker=MARKERS[algorithm],
            markevery=max(1, len(sub) // 10),
            linewidth=2.2 if algorithm == "SBIA" else 1.7,
            markersize=4,
        )
    ax_curve.set_xlabel(r"$f_{\mathrm{EA}}$")
    ax_curve.set_ylabel("RI")
    ax_curve.grid(alpha=0.25)
    ax_curve.legend(frameon=False)

    aucs = aucs.set_index("algorithm").loc[ALGORITHMS].reset_index()
    bars = ax_bar.barh(
        np.arange(len(aucs)),
        aucs["auc"],
        color=[COLORS[algorithm] for algorithm in aucs["algorithm"]],
    )
    ax_bar.set_yticks(np.arange(len(aucs)), aucs["algorithm"])
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("AUC")
    ax_bar.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, aucs["auc"]):
        ax_bar.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.3g}", va="center")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    output_dir: Path,
    network: NetworkInput,
    curves: pd.DataFrame,
    aucs: pd.DataFrame,
    selected_edges: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    curves.to_csv(output_dir / "curves.csv", index=False)
    aucs.to_csv(output_dir / "auc.csv", index=False)
    selected_edges.to_csv(output_dir / "selected_edges.csv", index=False)
    plot_curves(curves, aucs, output_dir / "summary.png")
    metadata = {
        "input": str(args.input) if args.input else None,
        "example": getattr(args, "example", None),
        "network": getattr(args, "network", None),
        "seed": getattr(args, "seed", None),
        "source_format": network.source_format,
        "nodes": len(network.labels),
        "directed": network.directed,
        "weighted": network.weighted,
        "budget": args.budget,
        "add_weight": args.add_weight,
        "algorithms": ALGORITHMS,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def connected_graph(
    kind: str,
    n: int,
    seed: int,
    *,
    ba_m: int,
    sw_k: int,
    sw_p: float,
    er_p: float,
) -> nx.Graph:
    for offset in range(10_000):
        s = seed + offset
        if kind == "BA":
            graph = nx.barabasi_albert_graph(n, ba_m, seed=s)
        elif kind == "SW":
            graph = nx.watts_strogatz_graph(n, sw_k, sw_p, seed=s)
        elif kind == "ER":
            graph = nx.erdos_renyi_graph(n, er_p, seed=s)
        else:
            raise ValueError(kind)
        if nx.is_connected(graph):
            return graph
    raise RuntimeError(f"could not generate connected {kind} graph")


def orient_strongly_connected(graph: nx.Graph, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = graph.number_of_nodes()
    adjacency = np.zeros((n, n), dtype=float)
    # A directed cycle guarantees strong connectivity, while the remaining
    # random orientations keep the example genuinely asymmetric.
    for node in range(n):
        adjacency[node, (node + 1) % n] = 1.0
    for src, dst in graph.edges():
        if adjacency[src, dst] or adjacency[dst, src]:
            continue
        if rng.random() < 0.5:
            adjacency[src, dst] = 1.0
        else:
            adjacency[dst, src] = 1.0
    return adjacency


def undirected_adjacency(graph: nx.Graph, weighted: bool, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = graph.number_of_nodes()
    adjacency = np.zeros((n, n), dtype=float)
    for src, dst in graph.edges():
        weight = float(rng.uniform(0.5, 1.5)) if weighted else 1.0
        adjacency[src, dst] = weight
        adjacency[dst, src] = weight
    return adjacency


def random_seed() -> int:
    return int(np.random.default_rng().integers(0, 2**32 - 1))


def synthetic_network(args: argparse.Namespace) -> NetworkInput:
    if args.example is None:
        raise ValueError("--example is required for synthetic generation")
    if args.network is None:
        raise ValueError("--network is required when --example is used")
    seed = args.seed if args.seed is not None else random_seed()
    args.seed = seed
    graph = connected_graph(
        args.network,
        args.n,
        seed,
        ba_m=args.ba_m,
        sw_k=args.sw_k,
        sw_p=args.sw_p,
        er_p=args.er_p,
    )
    if args.example == "fig4":
        adjacency = undirected_adjacency(graph, weighted=False, seed=seed)
        source_format = "synthetic:fig4_undirected_unweighted"
    elif args.example == "fig5":
        adjacency = orient_strongly_connected(graph, seed)
        source_format = "synthetic:fig5_directed_unweighted"
    elif args.example == "fig6":
        adjacency = undirected_adjacency(graph, weighted=True, seed=seed)
        source_format = "synthetic:fig6_undirected_weighted"
    else:
        raise ValueError(args.example)
    return NetworkInput(
        adjacency=adjacency,
        labels=[str(i) for i in range(adjacency.shape[0])],
        directed=detect_directed(adjacency),
        weighted=detect_weighted(adjacency),
        source_format=source_format,
    )


def parse_budget(text: str) -> Optional[int]:
    if text.lower() == "full":
        return None
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("budget must be positive or 'full'")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SBIA and centrality baselines on an input network."
    )
    parser.add_argument("--input", type=Path, help="Network file: square adjacency matrix or edge list.")
    parser.add_argument("--example", choices=["fig4", "fig5", "fig6"], help="Generate a synthetic example corresponding to Fig. 4, 5, or 6.")
    parser.add_argument("--network", choices=["BA", "SW", "ER"], help="Synthetic network family used with --example.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for synthetic example generation. If omitted, a random seed is generated and recorded.")
    parser.add_argument("--n", type=int, default=40, help="Number of nodes for synthetic examples.")
    parser.add_argument("--ba-m", type=int, default=2, help="BA attachment parameter for synthetic examples.")
    parser.add_argument("--sw-k", type=int, default=4, help="SW mean degree parameter for synthetic examples.")
    parser.add_argument("--sw-p", type=float, default=0.15, help="SW rewiring probability for synthetic examples.")
    parser.add_argument("--er-p", type=float, default=0.10, help="ER connection probability for synthetic examples.")
    parser.add_argument("--format", choices=["auto", "matrix", "edgelist"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("sbia_output"))
    parser.add_argument("--budget", type=parse_budget, default=None, help="Number of added edges, or 'full'.")
    parser.add_argument("--add-weight", type=float, default=1.0, help="Weight assigned to each added edge.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.input is not None and args.example is not None:
        parser.error("use either --input or --example, not both")
    if args.input is None and args.example is None:
        parser.error("either --input or --example is required")

    if args.example is not None:
        network = synthetic_network(args)
    else:
        network = load_network(args.input, args.format)
    curves, aucs, selected_edges = evaluate_network(
        network, budget=args.budget, add_weight=args.add_weight
    )
    write_outputs(args.output_dir, network, curves, aucs, selected_edges, args)
    print(
        "Detected network: "
        f"nodes={len(network.labels)}, directed={network.directed}, weighted={network.weighted}"
    )
    print(aucs.sort_values("auc", ascending=False).to_string(index=False))
    print(f"Wrote outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
