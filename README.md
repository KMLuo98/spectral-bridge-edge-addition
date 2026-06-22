# Spectral Bridge Edge Addition

This repository contains a single Python implementation of the spectral bridge edge-addition experiments used for Figs. 4--6 of the manuscript.

## Requirements

```bash
pip install numpy scipy networkx matplotlib pandas
```

## Input

The main input is a network file passed through `--input`. The recommended format is a square adjacency matrix in CSV or whitespace-delimited text form. For adjacency matrices, the script automatically detects whether the network is directed and whether it is weighted.

Edge-list inputs are also accepted, but a one-line-per-edge undirected edge list is intrinsically ambiguous. Use adjacency matrices when automatic detection must be unambiguous.

## Outputs

For each run, the script writes `auc.csv`, `curves.csv`, `selected_edges.csv`, `metadata.json`, and `summary.png`.

## Examples for Figs. 4--6

The script can generate adjacency-matrix example inputs corresponding to the three structural settings:

```bash
python spectral_bridge_edge_addition.py --write-examples examples
```

This creates:

- `examples/fig4_undirected_unweighted/`: BA, SW, and ER examples;
- `examples/fig5_directed_unweighted/`: BA, SW, and ER examples;
- `examples/fig6_undirected_weighted/`: BA, SW, and ER examples.

Run the Fig. 4 examples:

```bash
python spectral_bridge_edge_addition.py --input examples/fig4_undirected_unweighted/BA.csv --output-dir results/fig4_BA
python spectral_bridge_edge_addition.py --input examples/fig4_undirected_unweighted/SW.csv --output-dir results/fig4_SW
python spectral_bridge_edge_addition.py --input examples/fig4_undirected_unweighted/ER.csv --output-dir results/fig4_ER
```

Run the Fig. 5 examples:

```bash
python spectral_bridge_edge_addition.py --input examples/fig5_directed_unweighted/BA.csv --output-dir results/fig5_BA
python spectral_bridge_edge_addition.py --input examples/fig5_directed_unweighted/SW.csv --output-dir results/fig5_SW
python spectral_bridge_edge_addition.py --input examples/fig5_directed_unweighted/ER.csv --output-dir results/fig5_ER
```

Run the Fig. 6 examples:

```bash
python spectral_bridge_edge_addition.py --input examples/fig6_undirected_weighted/BA.csv --output-dir results/fig6_BA
python spectral_bridge_edge_addition.py --input examples/fig6_undirected_weighted/SW.csv --output-dir results/fig6_SW
python spectral_bridge_edge_addition.py --input examples/fig6_undirected_weighted/ER.csv --output-dir results/fig6_ER
```

Use `--budget N` to stop after a fixed number of edge additions. The default is the full admissible candidate-edge set.
