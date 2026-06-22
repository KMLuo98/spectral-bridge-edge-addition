# Spectral Bridge Edge Addition

This repository contains a single Python implementation of the spectral bridge
edge-addition experiments used for Figs. 4--6 of the manuscript.

## Requirements

```bash
pip install numpy scipy networkx matplotlib pandas
```

## Input Modes

The script supports two input modes.

Use a network file:

```bash
python spectral_bridge_edge_addition.py --input network.txt --output-dir results/network
```

The file may be a square adjacency matrix or an edge list.  For adjacency
matrices, the script automatically detects whether the network is directed and
whether it is weighted.

Use a synthetic Fig. 4--6 example without storing CSV inputs:

```bash
python spectral_bridge_edge_addition.py --example fig4 --network BA --output-dir results/fig4_BA
```

If `--seed` is omitted, a random seed is generated automatically and recorded in
`metadata.json`.

## Outputs

For each run, the script writes:

- `auc.csv`: AUC values for SBIA and the four baselines;
- `curves.csv`: RI curves during progressive edge addition;
- `selected_edges.csv`: selected edges at each step;
- `metadata.json`: detected network properties and run settings;
- `summary.png`: curve and AUC summary plot.

## Examples for Figs. 4--6

Run the Fig. 4 setting, undirected unweighted synthetic networks:

```bash
python spectral_bridge_edge_addition.py --example fig4 --network BA --output-dir results/fig4_BA
python spectral_bridge_edge_addition.py --example fig4 --network SW --output-dir results/fig4_SW
python spectral_bridge_edge_addition.py --example fig4 --network ER --output-dir results/fig4_ER
```

Run the Fig. 5 setting, directed unweighted synthetic networks:

```bash
python spectral_bridge_edge_addition.py --example fig5 --network BA --output-dir results/fig5_BA
python spectral_bridge_edge_addition.py --example fig5 --network SW --output-dir results/fig5_SW
python spectral_bridge_edge_addition.py --example fig5 --network ER --output-dir results/fig5_ER
```

Run the Fig. 6 setting, undirected weighted synthetic networks:

```bash
python spectral_bridge_edge_addition.py --example fig6 --network BA --output-dir results/fig6_BA
python spectral_bridge_edge_addition.py --example fig6 --network SW --output-dir results/fig6_SW
python spectral_bridge_edge_addition.py --example fig6 --network ER --output-dir results/fig6_ER
```

Use `--budget N` to stop after a fixed number of edge additions.  The default is
the full admissible candidate-edge set.

Synthetic-network parameters can also be changed from the command line:

```bash
python spectral_bridge_edge_addition.py --example fig4 --network SW --n 40 --sw-k 4 --sw-p 0.15
```

For reproducible synthetic runs, add `--seed INTEGER` to any synthetic command.
