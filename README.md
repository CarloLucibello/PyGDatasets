# PyGDatasets

[![Build Status](https://github.com/CarloLucibello/PyGDatasets.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/CarloLucibello/PyGDatasets.jl/actions/workflows/CI.yml?query=branch%3Amain)

This package provides a Julia interface to the datasets available in [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/stable/modules/datasets.html)
through [PythonCall](https://github.com/JuliaPy/PythonCall.jl).
PyG data types are conveniently converted to [GNNGraphs.jl](https://juliagraphs.org/GraphNeuralNetworks.jl/docs/GNNGraphs.jl/stable/) types, ready to be used by graph deep learning frameworks such as [GraphNeuralNetworks.jl](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl). 

## Installation

```julia
pkg> add PyGDatasets
```

## Usage

The package provides the following functions:
- `load_dataset(dataset::String; kws...)`: Load a pytorch geometric dataset and convert it to julia types.
- `pygdata_to_gnngraph(data)`: Convert a PyTorch Geometric graph type to a `GNNGraphs.jl` graph type. Handles both standard graphs and heterogeneous graphs.

## Examples

```julia
julia> using PyGDatasets

julia> dataset = load_dataset("TUDataset", name="MUTAG")
TUDataset(MUTAG) - InMemoryGNNDataset
  num_graphs: 188
  node_features: [:x]
  edge_features: [:edge_attr]
  graph_features: [:y]
  root: /Users/carlo/.julia/scratchspaces/44f67abd-f36e-4be4-bfe5-65f468a62b3d/datasets/TUDataset

julia> g = dataset[1]
GNNGraph:
  num_nodes: 17
  num_edges: 38
  ndata:
    x = 7×17 Matrix{Float32}
  edata:
    edge_attr = 4×38 Matrix{Float32}
  gdata:
    y = 1-element Vector{Int64}
```
Other examples:
```julia
load_dataset("KarateClub")
load_dataset("GNNBenchmarkDataset", name="CSL")
load_dataset("ZINC", subset=true, split="test")
load_dataset("Planetoid", root="./", name="Cora")
load_dataset("MoleculeNet", name="ESOL")

# heterographs
load_dataset("IMDB")
load_dataset("AMiner")
```

## Dataset References

- [PyG Datasets Cheatsheet](https://pytorch-geometric.readthedocs.io/en/stable/notes/data_cheatsheet.html)
- [PyG Datasets Documentation](https://pytorch-geometric.readthedocs.io/en/stable/modules/datasets.html)
