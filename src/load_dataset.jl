"""
    load_dataset(dataset::String; kws...)

Load the PyTorch Geometric dataset `name`, convert it to julia types, 
and return it as an  [`InMemoryGNNDataset`](@ref) object.

The keyword arguments `kws` are passed to the dataset constructor.

See [`pygdata_to_gnngraph`](@ref) for details on the conversion of 
PyTorch Geometric `Data` and `HeteroData` objects to `GNNGraph` and `GNNHeteroGraph` objects.

For PyG datasets requiring a `root` argument, the default value 
`"\$(PyGDatasets.DEFAULT_ROOT[])/\$name"` is provided if not specified.
This scratch space will be deleted when the package is removed.

# References

- [PyG Datasets Cheatsheet](https://pytorch-geometric.readthedocs.io/en/stable/notes/data_cheatsheet.html)
- [PyG Datasets Documentation](https://pytorch-geometric.readthedocs.io/en/stable/modules/datasets.html)

# Examples

```julia
dataset = load_dataset("TUDataset", root="./", name="MUTAG")
dataset = load_dataset("Planetoid", name="Cora") # use default root
dataset = load_dataset("IMDB") # heterogenous graph dataset
```
"""
function load_dataset(dataset::String; root=nothing, kws...)
    PyDataset = getproperty(pyg.datasets, Symbol(dataset))
    if has_argument(PyDataset, "root") && root === nothing 
        root = joinpath(DEFAULT_ROOT[], dataset)
        kws = (; root, kws...)
    elseif root !== nothing
        kws = (; root, kws...)
    end
    pyd = PyDataset(; kws...)
    if !pyisinstance(pyd, pyg.data.InMemoryDataset)
        error("Dataset $dataset is not an InMemoryDataset.")
    end
    graphs = [pygdata_to_gnngraph(data) for data in pyd]
    g = graphs[1]
    if g isa GNNHeteroGraph
        node_features = Dict(k => collect(keys(ndata)) for (k, ndata) in g.ndata)
        edge_features = Dict(k => collect(keys(edata)) for (k, edata) in g.edata)
    else # GNNGraph
        node_features = collect(keys(g.ndata))
        edge_features = collect(keys(g.edata))
    end
    graph_features = collect(keys(g.gdata))

    return InMemoryGNNDataset(dataset,
                              :name in keys(kws) ? kws[:name] : nothing,
                              graphs,
                              length(graphs),
                              node_features, 
                              edge_features, 
                              graph_features,
                              root)
end


function has_argument(cls, arg::String)
    signature = inspect.signature(cls.__init__)
    for (name, param) in signature.parameters.items()
        if pyconvert(String, name) == arg && Bool(param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD)
            return true
        end
    end
    return false
end