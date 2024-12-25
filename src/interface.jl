# function try_from_dlpack(x)
#     # try
#         return from_dlpack(x)
#     # catch
#     #     a = pyconvert(Array, x)
#     #     n = ndims(a)
#     #     return permutedims(a, ntuple(i -> n-i+1, n))
#     # end
# end


"""
    pygdata_to_gnngraph(data)

Convert a PyTorch Geometric `Data` object to a `GNNGraph`.

Conversion of torch tensors to julia arrays is copyless, 
the data is shared between the two languages. 
This is done using the `DLPack.jl` library.
Since torch tensors are row-major, the corresponding julia arrays
will have permuted dimensions.
"""
function pygdata_to_gnngraph(data)
    # edge_index needs .to_dense(), otherwise from_dlpack will fail (TODO report to DLPack.jl)
    # edge_index = from_dlpack(data.edge_index.to_dense())
    num_nodes = pyconvert(Int, data.num_nodes)
    num_edges = pyconvert(Int, data.num_edges)
    if length(data.edge_index) > 0
        py_src, py_dst = data.edge_index
        src = from_dlpack(py_src)
        dst = from_dlpack(py_dst)
        src, dst = src .+ 1 , dst .+ 1
    else
        src, dst = Int[], Int[]
    end
    @assert length(src) == num_edges
    @assert length(dst) == num_edges

    @assert all(1 .<= src)
    @assert all(src .<= num_nodes)
    @assert all(1 .<= dst)
    @assert all(dst .<= num_nodes)

    # if !all(src .<= num_nodes)
    #     n = maximum(src)
    #     @warn lazy"Found node index $n in edge index `src`, but only $num_nodes nodes in the graph.
    #     Updating num_nodes to $n. This message won't be displayed again."
    #     num_nodes = n
    # end
    # if !all(dst .<= num_nodes)
    #     n = maximum(dst)
    #     @warn lazy"Found node index $n in edge index `dst`, but only $num_nodes nodes in the graph. 
    #     Updating num_nodes to $n. This message won't be displayed again."
    #     num_nodes = n
    # end
    
    ndata = (;)
    edata = (;)
    gdata = (;)
    keys = [Symbol(k) for k in data.keys()]
    for k in keys
        k == :edge_index &&  continue
        k == :num_nodes && continue
        k == :num_edges && continue
        py_x = getproperty(data, k)
        if pyisinstance(py_x, torch.Tensor)
            x = from_dlpack(py_x)
            last_dim = size(x, ndims(x))
            if last_dim == num_nodes && num_nodes != num_edges
                ndata = (; ndata..., k => x)
            elseif last_dim == num_edges && num_nodes != num_edges
                edata = (; edata..., k => x)
            else # cannot disambiguate between node and edge data
                gdata = (; gdata..., k => x)
            end
        else
            x = pyconvert(Any, py_x)
            gdata = (; gdata..., k => x)
        end
    end
    return GNNGraph((src, dst); ndata, edata, gdata, num_nodes)
end


"""
    load_dataset(dataset::String; kws...)

Load the PyTorch Geometric dataset `name`, convert it to julia types, 
and return it as an  [`InMemoryGNNDataset`](@ref) object.

The keyword arguments `kws` are passed to the dataset constructor.

See [`pygdata_to_gnngraph`](@ref) for details on the conversion of 
PyTorch Geometric `Data` objects to `GNNGraph`s.

For PyG datasets requiring a `root` argument, the default value 
`"\$(PyGDatasets.DEFAULT_ROOT[])/\$name"` is provided if not specified.
This scratch space will be deleted when the package is removed.

# References

- [PyG Datasets Cheatsheet](https://pytorch-geometric.readthedocs.io/en/stable/notes/data_cheatsheet.html)
- [PyG Datasets Documentation](https://pytorch-geometric.readthedocs.io/en/stable/modules/datasets.html)

# Examples

```julia
load_dataset("TUDataset", root="./", name="MUTAG")
# is equivalent to the python code
# pyg.datasets.TUDataset(root="./", name="MUTAG")

load_dataset("Planetoid", name="Cora")
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
    node_features = collect(keys(graphs[1].ndata))
    edge_features = collect(keys(graphs[1].edata))
    graph_features = collect(keys(graphs[1].gdata))
        
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