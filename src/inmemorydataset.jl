"""
    InMemoryGNNDataset

A graph dataset that holds all graphs in memory as a collection of `GNNGraph`s.

# Fields

- `dataset::String`: The name of the dataset.
- `subdataset::Union{String, Nothing}`: The name of the subdataset if any.
- `graphs::Vector{GNNGraph}`: The collection of graphs.
- `num_graphs::Int`: The number of graphs in the dataset.
- `node_features::Vector{Symbol}`: The names of the node features.
- `edge_features::Vector{Symbol}`: The names of the edge features.
- `graph_features::Vector{Symbol}`: The names of the graph features.
- `root::Union{String, Nothing}`: The name of the root node.
"""
struct InMemoryGNNDataset{G<:AbstractGNNGraph, NF, EF, GF}
    dataset::String
    subdataset::Union{String,Nothing}
    graphs::Vector{G}
    num_graphs::Int
    node_features::NF
    edge_features::EF
    graph_features::GF
    root::Union{String,Nothing}
end

Base.getindex(d::InMemoryGNNDataset, i::Int) = d.graphs[i]
Base.length(d::InMemoryGNNDataset) = length(d.graphs)
Base.iterate(d::InMemoryGNNDataset) = iterate(d.graphs)
Base.iterate(d::InMemoryGNNDataset, i) = iterate(d.graphs, i)

function Base.show(io::IO, d::InMemoryGNNDataset)
    if get(io, :compact, false)
        print(io, "InMemoryGNNDataset(", length(d), ")")
    else
        print(io, d.dataset)
        if d.subdataset !== nothing
            print(io, "($(d.subdataset))")
        end
        println(io, " - InMemoryGNNDataset")
        print(io, "  num_graphs: ", length(d))
        if d.node_features isa Dict
            features = Dict(k => v for (k, v) in d.node_features if length(v) > 0)
            if length(features) > 0
                print(io, "\n  node_features: ", features)
            end     
        elseif length(d.node_features) > 0
            print(io, "\n  node_features: ", d.node_features)
        end
        if d.edge_features isa Dict
            features = Dict(k => v for (k, v) in d.edge_features if length(v) > 0)
            if length(features) > 0
                print(io, "\n  edge_features: ", features)
            end
        elseif length(d.edge_features) > 0
            print(io, "\n  edge_features: ", d.edge_features)
        end
        if length(d.graph_features) > 0
            print(io, "\n  graph_features: ", d.graph_features)
        end
        if d !== nothing
            print(io, "\n  root: ", d.root)
        end
    end
end
