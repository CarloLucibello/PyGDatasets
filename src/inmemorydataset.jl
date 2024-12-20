"""
    InMemoryGNNDataset

A graph dataset that holds all graphs in memory as a collection of `GNNGraph`s.
"""
struct InMemoryGNNDataset
    dataset::String
    subdataset::Union{String, Nothing}
    graphs::Vector{GNNGraph}
    node_features::Vector{Symbol}
    edge_features::Vector{Symbol}
    graph_features::Vector{Symbol}
    root::Union{String, Nothing}
end

Base.getindex(d::InMemoryGNNDataset, i::Int) = d.graphs[i]
Base.length(d::InMemoryGNNDataset) = length(d.graphs)

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
        if length(d.node_features) > 0
            print(io, "\n  node_features: ", d.node_features)
        end
        if length(d.edge_features) > 0
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
