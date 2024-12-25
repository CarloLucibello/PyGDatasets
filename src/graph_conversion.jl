function try_from_dlpack(py_x::Py)
    @assert pyisinstance(py_x, torch.Tensor)
    if !Bool(py_x.is_contiguous()) # otherwise from_dlpack will fail
        py_x = py_x.contiguous() 
    end
    return from_dlpack(py_x)
end


function get_edge_index(ei::Py)
    if length(ei) > 0
        py_src, py_dst = ei
        src = try_from_dlpack(py_src)
        dst = try_from_dlpack(py_dst)
        src, dst = src .+ 1 , dst .+ 1
    else
        src, dst = Int[], Int[]
    end
    return src, dst
end

to_node_t(k::Py) = to_node_t(pyconvert(String, k))
to_node_t(k::String) = Symbol(k)
to_node_t(k::Symbol) = k
to_edge_t(k::Py) = to_edge_t(pyconvert(Tuple, k))
to_edge_t(k::Tuple{String,String,String}) = Symbol.(k)
to_edge_t(k::Tuple{Symbol,Symbol,Symbol}) = k

"""
    pygdata_to_gnngraph(data)

Convert a PyG `Data` object to a `GNNGraph`,
and a PyG `HeteroData` object to a `GNNHeteroGraph`.

Conversion of torch tensors to julia arrays is copyless, 
the data is shared between the two languages. 
This is done using the `DLPack.jl` library.
Since torch tensors are row-major, the corresponding julia arrays
will have permuted dimensions.
"""
function pygdata_to_gnngraph(data::Py)
    if pyisinstance(data, pyg.data.HeteroData)
        return to_gnnheterograph(data)
    elseif pyisinstance(data, pyg.data.Data)
        return to_gnngraph(data)
    else
        error("Unsupported data type: $(pytype(data))")
    end
end


function to_gnngraph(data)
    # edge_index needs .to_dense(), otherwise from_dlpack will fail (TODO report to DLPack.jl)
    # edge_index = from_dlpack(data.edge_index.to_dense())
    num_nodes = pyconvert(Int, data.num_nodes)
    num_edges = pyconvert(Int, data.num_edges)
    src, dst = get_edge_index(data.edge_index)
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
            x = try_from_dlpack(py_x)
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

const NODE_T = Symbol
const EDGE_T = Tuple{Symbol,Symbol,Symbol}

function to_gnnheterograph(data)
    num_nodes = Dict(to_node_t(k) => pyconvert(Int, data[k].num_nodes) for k in data.node_types)
    num_edges = Dict(to_edge_t(k) => pyconvert(Int, data[k].num_edges) for k in data.edge_types)
    edge_index = Dict(to_edge_t(k) => get_edge_index(data[k].edge_index) for k in data.edge_types)
    for edge_t in keys(edge_index)
        src, dst = edge_index[edge_t]
        @assert length(src) == num_edges[edge_t]
        @assert length(dst) == num_edges[edge_t]
    end
    ndata = Dict{NODE_T,NamedTuple}(to_node_t(k) => (;) for k in data.node_types)
    edata = Dict{EDGE_T,NamedTuple}(to_edge_t(k) => (;) for k in data.edge_types)
    gdata = (;)
    for t in data.node_types
        jt = to_node_t(t)
        for k in data[t].keys()
            jk = Symbol(k)
            # @show jt jk 
            py_x = data[t][k]
            # @show pytype(py_x)
            x = try_from_dlpack(py_x)
            last_dim = size(x, ndims(x))
            # @show size(x) num_nodes[jt]
            if last_dim != num_nodes[jt]
                prop_dict = get(gdata, jk, Dict{NODE_T, Any}())
                prop_dict[jt] = x
                gdata = (; gdata..., jk => prop_dict)
            else
                ndata[jt] = (; ndata[jt]..., jk => x)
            end
        end
    end
    for t in data.edge_types
        jt = to_edge_t(t)
        for k in data[t].keys()
            jk = Symbol(k)
            jk == :edge_index && continue
            py_x = data[t][k]
            x = try_from_dlpack(py_x)
            last_dim = size(x, ndims(x))
            if last_dim != num_edges[jt] || jk == :edge_label_index
                prop_dict = get(gdata, jk, Dict{EDGE_T, Any}())
                prop_dict[jt] = x
                gdata = (; gdata..., jk => prop_dict)
            else
                edata[jt] = (; edata[jt]..., jk => x)
            end
        end
    end
    #TODO handle graph data
    return GNNHeteroGraph(edge_index; num_nodes, ndata, edata, gdata)
end
