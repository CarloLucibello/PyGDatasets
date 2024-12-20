module PyGDatasets

using PythonCall
using GNNGraphs: GNNGraph
using DLPack: from_dlpack
using Scratch: get_scratch!

include("inmemorydataset.jl")
export InMemoryGNNDataset

include("interface.jl")
export pygdata_to_gnngraph, load_dataset

const pyg = PythonCall.pynew()
const np = PythonCall.pynew()
const torch = PythonCall.pynew()
const inspect = PythonCall.pynew()

const DEFAULT_ROOT = Ref{String}("")

function __init__()
    # Since it is illegal in PythonCall to import a python module in a module, we need to do this here.
    # https://juliapy.github.io/PythonCall.jl/dev/pythoncall-reference/#PythonCall.Core.pycopy!
    PythonCall.pycopy!(pyg, pyimport("torch_geometric"))
    PythonCall.pycopy!(np, pyimport("numpy"))
    PythonCall.pycopy!(torch, pyimport("torch"))
    PythonCall.pycopy!(inspect, pyimport("inspect"))
    
    DEFAULT_ROOT[] = get_scratch!(@__MODULE__, "datasets")
end

end # module
