@testmodule TestModule begin

using Reexport
using PyGDatasets: pyg, np, torch
using PyGDatasets

@reexport using Test
@reexport using GNNGraphs
@reexport using PythonCall

export pyg, np, torch


end # module