@testitem "KarateClub" setup=[TestModule] begin
    using .TestModule
    d = load_dataset("KarateClub")
    @test d.num_graphs == 1
    @test d.node_features == [:y, :train_mask, :x]
    @test d.edge_features == Symbol[]
    @test d.graph_features == Symbol[]
    @test d.root === nothing
    g = d[1]
    src, dst = edge_index(g)
    g_py = pyg.datasets.KarateClub()[0]
    src_py, dst_py = g_py.edge_index
    @test src == pyconvert(Any, src_py.numpy()) .+ 1
    @test dst == pyconvert(Any, dst_py.numpy()) .+ 1
    @test src isa Vector{Int}
    @test dst isa Vector{Int}
    @test g.y == pyconvert(Any, g_py.y.numpy())
    @test g.x == pyconvert(Any, g_py.x.numpy().T)
    @test g.train_mask == pyconvert(Any, g_py.train_mask.numpy())
    @test g.num_nodes == pyconvert(Int, g_py.num_nodes)
    @test g.num_edges == pyconvert(Int, g_py.num_edges)
end

@testitem "Cora" setup=[TestModule] begin
    using .TestModule
    dataset = load_dataset("Planetoid", name="Cora")
    @test dataset.num_graphs == 1
    @test dataset.node_features == [:val_mask, :test_mask, :y, :train_mask, :x] 
    g = dataset[1]
    @test g isa GNNGraph
    @test g.num_nodes == 2708
    @test g.num_edges == 10556
    @test g.y isa Vector{Int}
    @test g.x isa Matrix{Float32}
    @test g.train_mask isa Vector{Bool}
    @test g.val_mask isa Vector{Bool}
    @test g.test_mask isa Vector{Bool}
    @test size(g.x) == (1433, 2708)
end

@testitem "FreeSolv" setup=[TestModule] begin
    using .TestModule
    dataset = load_dataset("MoleculeNet", name="FreeSolv")
    @test dataset.num_graphs == 642
    @test dataset.node_features == [:x]
    @test dataset.edge_features == [:edge_attr]
    @test dataset.graph_features == [:y, :smiles]
    @test length(dataset.graphs) == 642
    g = dataset[1]
    @test g.x isa Matrix{Int}
    @test size(g.x) == (9, 13)
    @test g.x == [6  7  6  6  8  6  6  6  6  6  6  8  6
                0  0  0  0  0  0  0  0  0  0  0  0  0
                4  3  4  3  1  3  3  3  3  3  3  2  4
                5  5  5  5  5  5  5  5  5  5  5  5  5
                3  0  3  0  0  0  1  1  0  1  1  0  3
                0  0  0  0  0  0  0  0  0  0  0  0  0
                4  3  4  3  3  3  3  3  3  3  3  3  4
                0  0  0  0  0  1  1  1  1  1  1  0  0
                0  0  0  0  0  1  1  1  1  1  1  0  0]
    @test g.edge_attr isa Matrix{Int}
    @test size(g.edge_attr) == (3, 26)
    @test g.gdata.y isa Matrix{Float32}
    @test size(g.gdata.y) == (1, 1)
end

@testitem "MUTAG" setup=[TestModule] begin
    using .TestModule
    dataset = load_dataset("TUDataset", name="MUTAG")
    @test dataset.num_graphs == 188
    @test dataset.node_features == [:x]
    @test dataset.edge_features == [:edge_attr]
    @test dataset.graph_features == [:y]
    @test length(dataset) == 188
    for g in dataset
        @test g.ndata.x isa Matrix{Float32}
        @test size(g.ndata.x) == (7, g.num_nodes)
    end

    g = dataset[31]
    src, dst = edge_index(g)
    @test src isa Vector{Int}
    @test dst isa Vector{Int}
    @test src == [ 0,  0,  1,  1,  2,  2,  3,  3,  3,  4,  4,  4,  5,  5,  6,  6,  6,  7,
                7,  8,  8,  9,  9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14,
                14, 14, 15, 15, 16, 16, 16, 17, 18, 19, 19, 19, 20, 21] .+ 1
    @test dst == [ 1,  5,  0,  2,  1,  3,  2,  4, 12,  3,  5,  6,  0,  4,  4,  7, 11,  6,
                    8,  7,  9,  8, 10,  9, 11, 15,  6, 10, 12,  3, 11, 13, 12, 14, 19, 13,
                    15, 16, 10, 14, 14, 17, 18, 16, 16, 13, 20, 21, 19, 19] .+ 1
end

@testitem "ESOL" setup=[TestModule] begin
    using .TestModule
    dataset = load_dataset("MoleculeNet", name="ESOL")
    @test dataset.num_graphs == 1128
    @test dataset.node_features == [:x]
    @test dataset.edge_features == [:edge_attr]
    @test dataset.graph_features == [:y, :smiles]
    @test length(dataset) == 1128
    @test dataset[1].smiles == "OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O "
    @test dataset[2].smiles == "Cc1occc1C(=O)Nc2ccccc2"
    @test dataset[3].smiles == "CC(C)=CCCC(C)=CC(=O)"
    @test dataset[4].smiles == "c1ccc2c(c1)ccc3c2ccc4c5ccccc5ccc43"
end

@testitem "IMDB" setup=[TestModule] begin
    using .TestModule
    dataset = load_dataset("IMDB")
    @test dataset.num_graphs == 1
    @test dataset.node_features == Dict(:director => [:x], :actor => [:x], :movie => [:val_mask, :test_mask, :y, :train_mask, :x])
    @test dataset.edge_features == Dict((:movie, :to, :actor) => [], (:movie, :to, :director) => [], (:actor, :to, :movie) => [], (:director, :to, :movie) => [])
    @test dataset.graph_features == []
    @test length(dataset) == 1
    g = dataset[1]
    @test g.num_nodes == Dict(:director => 2081, :actor => 5257, :movie => 4278)
    @test g.num_edges == Dict((:movie, :to, :actor) => 12828, (:movie, :to, :director) => 4278, (:actor, :to, :movie) => 12828, (:director, :to, :movie) => 4278)
    @test g.ndata[:director].x isa Matrix{Float32}
    @test size(g.ndata[:director].x) == (3066, 2081)
    @test g.ndata[:actor].x isa Matrix{Float32}
    @test size(g.ndata[:actor].x) == (3066, 5257)
    @test g.ndata[:movie].x isa Matrix{Float32}
    @test size(g.ndata[:movie].x) == (3066, 4278)
    @test g.ndata[:movie].val_mask isa Vector{Bool}
    @test g.ndata[:movie].test_mask isa Vector{Bool}
    @test g.ndata[:movie].train_mask isa Vector{Bool}
    @test sum(g.ndata[:movie].train_mask) == 400
    @test sum(g.ndata[:movie].val_mask) == 400
    @test sum(g.ndata[:movie].test_mask) == 3478
    @test g.ndata[:movie].y isa Vector{Int}
    src, dst = edge_index(g, (:movie, :to, :actor))
    @test src isa Vector{Int}
    @test dst isa Vector{Int}
    @test length(src) == length(dst) == 12828
    @test src[1:20] == [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7]
    @test dst[1:20] == [675, 2395, 5130, 2005, 2480, 3792, 891, 4293, 4651, 870, 2535, 4924, 1133, 3983, 4392, 1999, 2077, 2827, 562, 1331]
    src, dst = edge_index(g, (:movie, :to, :director))
    @test length(src) == length(dst) == 4278
    src, dst = edge_index(g, (:actor, :to, :movie))
    @test length(src) == length(dst) == 12828
    src, dst = edge_index(g, (:director, :to, :movie))
    @test length(src) == length(dst) == 4278
end

@testitem "MovieLens100K" setup=[TestModule] begin
    using .TestModule
    dataset = load_dataset("MovieLens100K")
    @test length(dataset) == 1
    g = dataset[1]
    @test g.num_nodes == Dict(:user => 943, :movie => 1682)
    @test g.num_edges == Dict((:movie, :rated_by, :user) => 80000, (:user, :rates, :movie) => 80000)
    @test g.gdata.edge_label[(:user, :rates, :movie)] isa Vector{Float32}
    @test length(g.gdata.edge_label[(:user, :rates, :movie)]) == 20000
    @test g.gdata.edge_label_index[(:user, :rates, :movie)] isa Matrix{Int}
    @test size(g.gdata.edge_label_index[(:user, :rates, :movie)]) == (20000, 2)
end
