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
