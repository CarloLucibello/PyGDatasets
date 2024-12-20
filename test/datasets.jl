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

