@testitem "KarateClub" setup=[TestModule] begin
    using .TestModule
    d = load_dataset("KarateClub")
    g = d[1]
    src, dst = edge_index(g)
    g_py = pyg.datasets.KarateClub()[0]
    src_py, dst_py = g_py.edge_index
    @test src == pyconvert(Any, src_py.numpy()) .+ 1
    @test dst == pyconvert(Any, dst_py.numpy()) .+ 1
end

