# Sharded runs on a mesh of fake CPU devices.  Its own process, since the device count
# is fixed when Reactant loads; `runtests.jl` runs it, or from the package root:
#
#     julia --project=test/reactant test/reactant/sharding.jl
ENV["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
using Laddie, Reactant, CUDA, Test
using Reactant: Sharding
Reactant.set_default_backend("cpu")

const KW = (; nx = 40, ny = 20, isomipcond = :warm)   # a 42 × 22 grid
prognostics(m) = [Array(getfield(getproperty(m, v), :present)) for v in (:D, :U, :V, :T, :S)]
maxrel(a, b) = maximum(maximum(abs, x .- y) / max(maximum(abs, y), eps()) for (x, y) in zip(a, b))
nshards(a) = length(unique(a.sharding.device_to_array_slices))
melt(dir, n) = Laddie.NCDatasets.NCDataset(ds -> coalesce.(Array(ds["melt"][:, :, :]), 0.0),
                                           joinpath(dir, n, "output.nc"))

@testset "Reactant sharding" begin
    mesh_y = Sharding.Mesh(collect(0:1), (:y,))
    mesh_xy = Sharding.Mesh(reshape(collect(0:3), 2, 2), (:x, :y))

    @testset "run! on a mesh matches the CPU: $name" for (name, backend, shards, output) in (
        ("y, with output", ReactantBackend(; mesh = mesh_y), 2, true),
        ("x by partition", ReactantBackend(; mesh = mesh_y, partition = (:y, nothing)), 2, false),
        ("x and y", ReactantBackend(; mesh = mesh_xy), 4, false),
    )
        dir = mktempdir()
        out(n) = output ? (; output = OutputConfig(; name = n, resultdir = dir, saveday = 0.25)) : (;)
        ref = build_isomip(CPU(); KW..., out("cpu")...)
        rsim = to_backend(build_isomip(CPU(); KW..., out("rx")...), backend)
        @test nshards(rsim.model.D.present) == shards
        @test nshards(rsim.model.forcing.ocean.Tz) == 1        # replicated
        for s in (ref, rsim)
            run!(s; days = 0.5, verbose = false)
        end
        @test rsim.clock.iteration == ref.clock.iteration
        @test maxrel(prognostics(rsim.model), prognostics(ref.model)) < 1e-8
        output && @test maxrel([melt(dir, "rx")], [melt(dir, "cpu")]) < 1e-8
    end

    @testset "fusion on a mesh" begin
        ext = Base.get_extension(Laddie, :LaddieReactantExt)
        @test ext._fusion(ReactantBackend(; mesh = mesh_y)) === :kernel
        @test ext._fusion(ReactantBackend(; mesh = mesh_xy)) === :xla
        @test ext._fusion(ReactantBackend(; mesh = mesh_y, fusion = :xla)) === :xla
        @test_throws ArgumentError to_backend(build_isomip(CPU(); KW...),
                                              ReactantBackend(; mesh = mesh_y, fusion = :native))
    end

    # 22 rows do not split over 4 devices: an error, not a silently replicated array.
    # Rounding the crop up fixes it (along x here: the cavity fills y).
    @testset "grid sizes that do not split" begin
        mesh4 = Sharding.Mesh(collect(0:3), (:p,))
        @test_throws ArgumentError to_backend(build_isomip(CPU(); KW...), ReactantBackend(; mesh = mesh4))
        crop(multiple) = MinRectangleDomainCropping(; margin = 2, multiple)
        split_x = ReactantBackend(; mesh = mesh4, partition = (:p, nothing))
        @test size(build_isomip(CPU(); KW..., domain_cropping = crop(1)).model.melt, 1) % 4 != 0
        sim = build_isomip(CPU(); KW..., domain_cropping = crop((4, 1)))
        @test size(sim.model.melt, 1) % 4 == 0
        @test nshards(to_backend(sim, split_x).model.D.present) == 4
    end
end
