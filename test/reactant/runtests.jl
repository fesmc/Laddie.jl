# Tests of the Reactant extension.  Not part of `Pkg.test()`: Reactant is a large
# dependency and every compile takes a minute or two.  Run from the package root:
#
#     julia --project=test/reactant -e 'using Pkg; Pkg.instantiate()'
#     julia --project=test/reactant test/reactant/runtests.jl
#
# Uses the GPU when CUDA is functional (native kernels), else Reactant's CPU target.
using Laddie, Reactant, CUDA, KernelAbstractions, Test
using Reactant: Enzyme

const GPU = CUDA.functional()
Reactant.set_default_backend(GPU ? "gpu" : "cpu")
const KW = (; nx = 40, ny = 20, isomipcond = :warm)

prognostics(m) = [Array(getfield(getproperty(m, v), :present)) for v in (:D, :U, :V, :T, :S)]
maxrel(a, b) = maximum(maximum(abs, x .- y) / max(maximum(abs, y), eps()) for (x, y) in zip(a, b))

@testset "Reactant extension" begin
    @testset "run! matches the CPU ($(GPU ? "GPU" : "CPU") target)" begin
        for FT in (Float64, Float32), tstep in (FixedDt(), AdaptiveDt())
            ref = build_isomip(CPU(); KW..., FT, tstep)
            rsim = to_backend(build_isomip(CPU(); KW..., FT, tstep), ReactantBackend())
            for s in (ref, rsim)
                run!(s; days = 0.5, verbose = false)
                run!(s; days = 0.5, verbose = false)   # reuses the compiled programs
            end
            @test rsim.clock.iteration == ref.clock.iteration
            err = maxrel(prognostics(rsim.model), prognostics(ref.model))
            # Native kernels are the KA kernels (bit-identical in Float64 with fixed
            # dt); raised ones and XLA's reductions agree to round-off.
            @test err < (FT == Float64 ? 1e-10 : 1e-3)
        end
    end

    @testset "output, log and restarts" begin
        dir = mktempdir()
        out(name) = OutputConfig(; name, resultdir = dir, saveday = 0.25, diagday = 0.25, restday = 0.5)
        ref = build_isomip(CPU(); KW..., output = out("cpu"))
        rsim = to_backend(build_isomip(CPU(); KW..., output = out("rx")), ReactantBackend())
        run!(ref; days = 1.0, verbose = false)
        run!(rsim; days = 1.0, verbose = false)
        melt(n) = Laddie.NCDatasets.NCDataset(ds -> coalesce.(Array(ds["melt"][:, :, :]), 0.0),
                                              joinpath(dir, n, "output.nc"))
        @test size(melt("rx")) == size(melt("cpu"))
        @test maxrel([melt("rx")], [melt("cpu")]) < 1e-8
        @test countlines(joinpath(dir, "rx", "log.txt")) == countlines(joinpath(dir, "cpu", "log.txt"))
        @test isfile(joinpath(dir, "rx", "restart_latest.jld2"))
    end

    @testset "forward-mode AD through reactant_compile" begin
        mk() = to_backend(build_isomip(CPU(); KW...), ReactantBackend())
        loss(model, dt, n) =
            (integrate!(model, dt, n); sum(model.melt .* model.imask) / sum(model.imask))
        sim = mk()
        dt, n = ConcreteRNumber(sim.clock.dt), ConcreteRNumber(30)
        primal = reactant_compile(loss, sim.model, dt, n)
        f(h) = (s = mk(); s.model.forcing.ocean.Tz .+= h; Reactant.to_number(primal(s.model, dt, n)))
        fd = (f(1e-3) - f(-1e-3)) / 2e-3
        dmodel = Enzyme.make_zero(sim.model)
        dmodel.forcing.ocean.Tz .= 1
        fwd(m, dm, dt, n) = Enzyme.autodiff(Enzyme.Forward, loss, Enzyme.Duplicated(m, dm),
                                            Enzyme.Const(dt), Enzyme.Const(n))
        d = Reactant.to_number(only(reactant_compile(fwd, sim.model, dmodel, dt, n)(sim.model, dmodel, dt, n)))
        @test d ≈ fd rtol = 1e-5
    end

    @testset "fusion strategies" begin
        @test_throws ArgumentError to_backend(build_isomip(CPU(); KW...), ReactantBackend(; fusion = :nope))
        strategies = GPU ? (:native, :kernel, :stencil, :xla) : (:kernel, :stencil, :xla)
        ref = build_isomip(CPU(); KW...)
        run!(ref; days = 0.1, verbose = false)
        for fusion in strategies
            rsim = to_backend(build_isomip(CPU(); KW...), ReactantBackend(; fusion))
            run!(rsim; days = 0.1, verbose = false)
            @test maxrel(prognostics(rsim.model), prognostics(ref.model)) < 1e-10
        end
    end
end
