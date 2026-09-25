# Tests of the Reactant extension.  Not part of `Pkg.test()`: Reactant is a large
# dependency and every compile takes a minute or two.  Run from the package root:
#
#     julia --project=test/reactant -e 'using Pkg; Pkg.instantiate()'
#     julia --project=test/reactant test/reactant/runtests.jl
#
# Uses the GPU when CUDA is functional (native kernels), else Reactant's CPU target.
using Laddie, Reactant, CUDA, KernelAbstractions, Random, Test
using Reactant: Enzyme

const GPU = CUDA.functional()
Reactant.set_default_backend(GPU ? "gpu" : "cpu")
const KW = (; nx = 40, ny = 20, isomipcond = :warm)

prognostics(m) = [Array(getfield(getproperty(m, v), :present)) for v in (:D, :U, :V, :T, :S)]
maxrel(a, b) = maximum(maximum(abs, x .- y) / max(maximum(abs, y), eps()) for (x, y) in zip(a, b))

# Paths of the non-finite numbers and arrays in a (shadow) model.
function nonfinite_leaves(x, path = "", out = String[])
    if x isa Number || x isa Reactant.ConcreteRNumber
        isfinite(Reactant.to_number(x)) || push!(out, path)
    elseif x isa AbstractArray{<:AbstractFloat} || x isa Reactant.ConcretePJRTArray
        all(isfinite, Array(x)) || push!(out, path)
    elseif !(x isa AbstractArray) && !(x isa AbstractString) && fieldcount(typeof(x)) > 0
        foreach(f -> nonfinite_leaves(getfield(x, f), "$path.$f", out), fieldnames(typeof(x)))
    end
    return out
end

# A 40×20 cavity with a melt-through gap (mask 4) under ConnectedGapsBC, and a bed for
# the relative thickness cap: neither fits `build_isomip`.
function gaps_sim(; kw...)
    mask = zeros(Int, 42, 22)
    mask[[1, end], :] .= 1
    mask[:, [1, end]] .= 1
    mask[2:4, 2:21] .= 2
    mask[5:39, 2:21] .= 3
    mask[40:41, 2:21] .= 0
    mask[18:21, 8:12] .= 4
    z_draft = [-600.0 + 400 * (i - 5) / 34 for i = 1:42, _ = 1:22]
    grid = Grid(mask, z_draft, 2000.0, 2000.0; z_bed = fill(-700.0, 42, 22),
                domain_cropping = NoDomainCropping())
    model = Model(grid; forcing = ISOMIPForcing(:warm),
                  params = Params(; max_layer_thickness = RelativeMaxLayerThickness(0.8)),
                  boundary = BoundaryConditions(; gaps = ConnectedGapsBC()))
    return Simulation(model; kw...)
end

# Every non-default scheme, in four runs.  Not covered: TopographicMaxLayerThickness
# (the kernel of RelativeMaxLayerThickness at fraction 1) and OceanForcing2D (not
# implemented).
const SCHEME_SCENARIOS = [
    "mixing" => () -> build_isomip(CPU(); KW..., gradient = PyGradient(),
        cfl = ConservativeCFL(), tstep = AdaptiveDt(),
        params = Params(; melting = TurbulentGamTMelting(), entrainment = GasparEntrainment(),
                        convection_scheme = RelaxToAmbient(),
                        max_layer_thickness = AbsoluteMaxLayerThickness(50.0),
                        lateral_viscosity = NonlinearLateralViscosity(),
                        front_pressure = TruncatedDepthGradient(),
                        momentum_advection = UpstreamMomentumAdvection()),
        boundary = BoundaryConditions(; open_ocean = NoInflow(), grounding_line = FreeSlipGL(),
                                      land = PartialSlipLand(1.0))),
    "rotation" => () -> build_isomip(CPU(); KW...,
        params = Params(; melting = UStarGamTMelting(), entrainment = HollandEntrainment(),
                        convection_scheme = ClampDensity(),
                        laplacian_weights = PastLaplacianWeights(),
                        coriolis = CoriolisParameter2D([-75.0 + 0.2j for _ = 1:42, j = 1:22])),
        boundary = BoundaryConditions(; grounding_line = PartialSlipGL(1.0), land = FreeSlipLand(),
                                      wall_advection = NoWallAdvection())),
    "prescribed" => () -> build_isomip(CPU(); KW...,
        params = Params(; melting = PrescribedMelting([i < 25 ? 2.0 : 8.0 for i = 1:42, _ = 1:22]))),
    "gaps" => gaps_sim,
]

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

    @testset "traced parameters" begin
        mk(; kw...) = to_backend(build_isomip(CPU(); KW..., kw...), ReactantBackend())
        loss(model, dt, n) =
            (integrate!(model, dt, n); sum(model.melt .* model.imask) / sum(model.imask))
        sim = mk()
        model = trace_parameters(sim.model)
        @test model.FT === Float64
        @test model.params.melting.gamTfix isa Reactant.ConcreteRNumber
        @test_throws ArgumentError trace_parameters(model; nope = 1)
        traced_sim = Simulation(map(f -> f === :model ? model : getfield(sim, f), fieldnames(Simulation))...)
        @test_throws ArgumentError run!(traced_sim; days = 0.1, verbose = false)
        dt, n = ConcreteRNumber(sim.clock.dt), ConcreteRNumber(30)
        primal = reactant_compile(loss, model, dt, n)
        # Other parameter values run through the same program, and match a program
        # compiled with them as constants (to round-off: XLA folds constants).
        C_d = 3e-3
        other = mk(; params = Params(; C_d))
        @test Reactant.to_number(primal(trace_parameters(mk().model; C_d), dt, n)) ≈
              Reactant.to_number(reactant_compile(loss, other.model, dt, n)(other.model, dt, n)) rtol = 1e-10
        # Forward derivatives with respect to C_d and L, from one program: the
        # direction is an input too.
        fwd(m, dm, dt, n) = Enzyme.autodiff(Enzyme.Forward, loss, Enzyme.Duplicated(m, dm),
                                            Enzyme.Const(dt), Enzyme.Const(n))
        tangent(name) = trace_parameters(Enzyme.make_zero(model); name => 1)
        dprog = reactant_compile(fwd, model, tangent(:C_d), dt, n)
        for (name, v) in ((:C_d, sim.model.C_d), (:L, sim.model.L))
            f(h) = Reactant.to_number(primal(trace_parameters(mk().model; name => v * (1 + h)), dt, n))
            fd = (f(1e-4) - f(-1e-4)) / (2e-4 * v)
            d = Reactant.to_number(only(dprog(trace_parameters(mk().model), tangent(name), dt, n)))
            @test d ≈ fd rtol = 1e-4
        end
    end

    @testset "fusion strategies" begin
        @test_throws ArgumentError to_backend(build_isomip(CPU(); KW...), ReactantBackend(; fusion = :nope))
        strategies = GPU ? (:native, :kernel, :xla) : (:kernel, :xla)
        ref = build_isomip(CPU(); KW...)
        run!(ref; days = 0.1, verbose = false)
        for fusion in strategies
            rsim = to_backend(build_isomip(CPU(); KW...), ReactantBackend(; fusion))
            run!(rsim; days = 0.1, verbose = false)
            @test maxrel(prognostics(rsim.model), prognostics(ref.model)) < 1e-10
        end
    end

    @testset "adaptive dt: schedule, then a differentiable replay" begin
        mk(; kw...) = trace_parameters(to_backend(build_isomip(CPU(); KW...), ReactantBackend()).model; kw...)
        ref = mk()
        sched = adaptive_schedule(ref, 100.0; days = 0.5)
        nseg = Reactant.to_number(sched.nseg)
        dts, steps = Array(sched.dt)[1:nseg], Array(sched.steps)[1:nseg]
        @test nseg > 1 && dts[end] > dts[1]           # dt grows from a small start
        @test 0.5 * 86400 <= sum(dts .* steps) < 0.5 * 86400 + dts[end]
        @test_throws ArgumentError integrate!(mk(), sched)   # only inside a compiled program
        # The replay reaches the state of the adaptive run (native kernels vs raised: round-off).
        replay(model, sched) = (integrate!(model, sched); nothing)
        m = mk()
        reactant_compile(replay, m, sched)(m, sched)
        @test maxrel(prognostics(m), prognostics(ref)) < 1e-10

        loss(model, sched) = (melt = integrate!(model, sched; means = (:melt,)).melt;
                              sum(melt .* model.imask) / sum(model.imask))
        fwd(m, dm, s) = Enzyme.autodiff(Enzyme.Forward, loss, Enzyme.Duplicated(m, dm), Enzyme.Const(s))
        rev(m, dm, s) = (Enzyme.autodiff(Enzyme.Reverse, loss, Enzyme.Active, Enzyme.Duplicated(m, dm),
                                         Enzyme.Const(s)); dm)
        model = mk()
        tangent() = trace_parameters(Enzyme.make_zero(model); C_d = 1)
        d = Reactant.to_number(only(reactant_compile(fwd, model, tangent(), sched)(mk(), tangent(), sched)))
        g = reactant_compile(rev, model, Enzyme.make_zero(model), sched)(mk(), Enzyme.make_zero(model), sched)
        @test isempty(nonfinite_leaves(g))
        @test Reactant.to_number(g.params.C_d) ≈ d rtol = 1e-10
        # The derivative is that of the run with this dt sequence.
        primal = reactant_compile(loss, model, sched)
        C_d = build_isomip(CPU(); KW...).model.C_d
        f(h) = Reactant.to_number(primal(mk(; C_d = C_d * (1 + h)), sched))
        @test (f(1e-5) - f(-1e-5)) / (2e-5 * C_d) ≈ d rtol = 1e-5
    end

    # Each scheme through `run!` (default strategy) and through the forward derivative
    # of the raised `:xla` program with respect to a traced parameter.  The loss is
    # the mean layer temperature, which depends on C_d under prescribed melting too.
    @testset "scheme coverage: $name" for (name, build) in SCHEME_SCENARIOS
        ref = build()
        rsim = to_backend(build(), ReactantBackend())
        run!(ref; days = 0.5, verbose = false)
        run!(rsim; days = 0.5, verbose = false)
        @test rsim.clock.iteration == ref.clock.iteration
        @test maxrel(prognostics(rsim.model), prognostics(ref.model)) < 1e-10

        mk(; kw...) = trace_parameters(to_backend(build(), ReactantBackend()).model; kw...)
        loss(model, dt, n) =
            (integrate!(model, dt, n); sum(model.T.present .* model.tmask) / sum(model.tmask))
        fwd(m, dm, dt, n) = Enzyme.autodiff(Enzyme.Forward, loss, Enzyme.Duplicated(m, dm),
                                            Enzyme.Const(dt), Enzyme.Const(n))
        model = mk()
        fresh = build()
        dt, n = ConcreteRNumber(fresh.clock.dt), ConcreteRNumber(20)
        primal = reactant_compile(loss, model, dt, n)
        C_d = fresh.model.C_d
        f(h) = Reactant.to_number(primal(mk(; C_d = C_d * (1 + h)), dt, n))
        fd = (f(1e-4) - f(-1e-4)) / (2e-4 * C_d)
        # The zero tangent goes through validating constructors (TurbulentGamTMelting).
        tangent() = trace_parameters(Enzyme.make_zero(model); C_d = 1)
        fprog = reactant_compile(fwd, model, tangent(), dt, n)
        # A fresh tangent per call: the program advances the tangent model in place.
        d = Reactant.to_number(only(fprog(mk(), tangent(), dt, n)))
        @test d ≈ fd rtol = 1e-4

        # Reverse mode (checkpointed traced loop): the gradient with respect to every
        # input, finite everywhere (NaN from unused branches shows up here first), and
        # equal to forward mode along C_d and along a random ocean profile.
        rev(m, dm, dt, n) = (Enzyme.autodiff(Enzyme.Reverse, loss, Enzyme.Active,
                                             Enzyme.Duplicated(m, dm), Enzyme.Const(dt),
                                             Enzyme.Const(n)); dm)
        g = reactant_compile(rev, model, Enzyme.make_zero(model), dt, n)(
            mk(), Enzyme.make_zero(model), dt, n)
        @test isempty(nonfinite_leaves(g))
        @test Reactant.to_number(g.params.C_d) ≈ d rtol = 1e-10
        vTz = randn(Random.Xoshiro(1), length(model.forcing.ocean.Tz))
        tz = Enzyme.make_zero(model)
        tz.forcing.ocean.Tz .= Reactant.to_rarray(vTz)
        @test sum(Array(g.forcing.ocean.Tz) .* vTz) ≈
              Reactant.to_number(only(fprog(mk(), tz, dt, n))) rtol = 1e-10
    end

    # Sharded runs need fake CPU devices, whose count is fixed when Reactant loads.
    @testset "sharding (own process: sharding.jl)" begin
        cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) $(joinpath(@__DIR__, "sharding.jl"))`
        @test success(pipeline(cmd; stdout, stderr))
    end
end
