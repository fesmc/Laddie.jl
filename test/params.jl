# Params: ISOMIP+ defaults, precision promotion, tracer bounds.

@testset "Params defaults: build_isomip matches Params()" begin
    # build_isomip fills in ISOMIP+-canonical parameters when `params` is not
    # given.  Those must agree field-for-field with `Params()`, otherwise
    # merely *passing* a params object to build_isomip silently changes the
    # physics — which is exactly what happened with max_layer_thickness
    # (build_isomip: Topographic, Params(): Absolute(100)), quietly turning
    # the AdaptiveTimeStepper accuracy test into an uncapped-vs-capped comparison.
    implicit = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm).model.params
    explicit = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                            params = Params(; FT)).model.params
    @test typeof(implicit) === typeof(explicit)
    for fn in fieldnames(typeof(implicit))
        @test getfield(implicit, fn) == getfield(explicit, fn)
    end
end

@testset "Params: parameterizations promoted to FT" begin
    # A mixed-precision call — FT = Float32 with objects built at Float64 —
    # yields a fully Float32 parameter set (no silent Float64 leakage that
    # would crash a Float32→Float64 setfield in the physics kernels).
    p = Params(; FT = Float32,
               entrainment  = GasparEntrainment(2.5),
               melting = FixedGamTMelting(0.00018),
               convection = ResetToAmbientConvection(0.005))
    @test p.entrainment  isa GasparEntrainment{Float32}
    @test p.melting isa FixedGamTMelting{Float32}
    @test p.convection isa ResetToAmbientConvection{Float32}
    @test !hasfield(typeof(p), :open_bc) && !hasfield(typeof(p), :gaps_bc)  # BCs live elsewhere
    @test p.lateral_viscosity isa PrescribedLateralViscosity

    p_nl = Params(; FT = Float32, lateral_viscosity = NonlinearLateralViscosity(10.0))
    @test p_nl.lateral_viscosity isa NonlinearLateralViscosity{Float32}
    @test p_nl.lateral_viscosity.C_visc isa Float32
    @test p_nl.lateral_viscosity.C_visc isa Float32

    # The payoff: a Float32 build + run from an explicit Params no longer
    # errors on a Float64-typed parameterization.  The Simulation promotes its
    # time stepper the same way (integer fields untouched).
    m = build_isomip(CPU(); FT = Float32, nx = 20, ny = 10, isomipcond = :warm,
                     params = Params(; FT = Float32),
                     time_stepper = AdaptiveTimeStepper(; cfl_target = 0.4))
    @test m.time_stepper isa AdaptiveTimeStepper{Float32}
    @test m.time_stepper.ncheck isa Int                  # integer field not converted
    @test m.nu isa Float32 && m.clock.dt isa Float32
    run!(m; days = 0.2, verbose = false)
    @test all(isfinite, m.model.melt) && eltype(m.model.melt) == Float32
end

@testset "Tracer bounds are a scheme" begin
    @test Params(; FT).tracer_bounds isa NoTracerBounds
    b = Params(; FT, tracer_bounds = TracerBounds()).tracer_bounds
    @test b isa TracerBounds{FT}
    @test (b.T_min, b.T_max, b.S_min, b.S_max) == (-5, 5, 32, 36)
    @test_throws ArgumentError TracerBounds(; T_min = 1.0, T_max = 0.0)
    @test_throws ArgumentError TracerBounds(; S_min = 36.0, S_max = 36.0)
    @test TracerBounds(; T_min = -5, S_max = Inf) isa TracerBounds{Float64}
    sim = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                       params = Params(; FT, tracer_bounds = TracerBounds(; T_max = 0.5, S_min = 34.0)))
    run!(sim; days = 0.5, verbose = false)
    m = sim.model
    act = m.tmask .> 0
    # future is the last value the bounds were applied to (before filtering)
    @test maximum(m.T.future[act]) <= 0.5
    @test minimum(m.S.future[act]) >= 34.0
    @test all(m.S.future[.!act] .== 0)          # never applied outside the domain

    # No bounds is the same run as infinite bounds, without the clamp kernel.
    run_with(bounds) = (s = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                                         params = Params(; FT, tracer_bounds = bounds));
                        run!(s; days = 0.2, verbose = false); s.model)
    free = run_with(NoTracerBounds())
    wide = run_with(TracerBounds(; T_min = -Inf, T_max = Inf, S_min = -Inf, S_max = Inf))
    @test free.T.present == wide.T.present && free.S.present == wide.S.present
end
