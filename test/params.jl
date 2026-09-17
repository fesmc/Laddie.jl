# Params: ISOMIP+ defaults, precision promotion, tracer bounds.

@testset "Params defaults: build_isomip matches Params()" begin
    # build_isomip fills in ISOMIP+-canonical parameters when `params` is not
    # given.  Those must agree field-for-field with `Params()`, otherwise
    # merely *passing* a params object to build_isomip silently changes the
    # physics — which is exactly what happened with max_layer_thickness
    # (build_isomip: Topographic, Params(): Absolute(100)), quietly turning
    # the AdaptiveDt accuracy test into an uncapped-vs-capped comparison.
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
               convection_scheme = ResetToAmbient(0.005))
    @test p.entrainment  isa GasparEntrainment{Float32}
    @test p.melting isa FixedGamTMelting{Float32}
    @test p.convection_scheme isa ResetToAmbient{Float32}
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
                     tstep = AdaptiveDt(; cfl_target = 0.4))
    @test m.tstep isa AdaptiveDt{Float32}
    @test m.tstep.ncheck isa Int                  # integer field not converted
    @test m.nu isa Float32 && m.clock.dt isa Float32
    run!(m; days = 0.2, verbose = false)
    @test all(isfinite, m.model.melt) && eltype(m.model.melt) == Float32
end

@testset "Tracer bounds are parameters" begin
    p = Params(; FT)
    @test (p.T_min, p.T_max, p.S_min, p.S_max) == (-5, 5, 32, 36)
    sim = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                       params = Params(; FT, T_max = 0.5, S_min = 34.0))
    run!(sim; days = 0.5, verbose = false)
    m = sim.model
    act = m.tmask .> 0
    # future is the last value the bounds were applied to (before filtering)
    @test maximum(m.T.future[act]) <= 0.5
    @test minimum(m.S.future[act]) >= 34.0
    @test all(m.S.future[.!act] .== 0)          # never applied outside the domain
end
