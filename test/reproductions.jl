# Aliases reproducing published LADDIE v1/v2 setups (src/simulation/presets.jl).

const V1_PARAMS = (ParamsCrossonDotsonv1, ParamsFRISv1, ParamsPIGv1, ParamsASEv1)
const V2_PARAMS = (ParamsISOMIPv2, ParamsAISv2, ParamsASEv2, ParamsPIGv2, ParamsMISOMIPv2,
                   ParamsMISOMIPGapsv2)

@testset "Params aliases" begin
    for f in (V1_PARAMS..., V2_PARAMS...)
        p = f()
        @test p isa Params{Float64}
        # Any field can be overridden, and the precision follows FT.
        q = f(; FT = Float32, H_min = 3.0)
        @test q isa Params{Float32}
        @test q.H_min === 3.0f0
        @test q.C_d_top === Float32(p.C_d_top)
    end
    @test_throws ArgumentError ParamsFRISv1(; not_a_field = 1.0)

    # The per-experiment v1 values (Lambert et al. 2023, Table 2) and the v1 code's.
    cd, fris = ParamsCrossonDotsonv1(), ParamsFRISv1()
    @test (cd.A_h, cd.K_h, cd.H_min, cd.u_tide) == (25.0, 25.0, 2.8, 0.01)
    @test (fris.A_h, fris.K_h, fris.H_min, fris.u_tide) == (50.0, 50.0, 4.4, 0.1)
    @test ParamsPIGv1() == cd
    for p in map(f -> f(), V1_PARAMS)
        @test p.entrainment.mu == 2.5               # the configs, not the paper's 0.5
        @test p.melting isa TurbulentGamTMelting
        @test p.tracer_form isa AdvectiveTracerForm
        @test p.rho0_seawater == 1028.0
    end

    # v2: the viscosity coefficient doubles as the wall viscosity, and UPSY's constants.
    for p in map(f -> f(), V2_PARAMS)
        @test p.lateral_viscosity isa NonlinearLateralViscosity
        @test p.A_h == p.lateral_viscosity.C_visc
        @test p.tracer_form isa FluxTracerForm
        @test p.convection_scheme isa ClampDensity
        @test p.drho_floor ≈ p.convection_scheme.d_rho_min / p.rho0_seawater
        @test (p.rho0_seawater, p.L, p.max_detrainment) == (1027.0, 3.335e5, 1e-3)
    end
    @test ParamsISOMIPv2().melting == UStarGamTMelting(2.78e-2)
    @test ParamsMISOMIPv2().melting == UStarGamTMelting(2.03e-2)
    @test ParamsMISOMIPGapsv2().melting == UStarGamTMelting(3.0e-2)
    @test ParamsAISv2().convection_scheme.d_rho_min == 0.02
end

# The reproduction aliases are a regression contract (AGENTS.md): a change elsewhere must
# not move what they produce.  The Simulation aliases need `papers/`, which CI does not
# have, so each parameter set runs on the same small ISOMIP cavity instead: 200 steps of
# 210 s, mean melt (m/yr) as recorded on 2026-11-17.  A deliberate change of an alias
# updates its value here and says so.  2026-11-18: the v1 aliases select v1's ice-slope
# stencil `CentredGradient` themselves; on this cavity it sees z_draft = 0 at the grounded
# cells (slope 0.18 in the 40 grounding-line and wall cells), so v1 melt rises from ~3.7
# to ~42 m/yr.
const REPRODUCTION_MEAN_MELT = (
    ParamsCrossonDotsonv1 => 41.942167852797816,
    ParamsFRISv1 => 56.338289385202444,
    ParamsPIGv1 => 41.942167852797816,
    ParamsASEv1 => 42.720908649664196,
    ParamsISOMIPv2 => 14.04424152290845,
    ParamsAISv2 => 4.176565541613683,
    ParamsASEv2 => 5.275845144368532,
    ParamsPIGv2 => 5.275844422226887,
    ParamsMISOMIPv2 => 12.037053630101514,
    ParamsMISOMIPGapsv2 => 14.241437057083251,
)

@testset "Params aliases: mean melt regression" begin
    for (f, mean_melt) in REPRODUCTION_MEAN_MELT
        sim = build_isomip(CPU(); nx = 40, ny = 20, params = f())
        for _ in 1:200
            time_step!(sim)
        end
        @test isapprox(meltstats(sim).mean_meltrate, mean_melt; rtol = 1e-10)
    end
end

@testset "Simulation aliases" begin
    if isdir(Laddie.DEFAULT_PAPERS_DIR)
        sim = SimulationPIGv1()
        @test sim.clock.dt == 120.0
        @test sim.stop == FixedSimulationEnd(30.0)
        @test count(==(3), sim.model.grid.mask) > 20_000

        # The v1 ASE run as it ran (Δx = 500 m on its 1 km cells) and as it was meant to.
        run_, meant = SimulationASEv1(), SimulationASEv1Theoretical()
        @test run_.model.grid.dx == 500.0
        @test meant.model.grid.dx == 1000.0
        @test run_.model.tmask == meant.model.tmask

        # Simulation keywords override the setup's.
        sim = SimulationISOMIPv2(; dt = 45.0)
        @test sim.clock.dt == 45.0
        @test sim.model.grid.dx == 2000.0
        for _ in 1:20
            time_step!(sim)
        end
        @test all(isfinite, sim.model.H.present)

        @test_throws ArgumentError SimulationMISOMIPGapsv2(; ocean = :ocean4)
        @test_throws ArgumentError SimulationFRISv1(; datadir = mktempdir())
    else
        @info "papers/ not found at $(Laddie.DEFAULT_PAPERS_DIR) — skipping the Simulation aliases."
        @test_skip false
    end
end
