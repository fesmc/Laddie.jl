# Numerics: fused kernels vs reference terms, conservation, precision.

@testset "Fused kernels match reference equation terms" begin
    # The fused step kernels in src/numerics/leapfrog.jl and the equation-term
    # functions in equation_terms.jl implement the same governing equations.
    # Reconstruct one leapfrog step from the term functions and require
    # the kernels to reproduce it, for both the scalar-coefficient
    # (FixedGamTMelting/ResetToAmbientConvection) and matrix-coefficient
    # (TurbulentGamTMelting/RelaxToAmbientConvection) kernel variants.
    configs = (
        Params(; FT),
        Params(; FT,
               melting = TurbulentGamTMelting(FT(13.8), FT(2432.0), FT(1.95e-6)),
               convection = RelaxToAmbientConvection(FT(10000.0)),
               entrainment  = HollandEntrainment(FT(0.01775))),
        Params(; FT, front_pressure = ZeroGradientFrontPressure()),
        Params(; FT, max_layer_thickness = AbsoluteMaxLayerThickness(FT(5.0))),
        Params(; FT, frazil_ice = RelaxedFrazilIce(FT(3600.0)),
               convection = ClampDensityConvection(FT(0.005))),
        Params(; FT, tracer_form = FluxTracerForm(),
               max_layer_thickness = AbsoluteMaxLayerThickness(FT(5.0))),
        Params(; FT, tracer_form = FluxTracerForm(), frazil_ice = RelaxedFrazilIce(FT(3600.0)),
               convection = ClampDensityConvection(FT(0.005))),
        Params(; FT, momentum_form = FluxMomentumForm(), max_layer_thickness = AbsoluteMaxLayerThickness(FT(5.0)),
               frazil_ice = RelaxedFrazilIce(FT(3600.0)), convection = ClampDensityConvection(FT(0.005))),
        Params(; FT, integrator = LeapfrogIntegrator(; closure_level = PastClosureLevel()),
               melting = TurbulentGamTMelting(FT(13.8), FT(2432.0), FT(1.95e-6)),
               entrainment  = HollandEntrainment(FT(0.01775))),
        Params(; FT, momentum_form = FluxMomentumForm(), integrator = LeapfrogIntegrator(; closure_level = PastClosureLevel())),
        Params(; FT, momentum_form = FluxMomentumForm(), momentum_advection = UpstreamMomentumAdvection(),
               max_layer_thickness = AbsoluteMaxLayerThickness(FT(5.0))),
    )
    for params in configs
        m = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm, params)
        run!(m; days = 0.2, verbose = false)   # develop a non-trivial flow
        # Frazil needs a supercooled layer, which ISOMIP+ never forms.
        params.frazil_ice isa RelaxedFrazilIce && supercool!(m.model, FT(0.05))
        Laddie.advance_leapfrog!(m.model, m.clock.dt)
        dt = 2 * m.clock.dt
        Laddie.step_thickness!(m.model, dt)
        Laddie.precompute_integration_terms!(m.model, m.model.H.present, dt)

        rhs_U =    u_advection(m.model) .-
                   u_pressure_depth(m.model)     .+ u_pressure_slope(m.model) .-
                   u_pressure_density(m.model)   .+ u_coriolis(m.model) .+
                   u_diffusion(m.model)
        U_ref = u_momentum_step(m.model, rhs_U, dt)

        rhs_V =    v_advection(m.model) .-
                   v_pressure_depth(m.model)     .+ v_pressure_slope(m.model) .-
                   v_pressure_density(m.model)   .- v_coriolis(m.model) .+
                   v_diffusion(m.model)
        V_ref = v_momentum_step(m.model, rhs_V, dt)

        rhs_T =    tracer_advection(m.model, m.model.T.present) .+
                   tracer_entrainment(m.model, m.model.Ta) .-
                   tracer_cap_detrainment(m.model, m.model.T.present) .+
                   T_ice_ocean_exchange(m.model) .+
                   tracer_diffusion(m.model, m.model.T.past) .-
                   tracer_convection(m.model, m.model.T.past, m.model.Ta) .+
                   T_frazil(m.model)
        T_ref = tracer_step(m.model, m.model.T, rhs_T, dt; extra_sink = frazil_field(m.model))

        rhs_S =    tracer_advection(m.model, m.model.S.present) .+
                   tracer_entrainment(m.model, m.model.Sa) .-
                   tracer_cap_detrainment(m.model, m.model.S.present) .+
                   tracer_diffusion(m.model, m.model.S.past) .-
                   tracer_convection(m.model, m.model.S.past, m.model.Sa)
        S_ref = tracer_step(m.model, m.model.S, rhs_S, dt)

        Laddie.step_u_momentum!(m.model, dt)
        Laddie.step_v_momentum!(m.model, dt)
        Laddie.step_temperature!(m.model, dt)
        Laddie.step_salinity!(m.model, dt)

        @test m.model.U.future ≈ U_ref rtol = 1e-10 atol = 1e-12
        @test m.model.V.future ≈ V_ref rtol = 1e-10 atol = 1e-12
        @test m.model.T.future ≈ T_ref rtol = 1e-10 atol = 1e-12
        @test m.model.S.future ≈ S_ref rtol = 1e-10 atol = 1e-12
        params.frazil_ice isa RelaxedFrazilIce && @test any(>(0), m.model.frazil)
    end
end

@testset "Conservation: H equation exact over one step" begin
    m = build_isomip(CPU(); nx=20, ny=10, isomipcond=:warm)
    Laddie.advance_leapfrog!(m.model, m.clock.dt)
    H_past = copy(m.model.H.past)
    src    = copy((m.model.convH .+ m.model.melt .+ m.model.nentr) .* m.model.tmask)
    Laddie.leapfrog_step!(m.model, m.clock.dt, 2)
    @test m.model.H.future ≈ H_past .+ src .* (2 * m.clock.dt)
end

const FBRK3_PARAMS = (; integrator = FBRK3Integrator())

@testset "Conservation: FB-RK3 H equation exact over one stage" begin
    m = build_isomip(CPU(); nx = 20, ny = 10, isomipcond = :warm, params = Params(; FBRK3_PARAMS...))
    run!(m; days = 0.2, verbose = false)   # develop a non-trivial flow
    model, dt = m.model, m.clock.dt
    β = model.params.integrator.beta3
    H_past = copy(model.H.past)
    Laddie._fbrk3_stage!(model, dt, (β, β))   # the last stage, over Δt
    src = (model.convH .+ model.melt .+ model.nentr .- model.detr_cap) .* model.tmask
    @test model.H.future ≈ H_past .+ src .* dt
    # The blend H*** = β₃Hⁿ⁺¹ + (1 − 2β₃)Hⁿ⁺¹ᐟ² + β₃Hⁿ that the momentum step saw.
    @test model.Hfb ≈ β .* model.H.future .+ (1 - 2β) .* model.H.present .+ β .* model.H.past
end

@testset "Flux forms: a uniform temperature stays uniform" begin
    # No melt (γ_T = 0) and ambient water at the layer's temperature: T then changes only
    # if the tracer fluxes disagree with the continuity equation.  FB-RK3 takes both with
    # the stage thickness (v2 takes the tracer flux with H*, D1 of the plan).
    isomip = ISOMIPForcing(:warm)
    ocean = OceanForcing1D(fill(0.5, length(isomip.Tz)), isomip.Sz, isomip.z)
    grid = build_isomip(CPU(); nx = 20, ny = 10).model.grid
    for schemes in ((;), FBRK3_PARAMS)
        params = Params(; melting = FixedGamTMelting(0.0), schemes...)
        sim = Simulation(Model(grid; forcing = ocean, params))
        run!(sim; days = 1.0, verbose = false)
        active = sim.model.tmask .> 0
        @test meltstats(sim).max_speed > 0.01   # the fresher layer flows
        @test maximum(abs, sim.model.T.present[active] .- 0.5) < 1e-12
    end
end

@testset "FB-RK3: first order in time, closer to the converged run than the leapfrog" begin
    # Both schemes are first order here: the lateral terms (from level n in every FB-RK3
    # stage, forward Euler), the implicit sinks and the preventive H bounds are.  Measured
    # 2026-11-20 on this setup over 0.5 d: H errors 8.6e-4, 4.0e-4 (FB-RK3) against
    # 4.7e-3, 2.3e-3 (leapfrog) at Δt = 240, 120 s.
    present = PresentClosureLevel()
    run_at(params, dt) = (s = build_isomip(CPU(); nx = 20, ny = 10, isomipcond = :warm, params, dt);
                          run!(s; days = 0.25, verbose = false); copy(s.model.H.present))
    relerr(a, b) = maximum(abs, a .- b) / maximum(abs, b)
    p_fb = Params(; FBRK3_PARAMS...)
    p_lf = Params(; integrator = LeapfrogIntegrator(; closure_level = present))
    ref = run_at(p_fb, 15.0)
    e240, e120, e60 = (relerr(run_at(p_fb, dt), ref) for dt in (240.0, 120.0, 60.0))
    @test 0.8 < log2(e240 / e120) < 1.5
    @test 0.8 < log2(e120 / e60) < 1.6
    @test e120 < relerr(run_at(p_lf, 120.0), ref) / 2
    # Both converge to the same run.
    @test relerr(run_at(p_lf, 15.0), ref) < 1e-3
end

@testset "Maximum layer thickness: preventive detrainment" begin
    # A binding cap is met by detraining layer water before the step, so the
    # stepped H lands on the cap without the corrective clamp.  That the detrained
    # water leaves at the layer's T and S is checked by the fused-kernel test.
    H_cap = FT(5.0)   # ISOMIP H reaches ~11 m here, so this binds
    m = build_isomip(CPU(); FT, nx=20, ny=10, isomipcond=:warm,
                     params = Params(; FT, max_layer_thickness = AbsoluteMaxLayerThickness(H_cap)))
    run!(m; days = 0.5, verbose = false)
    mm = m.model
    Laddie.advance_leapfrog!(mm, m.clock.dt)
    capped = (mm.detr_cap .> 0) .& (mm.tmask .> 0)
    @test any(capped)
    dt2 = 2 * m.clock.dt
    H_past = copy(mm.H.past)
    src = copy((mm.convH .+ mm.melt .+ mm.nentr .- mm.detr_cap) .* mm.tmask)
    Laddie.step_thickness!(mm, dt2)
    @test mm.H.future ≈ H_past .+ src .* dt2
    @test all(mm.H.future[capped] .≈ H_cap)    # lands on the cap before any clamp
    @test maximum(mm.H.future) <= H_cap * (1 + 10eps(FT))
    @test all(mm.detr_cap[mm.tmask .== 0] .== 0)
    run!(m; days = 0.5, verbose = false)
    @test maximum(m.model.H.present) <= H_cap * (1 + 10eps(FT))
end

@testset "Maximum layer thickness: no cap leaves detr_cap at zero" begin
    m = build_isomip(CPU(); FT, nx=20, ny=10, isomipcond=:warm)
    run!(m; days = 0.2, verbose = false)
    @test all(iszero, m.model.detr_cap)
end

@testset "Conservation: H ≥ H_min after 1-day run" begin
    m = build_isomip(CPU(); nx=20, ny=10, isomipcond=:warm)
    run!(m; days=1.0, verbose=false)
    active = m.model.tmask .> 0
    @test all(m.model.H.present[active] .>= m.model.H_min - 1e-10)
end

@testset "Conservation: H ≥ H_min after 1-day run (cold)" begin
    m = build_isomip(CPU(); nx=20, ny=10, isomipcond=:cold)
    run!(m; days=1.0, verbose=false)
    active = m.model.tmask .> 0
    @test all(m.model.H.present[active] .>= m.model.H_min - 1e-10)
end

@testset "Float32 vs Float64: mean melt within 1%" begin
    m64 = build_isomip(CPU(); FT=Float64, nx=20, ny=10, isomipcond=:warm)
    m32 = build_isomip(CPU(); FT=Float32, nx=20, ny=10, isomipcond=:warm)
    run!(m64; days=2.0, verbose=false)
    run!(m32; days=2.0, verbose=false)
    _, mn64, _ = meltstats(m64)
    _, mn32, _ = meltstats(m32)
    @test isfinite(mn64) && mn64 > 0
    @test isfinite(mn32)
    @test abs(Float64(mn32) - mn64) / mn64 < 0.01
end
