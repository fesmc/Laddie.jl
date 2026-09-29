# Numerics: fused kernels vs reference terms, conservation, precision.

@testset "Fused kernels match reference equation terms" begin
    # The fused step kernels in numerics.jl and the equation-term
    # functions in equation_terms.jl implement the same governing equations.
    # Reconstruct one leapfrog step from the term functions and require
    # the kernels to reproduce it, for both the scalar-coefficient
    # (FixedGamTMelting/ResetToAmbient) and matrix-coefficient
    # (TurbulentGamTMelting/RelaxToAmbient) kernel variants.
    configs = (
        Params(; FT),
        Params(; FT,
               melting = TurbulentGamTMelting(FT(13.8), FT(2432.0), FT(1.95e-6)),
               convection_scheme = RelaxToAmbient(FT(10000.0)),
               entrainment  = HollandEntrainment(FT(0.01775))),
    )
    for params in configs
        m = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm, params)
        run!(m; days = 0.2, verbose = false)   # develop a non-trivial flow
        Laddie.advance_leapfrog!(m.model, m.clock.dt)
        dt = 2 * m.clock.dt
        Laddie.step_thickness!(m.model, dt)
        Laddie.precompute_integration_terms!(m.model, m.clock.dt)

        rhs_U = .- u_thickness_tendency(m.model) .+ u_advection(m.model) .-
                   u_pressure_depth(m.model)     .+ u_pressure_slope(m.model) .-
                   u_pressure_density(m.model)   .+ u_coriolis(m.model) .-
                   u_bottom_drag(m.model)        .+ u_diffusion(m.model) .-
                   u_detrainment(m.model)
        U_ref = m.model.U.past .+
            Laddie.div0(rhs_U, Laddie.ip_t(m.model, m.model.D.present)) .* m.model.umask .* dt

        rhs_V = .- v_thickness_tendency(m.model) .+ v_advection(m.model) .-
                   v_pressure_depth(m.model)     .+ v_pressure_slope(m.model) .-
                   v_pressure_density(m.model)   .- v_coriolis(m.model) .-
                   v_bottom_drag(m.model)        .+ v_diffusion(m.model) .-
                   v_detrainment(m.model)
        V_ref = m.model.V.past .+
            Laddie.div0(rhs_V, Laddie.jp_t(m.model, m.model.D.present)) .* m.model.vmask .* dt

        rhs_T = .- tracer_thickness_tendency(m.model, m.model.T.present) .+
                   tracer_advection(m.model, m.model.T.present) .+
                   tracer_entrainment(m.model, m.model.Ta) .+
                   T_ice_ocean_exchange(m.model) .+
                   tracer_diffusion(m.model, m.model.T.past) .-
                   tracer_convection(m.model, m.model.T.past, m.model.Ta)
        T_ref = m.model.T.past .+ Laddie.div0(rhs_T, m.model.D.present) .* m.model.tmask .* dt

        rhs_S = .- tracer_thickness_tendency(m.model, m.model.S.present) .+
                   tracer_advection(m.model, m.model.S.present) .+
                   tracer_entrainment(m.model, m.model.Sa) .+
                   tracer_diffusion(m.model, m.model.S.past) .-
                   tracer_convection(m.model, m.model.S.past, m.model.Sa)
        S_ref = m.model.S.past .+ Laddie.div0(rhs_S, m.model.D.present) .* m.model.tmask .* dt

        Laddie.step_u_momentum!(m.model, dt)
        Laddie.step_v_momentum!(m.model, dt)
        Laddie.step_temperature!(m.model, dt)
        Laddie.step_salinity!(m.model, dt)

        @test m.model.U.future ≈ U_ref rtol = 1e-10 atol = 1e-12
        @test m.model.V.future ≈ V_ref rtol = 1e-10 atol = 1e-12
        @test m.model.T.future ≈ T_ref rtol = 1e-10 atol = 1e-12
        @test m.model.S.future ≈ S_ref rtol = 1e-10 atol = 1e-12
    end
end

@testset "Conservation: D equation exact over one step" begin
    m = build_isomip(CPU(); nx=20, ny=10, isomipcond=:warm)
    Laddie.advance_leapfrog!(m.model, m.clock.dt)
    D_past = copy(m.model.D.past)
    src    = copy((m.model.convD .+ m.model.melt .+ m.model.nentr) .* m.model.tmask)
    Laddie.leapfrog_step!(m.model, m.clock.dt, 2)
    @test m.model.D.future ≈ D_past .+ src .* (2 * m.clock.dt)
end

@testset "Conservation: D ≥ D_min after 1-day run" begin
    m = build_isomip(CPU(); nx=20, ny=10, isomipcond=:warm)
    run!(m; days=1.0, verbose=false)
    active = m.model.tmask .> 0
    @test all(m.model.D.present[active] .>= m.model.D_min - 1e-10)
end

@testset "Conservation: D ≥ D_min after 1-day run (cold)" begin
    m = build_isomip(CPU(); nx=20, ny=10, isomipcond=:cold)
    run!(m; days=1.0, verbose=false)
    active = m.model.tmask .> 0
    @test all(m.model.D.present[active] .>= m.model.D_min - 1e-10)
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
