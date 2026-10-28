# Physical parameterizations: melt, entrainment, convection, viscosity, layer caps, Coriolis.

@testset "ISOMIP+ warm cavity: build and basic physics" begin
    # Small grid (nx=20, ny=10) for a fast smoke test
    m = build_isomip(CPU(); nx=20, ny=10, isomipcond=:warm)

    @test size(m.model.tmask) == (22, 12)   # nx+2 × ny+2
    @test all(isfinite, m.model.melt)
    @test all(m.model.melt[m.model.tmask .> 0] .>= 0)   # melt rate non-negative under ice

    mx, mn, sp = meltstats(m)
    @test isfinite(mx) && isfinite(mn) && isfinite(sp)
    @test mx >= mn >= 0
end

@testset "ISOMIP+ cold cavity: build" begin
    m = build_isomip(CPU(); nx=20, ny=10, isomipcond=:cold)
    @test all(isfinite, m.model.melt)
end

@testset "Physical ordering: warm mean melt exceeds cold" begin
    mw = build_isomip(CPU(); nx=20, ny=10, isomipcond=:warm)
    mc = build_isomip(CPU(); nx=20, ny=10, isomipcond=:cold)
    _, mn_warm, _ = meltstats(mw)
    _, mn_cold, _ = meltstats(mc)
    @test mn_warm > mn_cold
end

@testset "TurbulentGamTMelting: build and short run" begin
    params = Params(;
        FT,
        melting = TurbulentGamTMelting(FT(13.8), FT(2432.0), FT(1.95e-6)),
        entrainment  = GasparEntrainment(FT(2.5)),
        convection_scheme = ResetToAmbient(FT(0.005)),
    )
    m = build_isomip(CPU(); FT, nx=20, ny=10, isomipcond=:warm, params)
    @test all(isfinite, m.model.melt)
    @test all(m.model.melt[m.model.tmask .> 0] .>= 0)
    run!(m; days=0.5, verbose=false)
    @test all(isfinite, m.model.H.present)
    @test all(isfinite, m.model.melt)
    @test all(m.model.melt[m.model.tmask .> 0] .>= 0)
end

@testset "TurbulentGamTMelting: transfer coefficients stay positive" begin
    # A Prandtl or Schmidt number that leaves no positive offset is rejected.
    @test_throws ArgumentError TurbulentGamTMelting(0.5, 2432.0, 1.95e-6)
    @test_throws ArgumentError TurbulentGamTMelting(13.8, 0.5, 1.95e-6)
    @test_throws ArgumentError TurbulentGamTMelting(13.8, 2432.0, 0.0)
    @test_throws ArgumentError Params(; melting = TurbulentGamTMelting(; Pr = 0.5))

    # A thin, slow layer (u★H/ν₀ ≪ 1).  With Pr = 1 the unfloored log term would
    # make the γT denominator negative; the floor leaves γ = u★/offset.
    mp = TurbulentGamTMelting(FT(1.0), FT(2432.0), FT(1.95e-6))
    m = build_isomip(CPU(); FT, nx = 20, ny = 10, params = Params(; FT, melting = mp)).model
    act = m.tmask .> 0
    m.ustar .= FT(1e-9) .* m.tmask
    Laddie._compute_turbulent_transfer_coefficients!(m, mp)
    @test all(m.gamT[act] .≈ FT(1e-9) / Laddie._log_layer_offset(mp.Pr))
    @test all(m.gamS[act] .≈ FT(1e-9) / Laddie._log_layer_offset(mp.Sc))
    @test all(iszero, m.gamT[.!act])
end

@testset "UStarGamTMelting: γ scales with u★" begin
    mp = UStarGamTMelting(FT(3.0e-2))
    m = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                     params = Params(; FT, melting = mp))
    run!(m; days = 0.5, verbose = false)
    mm = m.model
    act = mm.tmask .> 0
    @test all(isfinite, mm.H.present) && all(isfinite, mm.melt)
    @test all(mm.melt[act] .>= 0)
    @test mm.gamT[act] == mp.Gamma_T .* mm.ustar[act]
    @test all(iszero, mm.gamT[.!act])
    @test mm.gamS == mm.gamT ./ 35
    @test any(>(0), mm.melt)
end

@testset "HollandEntrainment: build and short run" begin
    params = Params(;
        FT,
        entrainment  = HollandEntrainment(FT(0.01775)),
        melting = FixedGamTMelting(FT(0.00018)),
        convection_scheme = ResetToAmbient(FT(0.005)),
    )
    m = build_isomip(CPU(); FT, nx=20, ny=10, isomipcond=:warm, params)
    @test all(isfinite, m.model.melt)
    @test all(m.model.melt[m.model.tmask .> 0] .>= 0)
    run!(m; days=0.5, verbose=false)
    @test all(isfinite, m.model.H.present)
    @test all(isfinite, m.model.melt)
    @test all(m.model.melt[m.model.tmask .> 0] .>= 0)
end

@testset "Entrainment: Gaspar is default, μ changes the solution" begin
    # GasparEntrainment reproduces the reference LADDIE production term
    # (2μ u★³/(g H δρ)) and must be the default so the Python verification holds.
    @test build_isomip(CPU(); nx=20, ny=10, isomipcond=:cold).model.entrainment isa
          GasparEntrainment
    mk(ep) = build_isomip(CPU(); FT, nx=20, ny=10, isomipcond=:warm,
        gradient = PyGradient(),
        params = Params(; FT, entrainment = ep, melting = FixedGamTMelting(FT(0.00018)),
                        convection_scheme = ResetToAmbient(FT(0.005))))
    m1 = mk(GasparEntrainment(FT(2.5)))
    m2 = mk(GasparEntrainment(FT(1.25)))
    run!(m1; days=0.5, verbose=false)
    run!(m2; days=0.5, verbose=false)
    @test all(isfinite, m1.model.entr) && all(isfinite, m2.model.entr)
    @test all(isfinite, m1.model.melt) && all(isfinite, m2.model.melt)
    @test !isapprox(m1.model.entr, m2.model.entr)
end

@testset "Lateral viscosity: PrescribedLateralViscosity bit-identical, NonlinearLateralViscosity differs" begin
    # PrescribedLateralViscosity is the default and must reproduce the
    # pre-AbstractLateralViscosity behaviour bit-for-bit: laplace_U/V now
    # dispatch on m.lateral_viscosity, but the Prescribed path is the exact
    # same kernel followed by the same `.*= A_h` that used to live in the
    # momentum-step kernels.
    m_def  = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm)
    m_presc = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                           params = Params(; FT, lateral_viscosity = PrescribedLateralViscosity()))
    run!(m_def;   days = 1.0, verbose = false)
    run!(m_presc; days = 1.0, verbose = false)
    @test m_presc.model.U.present == m_def.model.U.present
    @test m_presc.model.V.present == m_def.model.V.present
    @test m_presc.model.melt == m_def.model.melt

    # NonlinearLateralViscosity changes the solution and stays physical.
    m_nl = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                        params = Params(; FT, lateral_viscosity = NonlinearLateralViscosity(FT(10.0))))
    run!(m_nl; days = 1.0, verbose = false)
    @test all(isfinite, m_nl.model.H.present) && all(isfinite, m_nl.model.melt)
    @test all(m_nl.model.melt[m_nl.model.tmask .> 0] .>= 0)
    @test m_nl.model.V.present != m_def.model.V.present

    # Decision (a): grounding-line/land wall drag stays linear in the plain
    # A_h even under the nonlinear interior scheme, so switching grounding_line
    # away from the no-slip default must still change the solution under
    # NonlinearLateralViscosity
    # (i.e. the wall-drag term isn't accidentally zeroed or folded into the
    # shear-scaled interior term).
    m_nl_ns = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                           params = Params(; FT, lateral_viscosity = NonlinearLateralViscosity(FT(10.0))),
                           boundary = BoundaryConditions(; grounding_line = FreeSlipGL()))
    run!(m_nl_ns; days = 1.0, verbose = false)
    @test all(isfinite, m_nl_ns.model.H.present) && all(isfinite, m_nl_ns.model.melt)
    @test m_nl_ns.model.V.present != m_nl.model.V.present

    # The coefficient uses the full velocity-difference norm √(ΔU² + ΔV²),
    # as the reference's dUabs does — not just the component being diffused.
    # So changing V alone must change the U-viscosity, which a per-component
    # |ΔU| coefficient could not do.  Scale rather than offset V: the norm
    # sees only *differences*, so a uniform shift would change nothing.  For
    # the same reason the flux is coeff * ΔU, so this is only visible where
    # ΔU is itself nonzero — i.e. in the shelf interior, not at a wall.
    lU_before = copy(Laddie.laplace_U(m_nl.model))
    m_nl.model.V.past .*= 2
    @test Laddie.laplace_U(m_nl.model) != lU_before
    # ...and the same coupling must be absent under the plain Laplacian,
    # whose coefficient is a constant and never reads V at all.
    lU0_before = copy(Laddie.laplace_U(m_def.model))
    m_def.model.V.past .*= 2
    @test Laddie.laplace_U(m_def.model) == lU0_before
end

@testset "PrescribedMelting: prescribed rate, consistent heat sink, 2D field" begin
    @test PrescribedMelting().melt == 0            # keyword default constructs
    spy = Params(; FT).seconds_per_year
    sim = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                       params = Params(; FT, melting = PrescribedMelting(5.0)))
    m = sim.model
    run!(sim; days = 1.0, verbose = false)
    ice = m.imask .> 0
    @test all(m.melt[ice] .== FT(5.0) / spy)
    @test all(m.melt[.!ice] .== 0)
    @test meltstats(sim).mean_meltrate ≈ 5.0
    # The flow still develops: u★ and entrainment are computed as usual.
    @test maximum(m.ustar) > 0 && maximum(m.entr) > 0
    @test all(isfinite, m.T.present) && all(isfinite, m.H.present)
    # Tb is the local freezing point, and the temperature equation's
    # exchange term carries exactly the heat the prescribed melt needs.
    Laddie.update_melt!(m)
    @test m.Tb[ice] ≈ (m.l1 .* m.S.present .+ m.l2 .+ m.l3 .* m.z_draft)[ice]
    heat = @. m.melt * (m.L - m.c_i * (m.T_ice_base - m.Tb)) / m.c_p
    @test (m.gamT .* (m.T.present .- m.Tb))[ice] ≈ heat[ice]
    # Melting cools the layer relative to a run without melt.
    sim0 = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                        params = Params(; FT, melting = PrescribedMelting(0.0)))
    run!(sim0; days = 1.0, verbose = false)
    @test sum(m.T.present[ice]) < sum(sim0.model.T.present[ice])

    # A full-domain 2D field is cropped with the grid; a wrong size is rejected.
    M = fill(2.0, 22, 12); M[12:end, :] .= 8.0
    sim2 = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                        params = Params(; FT, melting = PrescribedMelting(M)))
    Laddie.update_melt!(sim2.model)
    m2 = sim2.model
    @test all(m2.melt[ice] .≈ (M ./ spy)[ice])
    @test_throws ArgumentError build_isomip(CPU(); FT, nx = 20, ny = 10,
        params = Params(; FT, melting = PrescribedMelting(zeros(3, 3))))
    @test_throws ArgumentError build_isomip(CPU(); FT, nx = 20, ny = 10,
        params = Params(; FT, melting = PrescribedMelting(NaN)))
    @test occursin("2.0 … 8.0", sprint(show, PrescribedMelting(M)))
end

@testset "AbsoluteMaxLayerThickness ignores bathymetry" begin
    @test AbsoluteMaxLayerThickness().H_max === 100.0f0
    @test Params(; max_layer_thickness = AbsoluteMaxLayerThickness()).max_layer_thickness.H_max === 100.0
    mk = zeros(Int, 12, 8)
    mk[1, :] .= 1; mk[end, :] .= 1; mk[:, 1] .= 1; mk[:, end] .= 1
    mk[2:3, 2:7] .= 2; mk[4:10, 2:7] .= 3; mk[11, 2:7] .= 0
    z_draft = fill(FT(-300.0), size(mk))
    z_bed = fill(FT(-310.0), size(mk))          # a 10 m water column everywhere
    grid = Grid(mk, z_draft, 1000.0, 1000.0; z_bed, domain_cropping = NoDomainCropping())
    m = Model(grid; forcing = ISOMIPForcing(:warm; FT),
              params = Params(; FT, max_layer_thickness = AbsoluteMaxLayerThickness(50.0)))
    m.H.future .= FT(80.0) .* m.tmask
    Laddie._clamp_thickness!(m)
    act = m.tmask .> 0
    @test all(m.H.future[act] .== 50)           # capped at H_max, not at 10 m
    @test all(m.H.future[.!act] .== 0)
end

@testset "RelativeMaxLayerThickness: fraction of the water column, with a floor" begin
    @test RelativeMaxLayerThickness(0.8).H_floor === 0.0
    @test RelativeMaxLayerThickness(0.8, 10).H_floor === 10.0
    @test Params(; FT = Float32, max_layer_thickness = RelativeMaxLayerThickness(0.8, 10.0)).max_layer_thickness ===
          RelativeMaxLayerThickness(0.8f0, 10.0f0)
    mk = zeros(Int, 12, 8)
    mk[1, :] .= 1; mk[end, :] .= 1; mk[:, 1] .= 1; mk[:, end] .= 1
    mk[2:3, 2:7] .= 2; mk[4:10, 2:7] .= 3; mk[11, 2:7] .= 0
    z_draft = fill(FT(-300.0), size(mk))
    z_bed = fill(FT(-310.0), size(mk))
    z_bed[4:6, :] .= FT(-302.0)                  # a 2 m water column under part of the shelf
    grid = Grid(mk, z_draft, 1000.0, 1000.0; z_bed, domain_cropping = NoDomainCropping())
    act = mk .== 3
    capped(mlt) = (m = Model(grid; forcing = ISOMIPForcing(:warm; FT), params = Params(; FT, max_layer_thickness = mlt));
                   m.H.future .= FT(80.0) .* m.tmask; Laddie._clamp_thickness!(m); m.H.future)
    thin = falses(size(mk)); thin[4:6, :] .= true
    H = capped(RelativeMaxLayerThickness(0.8))
    @test all(H[act .& thin] .≈ 1.6) && all(H[act .& .!thin] .≈ 8)
    H = capped(RelativeMaxLayerThickness(0.8, 5.0))
    @test all(H[act .& thin] .≈ 5) && all(H[act .& .!thin] .≈ 8)   # the floor only where 0.8·wc < 5 m
end

@testset "ClampDensity convection scheme: build and short run" begin
    params = Params(; FT, convection_scheme = ClampDensity(FT(0.005)))
    m = build_isomip(CPU(); FT, nx=20, ny=10, isomipcond=:warm, params)
    @test all(isfinite, m.model.melt)
    run!(m; days=0.5, verbose=false)
    @test all(isfinite, m.model.H.present)
    @test all(isfinite, m.model.melt)
end

@testset "RelaxToAmbient convection scheme: build and short run" begin
    params = Params(; FT, convection_scheme = RelaxToAmbient(FT(10000.0)))
    m = build_isomip(CPU(); FT, nx=20, ny=10, isomipcond=:warm, params)
    @test all(isfinite, m.model.melt)
    run!(m; days=0.5, verbose=false)
    @test all(isfinite, m.model.H.present)
    @test all(isfinite, m.model.melt)
end

@testset "RelaxedFrazilIce: rate, relaxation toward T_f, brine rejection" begin
    @test Params(; FT).frazil_ice === NoFrazilIce()
    @test build_isomip(CPU(); FT, nx = 20, ny = 10).model.frazil === zero(FT)

    @testset "$(nameof(typeof(tracer_form)))" for tracer_form in
            (AdvectiveTracerForm(), FluxTracerForm())
        τ = FT(3600.0)
        params = Params(; FT, frazil_ice = RelaxedFrazilIce(τ),
                        convection_scheme = ClampDensity(FT(0.005)), tracer_form)
        m = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm, params)
        run!(m; days = 0.2, verbose = false)
        mm = m.model
        @test all(iszero, mm.frazil)   # ISOMIP+ is never supercooled on its own

        dT = FT(0.05)
        supercool!(mm, dT)
        dt = m.clock.dt
        Laddie.advance_leapfrog!(mm, dt)
        ice = mm.imask .> 0
        # The rate whose latent heat removes the past level's supercooling over τ.
        @test mm.frazil[ice] ≈ mm.c_p / mm.L .* mm.H.present[ice] .* dT ./ τ
        @test all(iszero, mm.frazil[.!ice])

        # The same step with and without the frazil.  The difference is the frazil
        # alone wherever the H_min floor does not bind (ent2 would top up the loss).
        frz = copy(mm.frazil)
        free = ice .& (mm.ent2 .== 0)
        @test count(free) > 0.9 * count(ice)
        Laddie.leapfrog_step!(mm, dt, 2)
        with = map(copy, (H = mm.H.future, U = mm.U.future, T = mm.T.future, S = mm.S.future))
        fill!(mm.frazil, 0)
        Laddie.update_entrainment!(mm, dt)
        Laddie.leapfrog_step!(mm, dt, 2)

        # Water leaves the layer ...
        @test (with.H .- mm.H.future)[free] ≈ -2dt .* frz[free] rtol = 1e-8
        if tracer_form isa AdvectiveTracerForm
            # ... and its latent heat relaxes T toward T_f at the rate 1/τ ...
            @test (with.T .- mm.T.future)[free] ≈ fill(2dt * dT / τ, count(free)) rtol = 1e-8
            # ... while its salt stays behind ...
            @test (with.S .- mm.S.future)[free] ≈
                  (2dt .* mm.S.present .* frz ./ mm.H.present)[free] rtol = 1e-8
        else
            # The flux form divides the content by the new thickness plus the implicit
            # outflow, D = H⁺ + 2dt·(outflow + detr_cap) without the frazil (`m.Hq` holds
            # the outflow rate after the step).  The frazil takes 2dt·frazil off H⁺ and adds
            # it to the sink, so for T the denominator is unchanged and only its latent
            # heat is added: ΔT = 2dt·frazil·L/(c_p D) = 2dt·dT/τ · H/D ...
            D = mm.H.future .+ 2dt .* (mm.Hq .+ mm.detr_cap)
            @test (with.T .- mm.T.future)[free] ≈
                  (2dt .* frz .* mm.L ./ (mm.c_p .* D))[free] rtol = 1e-8
            # ... while its salt stays behind: the same content over a thinner layer.
            @test (with.S .* (D .- 2dt .* frz))[free] ≈ (mm.S.future .* D)[free] rtol = 1e-8
        end
        # ... and it carries its own momentum, so the velocity does not change.
        free_u = free .& circshift(free, (-1, 0))
        @test with.U[free_u] ≈ mm.U.future[free_u] rtol = 1e-10 atol = 1e-14

        # Stays finite and reduces the supercooling over a longer run.
        sc(mm) = maximum((mm.l1 .* mm.S.present .+ mm.l2 .+ mm.l3 .* mm.z_draft .-
                          mm.T.present)[ice])
        run!(m; days = 0.5, verbose = false)
        @test all(isfinite, mm.H.present) && all(isfinite, mm.T.present)
        @test sc(mm) < dT / 2
    end
end

@testset "Coriolis: 0D/2D defaults agree, latitude varies f, C-grid staggering" begin
    # The whole point of the default latitude being derived from the default f
    # rather than rounded to -70: the two options must agree exactly, so
    # switching to the geographic statement changes nothing until lat is set.
    @test Laddie.DEFAULT_LATITUDE ≈ -69.95 atol = 0.01   # ~70°S, not exactly
    @test 2 * Laddie.EARTH_ROTATION_RATE * sind(Laddie.DEFAULT_LATITUDE) ≈
          Laddie.DEFAULT_CORIOLIS_F
    iso(cp) = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                           params = Params(; FT, coriolis = cp))
    m0 = iso(CoriolisParameter0D())
    m2 = iso(CoriolisParameter2D())
    @test m0.model.f == m2.model.f
    run!(m0; days = 1.0, verbose = false)
    run!(m2; days = 1.0, verbose = false)
    @test m2.model.melt == m0.model.melt && m2.model.V.present == m0.model.V.present

    # ...and the default is what Params used to hold as the scalar `f`.
    @test all(m0.model.f .== -1.37e-4)
    @test !hasfield(typeof(getfield(m0.model, :params)), :f)

    # A scalar latitude is still an f-plane, but a different one.
    m75 = iso(CoriolisParameter2D(-75.0))
    @test all(m75.model.f .≈ 2 * Laddie.EARTH_ROTATION_RATE * sind(-75.0))
    run!(m75; days = 1.0, verbose = false)
    @test m75.model.V.present != m0.model.V.present          # stronger rotation, different flow
    @test all(isfinite, m75.model.melt)

    # Uniform f: the staggered copies equal the T-point field exactly.
    @test m0.model.fu == m0.model.f && m0.model.fv == m0.model.f

    # 2D latitude: f must be staggered onto the two velocity faces separately,
    # because on a C-grid U and V do not share a point.  A latitude varying in
    # y makes fv differ from f while fu (an x-average) does not.
    mask0 = copy(m0.model.mask)
    nx_t, ny_t = size(mask0)
    lat_y = [FT(-80 + 10 * (j - 1) / (ny_t - 1)) for _ in 1:nx_t, j in 1:ny_t]
    grid0 = Grid(mask0, copy(m0.model.z_draft), 2000.0, 2000.0; FT,
                 domain_cropping = NoDomainCropping())
    m_y = Simulation(Model(grid0; forcing = ISOMIPForcing(:warm; FT),
                           params = Params(; FT, coriolis = CoriolisParameter2D(lat_y))))
    @test m_y.model.f[1, 1] ≈ 2 * Laddie.EARTH_ROTATION_RATE * sind(-80.0)
    @test m_y.model.fu == m_y.model.f                        # constant along x
    @test m_y.model.fv != m_y.model.f                        # averaged across y
    @test m_y.model.fv[1, 1] ≈ (m_y.model.f[1, 1] + m_y.model.f[1, 2]) / 2
    run!(m_y; days = 1.0, verbose = false)
    @test all(isfinite, m_y.model.melt) && all(m_y.model.melt[m_y.model.imask .> 0] .>= 0)
    @test m_y.model.V.present != m0.model.V.present

    # The equivalent x-varying field swaps which face average is trivial.
    lat_x = [FT(-80 + 10 * (i - 1) / (nx_t - 1)) for i in 1:nx_t, _ in 1:ny_t]
    m_x = Simulation(Model(grid0; forcing = ISOMIPForcing(:warm; FT),
                           params = Params(; FT, coriolis = CoriolisParameter2D(lat_x))))
    @test m_x.model.fv == m_x.model.f && m_x.model.fu != m_x.model.f

    # A 2D latitude is cropped with the mask, not silently mismatched.
    m_crop = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                          params = Params(; FT, coriolis = CoriolisParameter2D(-70.0)),
                          domain_cropping = MinRectangleDomainCropping(margin = 2))
    @test size(m_crop.model.f) == size(m_crop.model.tmask)

    # Validation.
    @test_throws ArgumentError iso(CoriolisParameter2D(-120.0))
    @test_throws ArgumentError iso(CoriolisParameter2D(zeros(FT, 3, 3)))

    # Float32 promotion follows Params, like every other parameterization.
    @test Params(; FT = Float32, coriolis = CoriolisParameter0D()).coriolis isa
          CoriolisParameter0D{Float32}
end

@testset "Tracer form: advective (v1) vs flux (v2)" begin
    @test Params().tracer_form isa FluxTracerForm
    @test Params(; FT, tracer_form = AdvectiveTracerForm()).tracer_form isa AdvectiveTracerForm

    # Both forms discretise the same budget, but the advective one carries a larger
    # time-step error: on 2-day warm ISOMIP+, cutting dt from 210 s to 52.5 s moved
    # the mean melt by 7.3 % under the advective form and by 0.36 % under the flux
    # form, and the two forms meet as dt shrinks (5.22 vs 5.35 m/yr at 52.5 s).
    melt(tf, dt) = (m = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm, dt,
                                     params = Params(; FT, tracer_form = tf));
                    run!(m; days = 2.0, verbose = false); meltstats(m)[2])
    flux = melt(FluxTracerForm(), 210.0), melt(FluxTracerForm(), 52.5)
    adv = melt(AdvectiveTracerForm(), 210.0), melt(AdvectiveTracerForm(), 52.5)
    @test abs(flux[1] - flux[2]) / flux[2] < 0.01
    @test abs(adv[1] - adv[2]) / adv[2] > 5 * abs(flux[1] - flux[2]) / flux[2]
    @test abs(flux[2] - adv[2]) / adv[2] < 0.05
end

@testset "Flux tracer form: the clamp moves water with its tracer" begin
    upd(H_future, H_star; q_past = 2.0, H_past = 4.0, rhs = 0.5, qa = 7.0, tmask = 1.0, dt = 2.0) =
        Laddie._tracer_update(FluxTracerForm(), q_past, q_past, rhs, 0.0, NaN, H_past, H_future, H_star, qa, tmask, dt)
    content = 4.0 * 2.0 + 0.5 * 2.0                     # H⁻q⁻ + rhs·dt = 9
    @test upd(6.0, 6.0) ≈ content / 6                   # unclamped: content / H⁺
    @test upd(3.0, 6.0) ≈ content / 6                   # cut at the cap: q unchanged
    @test upd(8.0, 6.0) ≈ (content + 2 * 7.0) / 8       # raised by the floor: 2 m of ambient
    @test upd(1.0, -0.5) ≈ (content + 1.5 * 7.0) / 1    # H* below zero: as v2's entr_dmin
    @test upd(3.0, 6.0; tmask = 0.0) == 2.0             # masked: q⁻ kept
    # The cut leaves the old relation q = content / H*, whatever the clamp's size.
    @test upd(1.0, 50.0) ≈ content / 50
end

@testset "Flux tracer form: outflow taken at the new level" begin
    # A 1 m cell flushed by fresh water at 50 times its volume per step: inflow
    # equals outflow, so H stays 1 m.  Explicit, the outflow of the present salt
    # would leave a negative content; implicit, q⁺ is the mean of the old and the
    # incoming water, weighted by their volumes.
    sink, dt, q = 50.0, 1.0, 34.0
    rhs = sink * 0.0 - sink * q                           # inflow at q = 0, outflow at q
    upd(; kw...) = Laddie._tracer_update(FluxTracerForm(), q, q, rhs, sink, NaN, 1.0, 1.0, 1.0,
                                         34.5, 1.0, dt; kw...)
    @test upd() ≈ q / (1 + sink * dt)                     # 1 m of old water, 50 m of fresh
    @test 0 <= upd() <= q
    @test (1.0 * q + rhs * dt) / 1.0 < 0                  # what the explicit update gives
    # With no outflow the update is the explicit one.
    @test Laddie._tracer_update(FluxTracerForm(), 2.0, 3.0, 0.5, 0.0, NaN, 4.0, 4.5, 4.5, 7.0, 1.0, 2.0) ≈
          (4.0 * 2.0 + 0.5 * 2.0) / 4.5
end

@testset "Momentum advection: centred (v1) vs upstream (v2)" begin
    @test Params().momentum_advection isa CentredMomentumAdvection
    @test Params(; FT, momentum_advection = UpstreamMomentumAdvection()).momentum_advection isa
          UpstreamMomentumAdvection

    mk(ma) = build_isomip(CPU(); FT, nx = 40, ny = 20, isomipcond = :warm,
        params = Params(; FT, melting = FixedGamTMelting(0.00018),
                        momentum_advection = ma)).model

    # Consistency with discrete continuity: a uniform velocity field on a uniform
    # layer has no advective tendency wherever the stencil sees only active
    # cells.  This is what a centred face thickness with an upstream velocity
    # would break.
    for ma in (CentredMomentumAdvection(), UpstreamMomentumAdvection())
        m = mk(ma)
        m.U.present .= 1.0
        m.V.present .= 0.0
        m.H.present .= 10.0
        au = Array(copy(Laddie.momentum_advection_U(m)))
        av = Array(copy(Laddie.momentum_advection_V(m)))
        um = Array(m.umask)
        interior = [
            2 <= i <= size(um, 1) - 1 && 2 <= j <= size(um, 2) - 1 &&
            um[i, j] > 0 && um[i+1, j] > 0 && um[i-1, j] > 0 &&
            um[i, j+1] > 0 && um[i, j-1] > 0
            for i in axes(um, 1), j in axes(um, 2)
        ]
        @test count(interior) > 100
        @test maximum(abs, au[interior]) == 0
        @test maximum(abs, av) == 0
    end

    # Donor-cell upwinding is dissipative where the centred scheme is not, so it
    # cannot leave the flow more energetic.
    run1(ma) = (s = build_isomip(CPU(); FT, nx = 40, ny = 20, isomipcond = :warm,
                    params = Params(; FT, melting = FixedGamTMelting(0.00018),
                                    momentum_advection = ma));
                run!(s; days = 1.0, verbose = false); meltstats(s))
    centred = run1(CentredMomentumAdvection())
    upstream = run1(UpstreamMomentumAdvection())
    @test upstream.max_speed <= centred.max_speed
    @test upstream.mean_meltrate ≈ centred.mean_meltrate rtol = 0.05
end

@testset "AbstractIceConduction: closures, heat balance, warnings" begin
    quiet(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())

    # The default is the Holland & Jenkins linearisation and warns of nothing; the
    # others warn (once per session under the console logger, through `maxlog`).
    @test Params(; FT).ice_conduction === HollandJenkinsConduction()
    @test_logs HollandJenkinsConduction()
    @test_logs (:warn, r"experimental ice-conduction closure") MeltOnlyConduction()
    @test_logs (:warn, r"LinearProfileConduction") LinearProfileConduction()

    none, meltonly, lin = quiet() do
        NoConduction(), MeltOnlyConduction(), LinearProfileConduction(FT(2.1))
    end
    @test lin.conductivity === FT(2.1)
    @test quiet(() -> LinearProfileConduction().conductivity) === 2.1
    @test sprint(show, lin) == "LinearProfileConduction(conductivity = 2.1 W/m/K)"
    @test sprint(show, none) == "NoConduction()"
    @test occursin("ice conduction = HollandJenkinsConduction()",
                   sprint(show, MIME("text/plain"), Params(; FT)))
    # Promotion to the precision of the Params.
    p32 = quiet(() -> Params(; FT = Float32, ice_conduction = LinearProfileConduction(2.1)))
    @test p32.ice_conduction.conductivity === 2.1f0

    # The balance at the interface, for a melting and a supercooled (freezing) layer:
    #   c_p γT (T − Tb) = ṁ (L − c_i,eff (T_i − Tb)) + q,   γS (S − Sb) = ṁ Sb,
    # with Tb the liquidus at Sb, and c_i,eff and q set by the closure.
    function balance(closure, fmelt, ffreeze, k; supercooled)
        sim = quiet() do
            build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                         params = Params(; FT, ice_conduction = closure))
        end
        m = sim.model
        Tf = @. m.l1 * m.S.present + m.l2 + m.l3 * m.z_draft
        supercooled && (m.T.present .= ifelse.(m.imask .> 0, Tf .- 0.05, m.T.present))
        Laddie.update_melt!(m)
        ice = m.imask .> 0
        ci = @. m.c_i * ifelse(m.T.present > Tf, fmelt, ffreeze)
        Hi = @. -m.z_draft * m.rho0_seawater / m.rho_ice
        q = @. k * (Tf - m.T_ice_base) / (Hi * m.rho_freshwater)
        Sb = @. m.gamS * m.S.present / (m.gamS + m.melt)
        # Tb is the liquidus at Sb exactly where there is no melt-proportional conduction.
        # Where there is, the kernel's quadratic takes the m² coefficient as L − c_i T_i
        # rather than L − c_i (T_i − T̃_f), a ~1e-3 K approximation of long standing.
        Tliq = @. m.l1 * Sb + m.l2 + m.l3 * m.z_draft
        if all(iszero, ci[ice])
            @test m.Tb[ice] ≈ Tliq[ice]
        else
            @test maximum(abs, (m.Tb .- Tliq)[ice]) < 1e-2
        end
        lhs = @. m.c_p * m.gamT * (m.T.present - m.Tb)
        rhs = @. m.melt * (m.L - ci * (m.T_ice_base - m.Tb)) + q
        @test lhs[ice] ≈ rhs[ice]
        @test all(isfinite, m.melt) && all(isfinite, m.Tb)
        sign = supercooled ? -1 : 1
        @test all(sign .* m.melt[ice] .> 0)
        return copy(m.melt)
    end

    cases = ((HollandJenkinsConduction(), 1, 1, 0), (meltonly, 1, 0, 0),
             (none, 0, 0, 0), (lin, 0, 0, FT(2.1)))
    melt = Dict(sc => [balance(c, f1, f2, k; supercooled = sc) for (c, f1, f2, k) in cases]
                for sc in (false, true))

    # Where melting, MeltOnly is the default closure; where freezing, it is NoConduction.
    @test melt[false][2] == melt[false][1]
    @test melt[true][2] == melt[true][3]
    # No conduction lets L_eff fall to L, so the melt is larger than with the colder-ice
    # latent heat (by about c_i |T_i| / L, 15 % for T_i = -25 °C).
    ice = build_isomip(CPU(); FT, nx = 20, ny = 10).model.imask .> 0
    @test all(melt[false][3][ice] .> melt[false][1][ice])
    # The linear profile conducts a fixed heat sink: less melt than without conduction.
    @test all(melt[false][4][ice] .< melt[false][3][ice])

    # The prescribed melt carries the same conduction in its heat sink.
    for (c, f1, f2, k) in cases
        sim = quiet() do
            build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                         params = Params(; FT, melting = PrescribedMelting(5.0),
                                         ice_conduction = c))
        end
        m = sim.model
        Laddie.update_melt!(m)
        Tf = @. m.l1 * m.S.present + m.l2 + m.l3 * m.z_draft
        ci = @. m.c_i * ifelse(m.T.present > Tf, f1, f2)
        Hi = @. -m.z_draft * m.rho0_seawater / m.rho_ice
        q = @. k * (Tf - m.T_ice_base) / (Hi * m.rho_freshwater)
        heat = @. (m.melt * (m.L - ci * (m.T_ice_base - m.Tb)) + q) / m.c_p
        @test (m.gamT .* (m.T.present .- m.Tb))[ice] ≈ heat[ice]
    end

    # Every closure runs.
    for (c, _, _, _) in cases
        sim = quiet() do
            build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                         params = Params(; FT, ice_conduction = c))
        end
        run!(sim; days = 0.5, verbose = false)
        @test all(isfinite, sim.model.H.present) && all(isfinite, sim.model.melt)
    end
end

@testset "UStarGamTMelting ratio: default 35, validation, γS = γT / ratio" begin
    quiet(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())
    @test UStarGamTMelting(FT(3.0e-2)).ratio === FT(35)
    @test UStarGamTMelting().ratio === 35.0
    @test_logs UStarGamTMelting(3.0e-2, 35)          # the default ratio does not warn
    @test_logs (:warn, r"heat-to-salt transfer ratio") UStarGamTMelting(3.0e-2, 70)
    mp = quiet(() -> UStarGamTMelting(FT(3.0e-2), 70))   # mixed types promote
    @test mp === quiet(() -> UStarGamTMelting(FT(3.0e-2), FT(70)))
    @test_throws ArgumentError UStarGamTMelting(3.0e-2, -1.0)
    @test_throws ArgumentError UStarGamTMelting(3.0e-2, 0.0)
    p32 = quiet(() -> Params(; FT = Float32, melting = UStarGamTMelting(3.0e-2, 70.0)))
    @test p32.melting === UStarGamTMelting{Float32}(3.0f-2, 70.0f0)

    sim = quiet() do
        build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                     params = Params(; FT, melting = mp))
    end
    run!(sim; days = 0.5, verbose = false)
    mm = sim.model
    act = mm.tmask .> 0
    @test mm.gamT[act] == mp.Gamma_T .* mm.ustar[act]
    @test mm.gamS == mm.gamT ./ 70
    @test all(isfinite, mm.melt)
end

@testset "TwoEquationMelting: linear heat balance at Tf(S), three-equation limit" begin
    quiet(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())
    @test_logs (:warn, r"TwoEquationMelting is an experimental") TwoEquationMelting()
    @test quiet(() -> TwoEquationMelting().Gamma_TS) === 0.0178
    @test_throws ArgumentError TwoEquationMelting(0.0)
    p32 = quiet(() -> Params(; FT = Float32, melting = TwoEquationMelting(0.01)))
    @test p32.melting === TwoEquationMelting{Float32}(0.01f0)

    mk(; melting, conduction = HollandJenkinsConduction(), supercooled = false) = begin
        sim = quiet() do
            build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                         params = Params(; FT, melting, ice_conduction = conduction))
        end
        m = sim.model
        run!(sim; days = 0.2, verbose = false)
        Tf = @. m.l1 * m.S.present + m.l2 + m.l3 * m.z_draft
        supercooled && (m.T.present .= ifelse.(m.imask .> 0, Tf .- 0.05, m.T.present))
        Laddie.update_melt!(m)
        m, Tf
    end

    Γ = FT(0.0178)
    for (conduction, fmelt, ffreeze, k) in (
        (HollandJenkinsConduction(), 1, 1, 0),
        (quiet(() -> MeltOnlyConduction()), 1, 0, 0),
        (quiet(() -> LinearProfileConduction(FT(2.1))), 0, 0, FT(2.1)),
    ), supercooled in (false, true)
        m, Tf = mk(; melting = quiet(() -> TwoEquationMelting(Γ)), conduction, supercooled)
        ice = m.imask .> 0
        # γ_TS = Γ_TS u★, interface at the freezing point of the layer salinity.
        @test m.gamT[m.tmask .> 0] == (Γ .* m.ustar)[m.tmask .> 0]
        @test all(iszero, m.gamS)
        @test m.Tb[ice] == Tf[ice]
        # The heat balance, linear in ṁ.
        ci = @. m.c_i * ifelse(m.T.present > Tf, fmelt, ffreeze)
        Hi = @. -m.z_draft * m.rho0_seawater / m.rho_ice
        q = @. k * (Tf - m.T_ice_base) / (Hi * m.rho_freshwater)
        lhs = @. m.c_p * m.gamT * (m.T.present - m.Tb)
        rhs = @. m.melt * (m.L - ci * (m.T_ice_base - m.Tb)) + q
        @test lhs[ice] ≈ rhs[ice]
        # Gap and open-ocean cells do not melt; their Tb leaves the exchange term at zero.
        @test all(iszero, m.melt[.!ice])
        @test all(isfinite, m.melt)
        supercooled && conduction isa HollandJenkinsConduction && @test all(<(0), m.melt[ice])
    end

    # With γS → ∞ the salt balance forces Sb → S and the three-equation scheme reduces to
    # this one, ratio → 0 being γS = γT / ratio → ∞.
    m2, _ = mk(; melting = quiet(() -> TwoEquationMelting(Γ)))
    m3, _ = mk(; melting = quiet(() -> UStarGamTMelting(Γ, FT(1e-8))))
    ice = m2.imask .> 0
    @test m3.melt[ice] ≈ m2.melt[ice] rtol = 1e-5
    @test m3.Tb[ice] ≈ m2.Tb[ice] rtol = 1e-5

    # A short run of the scheme.
    sim = quiet() do
        build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                     params = Params(; FT, melting = TwoEquationMelting(Γ)))
    end
    run!(sim; days = 0.5, verbose = false)
    @test all(isfinite, sim.model.H.present) && all(isfinite, sim.model.T.present)
    @test any(>(0), sim.model.melt)
end

@testset "AbstractEquationOfState: LinearEOS default, RoquetEOS contrast, convection" begin
    quiet(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())
    mk(; kw...) = quiet() do
        build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm, params = Params(; FT, kw...))
    end

    @test Params(; FT).equation_of_state === LinearEOS()
    @test occursin("eq. of state   = LinearEOS()", sprint(show, MIME("text/plain"), Params(; FT)))
    @test_logs LinearEOS()
    @test_logs (:warn, r"RoquetEOS is an experimental equation of state") RoquetEOS()
    eos = quiet(() -> RoquetEOS())
    @test (eos.Cb, eos.Theta0, eos.Th, eos.b0) == (0.011, -4.5, 2.5e-5, 0.77)
    @test quiet(() -> RoquetEOS(0.011, -4.5, 2.5e-5, 1)).b0 === 1.0     # mixed types promote
    @test_throws ArgumentError RoquetEOS(-0.01, -4.5, 2.5e-5, 0.77)
    @test_throws ArgumentError RoquetEOS(0.011, -4.5, 2.5e-5, 0.0)
    @test sprint(show, eos) == "RoquetEOS(Cb = 0.011, Theta0 = -4.5, Th = 2.5e-5, b0 = 0.77)"
    p32 = quiet(() -> Params(; FT = Float32, equation_of_state = RoquetEOS()))
    @test p32.equation_of_state === RoquetEOS{Float32}(0.011f0, -4.5f0, 2.5f-5, 0.77f0)

    # The coefficients: the linear ones are the Params constants, with no T or Z dependence.
    m_lin = mk().model
    @test Laddie._eos_coefficients(m_lin) === (; beta = m_lin.beta, a0 = m_lin.alpha, aT = zero(FT), aZ = zero(FT))
    m_rq = mk(equation_of_state = eos).model
    c = Laddie._eos_coefficients(m_rq)
    ρ0 = m_rq.rho0_seawater
    @test c.beta ≈ 0.77 / ρ0 && c.aT ≈ 0.011 / ρ0 && c.aZ ≈ 2.5e-5 / ρ0 && c.a0 ≈ 0.011 * 4.5 / ρ0

    # α and β of Roquet et al. (2015), through the pointwise contrast (kg m⁻³ K⁻¹ and per g kg⁻¹).
    α(T, Z) = -Laddie._drho(35.0, 35.0, T + 1e-3, T - 1e-3, Z, c) * ρ0 / 2e-3
    @test α(-1.9, 0.0) ≈ 0.011 * 2.6 rtol = 1e-9                  # 0.0286: 0.028 at freezing, as in the paper
    @test α(-1.9, 500.0) ≈ 0.011 * 2.6 + 2.5e-5 * 500 rtol = 1e-9 # thermobaricity
    @test α(0.5, 500.0) ≈ 0.011 * 5.0 + 2.5e-5 * 500 rtol = 1e-9  # cabbeling: α rises with Θ
    @test α(-4.5, 0.0) ≈ 0 atol = 1e-12                           # Θ₀: no surface expansion
    @test Laddie._drho(35.1, 35.0, 0.0, 0.0, 300.0, c) * ρ0 ≈ 0.077 rtol = 1e-9
    # Antisymmetric and zero for equal states.
    @test Laddie._drho(34.7, 34.4, 0.5, -1.0, 400.0, c) ≈ -Laddie._drho(34.4, 34.7, -1.0, 0.5, 400.0, c)
    @test Laddie._drho(34.5, 34.5, -1.5, -1.5, 400.0, c) == 0
    # The reference contrasts of the planning notes (warm plume against a warm ambient, at 500 m).
    @test Laddie._drho(34.7, 34.4, 0.5, -1.0, 500.0, c) ≈ 1.38e-4 rtol = 5e-3
    @test Laddie._drho(34.7, 34.4, 0.5, -1.0, 500.0, Laddie._eos_coefficients(m_lin)) ≈ 1.79e-4 rtol = 5e-3

    # `update_density!` evaluates Roquet's polynomial at the layer mid-depth: independent check.
    ρp(T, S, Z) = (-0.011 / 2 * (T + 4.5)^2 - 2.5e-5 * Z * T + 0.77 * S) / ρ0
    sim = mk(equation_of_state = eos)
    run!(sim; days = 0.2, verbose = false)
    m = sim.model
    Laddie.update_density!(m)
    Z = @. -m.z_draft + m.H.present / 2
    ref = @. ρp(m.Ta, m.Sa, Z) - ρp(m.T.present, m.S.present, Z)
    act = m.tmask .> 0
    @test m.drho[act] ≈ ref[act] rtol = 1e-9
    @test all(iszero, m.drho[.!act])
    # An `alpha` override still moves a LinearEOS run (the constants stay the source of truth) ...
    m1 = mk().model; m2 = mk(alpha = 2 * m1.alpha).model
    Laddie.update_density!(m1); Laddie.update_density!(m2)
    @test m1.drho != m2.drho
    # ... and does not touch a RoquetEOS one.
    m3 = mk(equation_of_state = eos, alpha = 2 * m1.alpha).model
    m4 = mk(equation_of_state = eos).model
    Laddie.update_density!(m3); Laddie.update_density!(m4)
    @test m3.drho == m4.drho

    # ResetToAmbient resets to a contrast of exactly the threshold, whatever the EOS.
    for e in (LinearEOS(), eos)
        s = mk(equation_of_state = e, convection_scheme = ResetToAmbient(FT(0.005)))
        run!(s; days = 0.2, verbose = false)
        mm = s.model
        ice = mm.imask .> 0
        mm.S.present .= ifelse.(ice, mm.Sa .+ 0.5, mm.S.present)     # saltier than ambient: unstable
        Laddie.update_density!(mm)
        @test all(<(0), mm.drho[ice])
        Laddie.update_convection!(mm, mm.convection_scheme)
        @test all(mm.convection[ice] .== 1)
        @test mm.drho[ice] ≈ fill(FT(0.005) / mm.rho0_seawater, count(ice)) rtol = 1e-9
        Laddie.update_density!(mm)
        @test mm.drho[ice] ≈ fill(FT(0.005) / mm.rho0_seawater, count(ice)) rtol = 1e-9
    end

    # Every convection scheme and entrainment runs under RoquetEOS, and the melt differs from linear.
    melts = map((ResetToAmbient(FT(0.005)), ClampDensity(FT(0.005)), RelaxToAmbient(FT(10000.0)))) do cs
        s = mk(equation_of_state = eos, convection_scheme = cs)
        run!(s; days = 0.5, verbose = false)
        @test all(isfinite, s.model.H.present) && all(isfinite, s.model.melt) && all(isfinite, s.model.drho)
        return copy(s.model.melt)
    end
    s_h = mk(equation_of_state = eos, entrainment = HollandEntrainment(FT(0.01775)))
    run!(s_h; days = 0.5, verbose = false)
    @test all(isfinite, s_h.model.H.present)
    s_l = mk(convection_scheme = ResetToAmbient(FT(0.005)))
    run!(s_l; days = 0.5, verbose = false)
    @test s_l.model.melt != melts[1]
end

@testset "SmoothKinks: C¹ blends, exact outside the band" begin
    @test Params().kinks === ExactKinks()
    @test Params(; FT = Float32, kinks = SmoothKinks()).kinks.H === 0.1f0
    k, ε = SmoothKinks(), 0.1
    smax0(x) = Laddie._smax0(k, x, ε)
    d(f, x) = ForwardDiff.derivative(f, x)
    # max(x, 0): exact outside [-ε, ε], value and slope continuous at the band edges.
    @test smax0(-0.5) === 0.0 && smax0(0.5) === 0.5 && smax0(0.0) ≈ ε / 4
    for x in (-ε, ε)
        @test smax0(x - 1e-9) ≈ smax0(x + 1e-9) atol = 1e-8
        @test d(smax0, x - 1e-9) ≈ d(smax0, x + 1e-9) atol = 1e-7
    end
    # max/min/clamp follow it, and stay exact next to a much larger operand.
    @test Laddie._smax(k, 2.0, 0.5, ε) === 2.0 && Laddie._smin(k, 2.0, 0.5, ε) === 0.5
    @test Laddie._smin(k, 0.5, 0.5, ε) ≈ 0.5 - ε / 4
    @test Laddie._smin(SmoothKinks(; w = 1f-8), 1f6, 3f-5, 1f-8) === 3f-5
    @test Laddie._sclamp(k, 7.0, -1.0, 1.0, ε) === 1.0
    # An infinite bound (a bed of -Inf) neither leaks into the value nor the derivative.
    @test Laddie._smin(k, 3.0, Inf, ε) === 3.0
    @test d(h -> Laddie._smin(k, h, Inf, ε), 3.0) === 1.0
    # The upwind split is conservative, and a zero flux (a wall) carries nothing.
    for u in (-0.3, -0.05, 0.0, 0.02, 0.4)
        @test smax0(u) + Laddie._sneg(k, u, ε) ≈ u atol = 1e-15
    end
    @test Laddie._supwind(k, 0.0, 1.0, 5.0, ε) === 0.0
    @test Laddie._supwind(k, 0.5, 1.0, 5.0, ε) === 0.5 && Laddie._supwind(k, -0.5, 1.0, 5.0, ε) === -2.5
    @test Laddie._sabs(k, -0.5, ε) === 0.5 && Laddie._sabs(k, 0.0, ε) ≈ ε / 2
    # √max(x, 0): exact above ε², zero below -3ε², slope bounded by 1/2ε.
    ssqrt(x) = Laddie._ssqrt(k, x, ε)
    @test ssqrt(0.04) === 0.2 && ssqrt(-0.04) === 0.0
    @test d(ssqrt, ε^2 - 1e-12) ≈ d(ssqrt, ε^2 + 1e-12) rtol = 1e-6
    @test maximum(x -> d(ssqrt, x), range(-3ε^2, ε^2; length = 101)) <= 1 / 2ε + 1e-9
    # The c_i switch at the freezing point, blended over [-ε, ε].
    sw(x) = Laddie._sswitch(k, x, 1.0, 0.0, ε)
    @test sw(-0.2) === 0.0 && sw(0.2) === 1.0 && sw(0.0) ≈ 0.5
    @test abs(d(sw, ε)) < 1e-12
    # An ε of zero is the exact operation, bit for bit.
    k0 = SmoothKinks(; H = 0.0)
    @test Laddie._smax(k0, 0.3, 0.7, 0.0) === 0.7 && Laddie._ssqrt(k0, 0.3, 0.0) === sqrt(0.3)
end

@testset "SmoothKinks: exact at ε = 0, close to exact at the defaults" begin
    p = (; entrainment = GasparEntrainment(2.5), melting = TurbulentGamTMelting(),
         convection_scheme = ClampDensity(0.005), frazil_ice = RelaxedFrazilIce(),
         max_layer_thickness = AbsoluteMaxLayerThickness(30.0), lateral_viscosity = NonlinearLateralViscosity(0.6),
         momentum_advection = UpstreamMomentumAdvection(), H_min = 2.0, v_cut = 0.3)
    zero_eps = SmoothKinks(; H = 0.0, u = 0.0, uH = 0.0, w = 0.0, drho = 0.0, T = 0.0, S = 0.0, log = 0.0)
    runs = map((ExactKinks(), zero_eps, SmoothKinks())) do kinks
        s = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm, params = Params(; FT, p..., kinks))
        run!(s; days = 1.0, verbose = false)
        (copy(s.model.melt), copy(s.model.H.present), copy(s.model.U.present))
    end
    @test runs[2] == runs[1]
    melt_exact, melt_smooth = runs[1][1], runs[3][1]
    @test all(isfinite, melt_smooth) && melt_smooth != melt_exact
    @test sum(melt_smooth) ≈ sum(melt_exact) rtol = 1e-2
end
