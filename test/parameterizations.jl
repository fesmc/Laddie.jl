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
    @test all(isfinite, m.model.D.present)
    @test all(isfinite, m.model.melt)
    @test all(m.model.melt[m.model.tmask .> 0] .>= 0)
end

@testset "TurbulentGamTMelting: transfer coefficients stay positive" begin
    # A Prandtl or Schmidt number that leaves no positive offset is rejected.
    @test_throws ArgumentError TurbulentGamTMelting(0.5, 2432.0, 1.95e-6)
    @test_throws ArgumentError TurbulentGamTMelting(13.8, 0.5, 1.95e-6)
    @test_throws ArgumentError TurbulentGamTMelting(13.8, 2432.0, 0.0)
    @test_throws ArgumentError Params(; melting = TurbulentGamTMelting(; Pr = 0.5))

    # A thin, slow layer (u★D/ν₀ ≪ 1).  With Pr = 1 the unfloored log term would
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
    @test all(isfinite, mm.D.present) && all(isfinite, mm.melt)
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
    @test all(isfinite, m.model.D.present)
    @test all(isfinite, m.model.melt)
    @test all(m.model.melt[m.model.tmask .> 0] .>= 0)
end

@testset "Entrainment: Lambert is default, Gaspar (literal Eq. 14) differs" begin
    # LambertEntrainment reproduces the reference LADDIE production term
    # (2μ u★³/(g D δρ)) and must be the default so the Python verification
    # holds; GasparEntrainment is the literal Eq. 14 (μ u★³/(g D² δρ)).
    @test build_isomip(CPU(); nx=20, ny=10, isomipcond=:cold).model.entrainment isa
          LambertEntrainment
    mk(ep) = build_isomip(CPU(); FT, nx=20, ny=10, isomipcond=:warm,
        gradient = PyGradient(),
        params = Params(; FT, entrainment = ep, melting = FixedGamTMelting(FT(0.00018)),
                        convection_scheme = ResetToAmbient(FT(0.005))))
    ml = mk(LambertEntrainment(FT(2.5)))
    mg = mk(GasparEntrainment(FT(2.5)))
    run!(ml; days=0.5, verbose=false)
    run!(mg; days=0.5, verbose=false)
    @test all(isfinite, ml.model.entr) && all(isfinite, mg.model.entr)
    @test all(isfinite, ml.model.melt) && all(isfinite, mg.model.melt)
    # Same μ but different production term ⇒ the entrainment fields diverge.
    @test !isapprox(ml.model.entr, mg.model.entr)
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
    @test all(isfinite, m_nl.model.D.present) && all(isfinite, m_nl.model.melt)
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
    @test all(isfinite, m_nl_ns.model.D.present) && all(isfinite, m_nl_ns.model.melt)
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
    @test all(isfinite, m.T.present) && all(isfinite, m.D.present)
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
    @test AbsoluteMaxLayerThickness().D_max === 100.0f0
    @test Params(; max_layer_thickness = AbsoluteMaxLayerThickness()).max_layer_thickness.D_max === 100.0
    mk = zeros(Int, 12, 8)
    mk[1, :] .= 1; mk[end, :] .= 1; mk[:, 1] .= 1; mk[:, end] .= 1
    mk[2:3, 2:7] .= 2; mk[4:10, 2:7] .= 3; mk[11, 2:7] .= 0
    z_draft = fill(FT(-300.0), size(mk))
    z_bed = fill(FT(-310.0), size(mk))          # a 10 m water column everywhere
    grid = Grid(mk, z_draft, 1000.0, 1000.0; z_bed, domain_cropping = NoDomainCropping())
    m = Model(grid; forcing = ISOMIPForcing(:warm; FT),
              params = Params(; FT, max_layer_thickness = AbsoluteMaxLayerThickness(50.0)))
    m.D.future .= FT(80.0) .* m.tmask
    Laddie._clamp_thickness!(m)
    act = m.tmask .> 0
    @test all(m.D.future[act] .== 50)           # capped at D_max, not at 10 m
    @test all(m.D.future[.!act] .== 0)
end

@testset "ClampDensity convection scheme: build and short run" begin
    params = Params(; FT, convection_scheme = ClampDensity(FT(0.005)))
    m = build_isomip(CPU(); FT, nx=20, ny=10, isomipcond=:warm, params)
    @test all(isfinite, m.model.melt)
    run!(m; days=0.5, verbose=false)
    @test all(isfinite, m.model.D.present)
    @test all(isfinite, m.model.melt)
end

@testset "RelaxToAmbient convection scheme: build and short run" begin
    params = Params(; FT, convection_scheme = RelaxToAmbient(FT(10000.0)))
    m = build_isomip(CPU(); FT, nx=20, ny=10, isomipcond=:warm, params)
    @test all(isfinite, m.model.melt)
    run!(m; days=0.5, verbose=false)
    @test all(isfinite, m.model.D.present)
    @test all(isfinite, m.model.melt)
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
        m.D.present .= 10.0
        au = Array(copy(Laddie.upwind_advection_U(m)))
        av = Array(copy(Laddie.upwind_advection_V(m)))
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
