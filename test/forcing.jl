# Ocean and ice forcing: profile resampling, ambient lookup, basal ice temperature.

@testset "OceanForcing1D: resampling, sorting, flat extrapolation" begin
    z_c = [-1000.0, -500.0, -100.0]
    T_c = [   1.0,     0.0,   -1.0]
    S_c = [  34.7,    34.2,   33.8]
    f = OceanForcing1D(T_c, S_c, z_c; FT)
    @test f.dz == 1.0
    @test f.z0 == -5000.0
    @test f.z == FT.(-5000.0:1.0:-1.0)
    # Flat extrapolation below the deepest sample (z = -5000 < -1000)
    @test f.Tz[1] ≈ 1.0
    @test f.Sz[1] ≈ 34.7
    # Flat extrapolation above the shallowest sample (z = -1 > -100)
    @test f.Tz[end] ≈ -1.0
    @test f.Sz[end] ≈ 33.8
    # Linear interpolation at z = -750 (midway between -1000 and -500)
    k = findfirst(==(FT(-750.0)), f.z)
    @test f.Tz[k] ≈ 0.5
    @test f.Sz[k] ≈ 34.45

    # Descending input (CSV convention: surface first) gives the same result
    f_rev = OceanForcing1D(reverse(T_c), reverse(S_c), reverse(z_c); FT)
    @test f_rev.Tz == f.Tz
    @test f_rev.Sz == f.Sz

    # Duplicate depths are tolerated (first occurrence kept)
    f_dup = OceanForcing1D([1.0, 2.0, -1.0], [34.7, 34.6, 33.8],
                           [-1000.0, -1000.0, -100.0]; FT)
    @test all(isfinite, f_dup.Tz)

    # Length mismatch must throw
    @test_throws ArgumentError OceanForcing1D(T_c[1:2], S_c, z_c; FT)
end

@testset "OceanForcing1D: reproduces ISOMIPForcing from coarse samples" begin
    # The warm ISOMIP profile is linear in z, so 3 samples recover it exactly.
    isomip = ISOMIPForcing(:warm; FT)
    T_lin(z) = -1.9 + z * (1.0 - (-1.9)) / (-720.0)
    S_lin(z) = 33.8 + z * (34.7 - 33.8) / (-720.0)
    z_c  = [-5000.0, -720.0, -1.0]
    prof = OceanForcing1D(T_lin.(z_c), S_lin.(z_c), z_c; FT)
    @test prof.Tz ≈ isomip.Tz
    @test prof.Sz ≈ isomip.Sz

    # Same domain, both forcings: initial melt fields must agree
    nx_i, ny_i = 6, 4
    mask = zeros(Int, nx_i + 2, ny_i + 2)
    mask[1, :]   .= 1;   mask[end, :] .= 1
    mask[:, 1]   .= 1;   mask[:, end] .= 1
    mask[2:3, 2:end-1]   .= 2
    mask[4:end-1, 2:end-1] .= 3
    z_draft_raw = fill(-400.0, nx_i + 2, ny_i + 2)

    m1 = Simulation(Model(Grid(mask, z_draft_raw, 2000.0, 2000.0; FT);
                          forcing = isomip, params = Params(; FT)))
    m2 = Simulation(Model(Grid(mask, z_draft_raw, 2000.0, 2000.0; FT);
                          forcing = prof, params = Params(; FT)))
    @test m2.model.melt ≈ m1.model.melt
    run!(m2; days = 0.5, verbose = false)
    @test all(isfinite, m2.model.D.present)
    @test all(isfinite, m2.model.melt)
end

@testset "Ambient profile lookup honours dz" begin
    # A linear profile is interpolated exactly, so a 2 m grid must give the
    # same ambient fields as the canonical 1 m grid.
    grid = build_isomip(CPU(); FT, nx = 20, ny = 10).model.grid
    Tlin(z) = 1 + z / 1000
    Slin(z) = 34 - z / 2000
    mk(z) = Model(grid; forcing = Laddie.OceanForcing1D(FT.(Tlin.(z)), FT.(Slin.(z)),
                                                        FT.(z), FT(step(z)), FT(first(z))))
    m1 = mk(-5000.0:1.0:-1.0)
    m2 = mk(-5000.0:2.0:-2.0)
    # Model samples the profile before D is initialised; resample at the layer base.
    Laddie.update_ambient_fields!(m1)
    Laddie.update_ambient_fields!(m2)
    act = m1.tmask .> 0
    zb = (m1.z_draft .- m1.D.present)[act]
    @test m1.Ta[act] ≈ Tlin.(zb)
    @test m2.Ta[act] ≈ m1.Ta[act]
    @test m2.Sa[act] ≈ m1.Sa[act]
end

@testset "Ice forcing: scalar and 2D T_ice_base, default is inert" begin
    # T_ice_base moved out of Params and into the forcing.  The default must
    # reproduce the old scalar Params.T_i = -25.0 exactly, or every existing
    # result shifts.
    m_def = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm)
    @test m_def.model.T_ice_base isa Matrix{FT}
    @test size(m_def.model.T_ice_base) == size(m_def.model.tmask)
    @test all(m_def.model.T_ice_base .== -25)
    @test !hasfield(typeof(getfield(m_def.model, :params)), :T_i)

    mask0 = copy(m_def.model.mask); zd0 = copy(m_def.model.z_draft)
    shelf_cols = [i for i in axes(mask0, 1) if any(==(3), @view mask0[i, :])]
    grid0 = Grid(mask0, zd0, 2000.0, 2000.0; FT, domain_cropping = NoDomainCropping())
    build(ice) = Simulation(Model(grid0;
        forcing = CavityForcing(ISOMIPForcing(:warm; FT), ice),
        params = Params(; FT, entrainment = LambertEntrainment(FT(2.5)),
                        melting = FixedGamTMelting(FT(0.00018))),
        boundary = BoundaryConditions(; open_ocean = ZeroGradientInflow())))

    # An explicit CavityForcing with the same uniform value is bit-identical to
    # the implicit default, so the move is provably inert.
    m_exp = build(PrescribedIceForcing(FT(-25.0)))
    run!(m_def; days = 1.0, verbose = false)
    run!(m_exp; days = 1.0, verbose = false)
    @test m_exp.model.melt == m_def.model.melt
    @test m_exp.model.T.present == m_def.model.T.present

    # Warmer ice melts more: L_eff = L - c_i*T_i shrinks from 3.84e5 J/kg at
    # -25 degC to 3.34e5 at 0 degC.  The response is damped well below that 15%
    # because the extra melt cools and freshens the layer that drives it.
    m_warm = build(PrescribedIceForcing(FT(0.0)))
    run!(m_warm; days = 1.0, verbose = false)
    @test sum(m_warm.model.melt) / sum(m_def.model.melt) ≈ 1.072 rtol = 0.02

    # A 2D field is the point of the move: temperate ice over the upstream half
    # of the shelf, cold ice over the rest, must land strictly between the two
    # uniform runs and match each of them on its own half.
    half = shelf_cols[1:(length(shelf_cols) ÷ 2)]
    Ti = fill(FT(-25.0), size(mask0)); Ti[half, :] .= 0
    m_2d = build(PrescribedIceForcing(Ti))
    run!(m_2d; days = 1.0, verbose = false)
    @test sum(m_def.model.melt) < sum(m_2d.model.melt) < sum(m_warm.model.melt)
    @test sum(m_2d.model.melt[half, :]) > sum(m_def.model.melt[half, :])
    @test m_2d.model.melt[m_2d.model.imask .> 0] != m_warm.model.melt[m_warm.model.imask .> 0]

    # The turbulent-gamT variant is the second melt kernel; it takes the same
    # per-cell L_eff path and must stay physical on the same 2D field.
    m_turb = Simulation(Model(grid0;
        forcing = CavityForcing(ISOMIPForcing(:warm; FT), PrescribedIceForcing(Ti)),
        params = Params(; FT, melting = TurbulentGamTMelting())))
    run!(m_turb; days = 1.0, verbose = false)
    @test all(isfinite, m_turb.model.melt) && all(m_turb.model.melt .>= 0)

    # Validation: wrong shape, ice above the melting point, NaN.
    @test_throws ArgumentError build(PrescribedIceForcing(zeros(FT, 3, 3)))
    @test_throws ArgumentError build(PrescribedIceForcing(FT(5.0)))
    bad = fill(FT(-25.0), size(mask0)); bad[5, 5] = NaN
    @test_throws ArgumentError build(PrescribedIceForcing(bad))

    # A 2D field is cropped with the mask rather than silently mismatched.
    marked = fill(FT(-25.0), size(mask0)); marked[shelf_cols[3], 6] = FT(-2.0)
    m_crop = Model(Grid(mask0, zd0, 2000.0, 2000.0; FT,
                        domain_cropping = MinRectangleDomainCropping(margin = 2));
                   forcing = CavityForcing(ISOMIPForcing(:warm; FT), PrescribedIceForcing(marked)))
    @test size(m_crop.T_ice_base) == size(m_crop.tmask)
    @test count(==(FT(-2.0)), m_crop.T_ice_base) == 1

    # A bare ocean forcing still works and picks up the default ice.
    m_bare = Model(grid0; forcing = ISOMIPForcing(:warm; FT))
    @test getfield(m_bare, :forcing) isa CavityForcing
    @test all(m_bare.T_ice_base .== Laddie.DEFAULT_T_ICE_BASE)
end
