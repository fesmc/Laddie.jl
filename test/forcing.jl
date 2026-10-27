# Ocean and ice forcing: profile resampling, ambient lookup, ice temperature (T_ice_base).

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
    @test all(isfinite, m2.model.H.present)
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
    # Model samples the profile before H is initialised; resample at the layer base.
    Laddie.update_ambient_fields!(m1)
    Laddie.update_ambient_fields!(m2)
    act = m1.tmask .> 0
    zb = (m1.z_draft .- m1.H.present)[act]
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
        params = Params(; FT, entrainment = GasparEntrainment(FT(2.5)),
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
    @test sum(m_warm.model.melt) / sum(m_def.model.melt) ≈ 1.049 rtol = 0.02   # 1.072 under AdvectiveTracerForm

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

@testset "PerBasinOceanForcing1D: each basin samples its own profile" begin
    grid = build_isomip(CPU(); FT, nx = 20, ny = 10).model.grid
    warm, cold = ISOMIPForcing(:warm; FT), ISOMIPForcing(:cold; FT)
    # Basin 1 on the first half of the columns, basin 2 on the rest (full domain,
    # border ring included).
    basin = fill(1, grid.input_size)
    basin[:, (grid.input_size[2]÷2+1):end] .= 2
    f = PerBasinOceanForcing1D([warm, cold], basin)
    @test size(f.Tz) == (2, length(warm.z))
    @test f.Tz[1, :] ≈ warm.Tz && f.Sz[2, :] ≈ cold.Sz
    m, mw, mc = (Model(grid; forcing = o) for o in (f, warm, cold))
    foreach(Laddie.update_ambient_fields!, (m, mw, mc))
    b = m.basin   # cropped with the grid
    @test size(b) == size(m.Ta)
    @test m.Ta[b .== 1] == mw.Ta[b .== 1] && m.Sa[b .== 1] == mw.Sa[b .== 1]
    @test m.Ta[b .== 2] == mc.Ta[b .== 2] && m.Sa[b .== 2] == mc.Sa[b .== 2]
    @test any(b .== 1) && any(b .== 2)
    run!(Simulation(m); days = 0.2, verbose = false)
    @test all(isfinite, m.H.present) && all(isfinite, m.melt)
end

@testset "PerBasinOceanForcing1D: profiles of different depth ranges, checks" begin
    # A profile on a shorter grid is extended flat onto the common one.
    profile(z, T, S) = Laddie.OceanForcing1D(FT.(T.(z)), FT.(S.(z)), FT.(z), FT(1), FT(first(z)))
    deep = profile(-1000.0:1.0:-1.0, z -> 3 + z / 500, z -> 35 + z / 1000)
    shallow = profile(-200.0:1.0:-1.0, z -> -1.5 - z / 400, z -> 33.5 - z / 400)
    f = PerBasinOceanForcing1D([deep, shallow], fill(2, 4, 3))
    @test first(f.z) == -1000 && last(f.z) == -1
    @test f.Tz[1, :] ≈ deep.Tz
    @test all(f.Tz[2, f.z .<= -200] .≈ shallow.Tz[1])   # flat below its own range
    @test f.Tz[2, end] ≈ shallow.Tz[end]
    @test eltype(f.basin) == Int32
    @test_throws ArgumentError PerBasinOceanForcing1D([deep, shallow], fill(3, 4, 3))
    @test_throws ArgumentError PerBasinOceanForcing1D([deep, shallow], fill(0, 4, 3))
    grid = build_isomip(CPU(); FT, nx = 20, ny = 10).model.grid
    @test_throws ArgumentError Model(grid; forcing = PerBasinOceanForcing1D([deep], fill(1, 3, 3)))
    @test occursin("2 basin profiles", sprint(show, f))
end

@testset "nearest_seed: exact Euclidean distance and label" begin
    labels = zeros(Int, 23, 17)
    seeds = [(3, 4), (20, 2), (11, 15), (18, 13), (1, 17)]
    for (k, (i, j)) in enumerate(seeds)
        labels[i, j] = k
    end
    lab, d2 = nearest_seed(labels)
    for I in CartesianIndices(labels)
        d = [(I[1] - i)^2 + (I[2] - j)^2 for (i, j) in seeds]
        @test d2[I] == minimum(d)
        @test d[lab[I]] == minimum(d)   # a nearest seed (ties may go either way)
    end
    @test all(lab[i, j] == k for (k, (i, j)) in enumerate(seeds))
    lab0, d0 = nearest_seed(zeros(Int, 4, 3))
    @test all(==(0), lab0) && all(isinf, d0)
end

@testset "front_sources: through the cavity, deepest neighbour, fallback" begin
    # Two shelves (3) separated by grounded ice (2) along j = 5, open ocean (0) at
    # i ≥ 10.  The upper shelf meets the ocean only at its far corner (9, 10), so
    # its cells must take that front even where the lower front is nearer in a
    # straight line.  An enclosed pool (0) in the lower shelf is cavity, and an
    # isolated shelf cell with no front takes the nearest front in a straight line.
    mask = fill(1, 12, 12)
    mask[2:11, 2:11] .= 2
    mask[4:9, 2:4] .= 3
    mask[4:9, 6:10] .= 3
    mask[9, 6:9] .= 2
    mask[10:11, 2:11] .= 0
    mask[6, 3] = 0       # enclosed water
    mask[2, 11] = 3      # shelf cell without a front
    # Front cell (9, 3) has one open neighbour, (10, 3); the enclosed pool's
    # neighbours are all shelf.  `depth` ranks ties between open neighbours.
    depth = zeros(12, 12)
    src, lost = front_sources(mask; depth)
    C = CartesianIndices(mask)
    srcij(i, j) = Tuple(C[src[i, j]])
    @test lost == 1
    @test all(srcij(i, j)[2] in 2:4 for i = 4:9, j = 2:4)                 # lower shelf, own front
    @test all(srcij(i, j) == (10, 10) for i = 4:8, j = 6:10)              # upper shelf, through the corner
    @test srcij(8, 6) == (10, 10)   # straight line would pick the lower front (√5 vs √17)
    @test src[6, 3] > 0             # enclosed water is cavity
    @test srcij(9, 3) == (10, 3)
    @test all(src[mask .== 2] .== 0) && all(src[10:11, :] .== 0)
    @test srcij(2, 11)[1] == 10     # the fallback still picks an open-ocean column
    # Only wet open cells can be sources: dry out (10, 3) and (9, 3) takes a neighbour's.
    wet = trues(size(mask)); wet[10, 3] = false
    src2, _ = front_sources(mask; wet)
    @test Tuple(C[src2[9, 3]]) != (10, 3)
    @test_throws ArgumentError front_sources(fill(3, 5, 5))
    # A front cell with several open neighbours takes the one of lowest `depth`.
    island = fill(1, 6, 6)
    island[2:5, 2:5] .= 0
    island[3, 3] = 3
    d = zeros(6, 6); d[4, 3] = -800; d[3, 4] = -500
    s3, _ = front_sources(island; depth = d)
    @test Tuple(CartesianIndices(island)[s3[3, 3]]) == (4, 3)
end

@testset "front_ocean_forcing: cavity cells take their front's column" begin
    # An ISOMIP+ channel whose shelf ends at x = 30 km, with open ocean beyond.
    # Each open-ocean column is the warm profile shifted by 0.1 °C per column, so
    # a cavity cell's profile tells which column fed it: the front of its row.
    sim = build_isomip(CPU(); FT, nx = 20, ny = 10, xfront = 30_000.0,
                       domain_cropping = NoDomainCropping())
    grid = sim.model.grid
    mask = grid.mask
    warm = ISOMIPForcing(:warm; FT)
    z = collect(-1000.0:50.0:-50.0)
    prof(zz) = -1.9 + zz * (1.0 + 1.9) / -720.0
    nx, ny = size(mask)
    T = [mask[i, j] == 0 ? prof(min(z[k], -1.0)) + 0.1i : NaN for i = 1:nx, j = 1:ny, k = eachindex(z)]
    S = [mask[i, j] == 0 ? 34.0 : NaN for i = 1:nx, j = 1:ny, k = eachindex(z)]
    ifront = findlast(i -> mask[i, 5] == 3, 1:nx)
    T[ifront+1, 2, 1:5] .= NaN   # a shorter column extends flat below its water
    f = front_ocean_forcing(T, S, z, mask; FT)
    @test f isa PerBasinOceanForcing1D
    @test size(f.Tz, 1) == count(mask[ifront+1, :] .== 0)   # one profile per front cell's source
    for j = 3:(ny-1), i = 2:ifront
        mask[i, j] == 3 || continue
        @test f.Tz[f.basin[i, j], :] ≈ T[ifront+1, j, :]
    end
    @test all(f.Tz[f.basin[ifront, 2], 1:5] .≈ T[ifront+1, 2, 6])
    @test all(1 .<= f.basin .<= size(f.Tz, 1))
    m = Model(grid; forcing = f)
    run!(Simulation(m); days = 0.2, verbose = false)
    @test all(isfinite, m.H.present) && all(isfinite, m.melt)
    @test_throws ArgumentError front_ocean_forcing(T[1:5, :, :], S, z, mask)
end

@testset "regrid_ocean and fill_ocean!" begin
    x, y = collect(0.0:8.0:64.0), collect(0.0:8.0:40.0)
    lin(xx, yy, k) = 0.1xx - 0.05yy + k
    T = [lin(xx, yy, k) for xx in x, yy in y, k = 1:3]
    S = T .+ 30
    px, py = collect(1.0:2.0:63.0), collect(1.0:2.0:39.0)
    o = regrid_ocean(T, S, x, y, px, py)
    @test size(o.T) == (length(px), length(py), 3)
    @test o.T ≈ [lin(xx, yy, k) for xx in px, yy in py, k = 1:3]   # bilinear is exact on linear fields
    @test o.S ≈ o.T .+ 30
    # Missing source cells are left out, the weights renormalised: a constant
    # field stays constant next to a hole, and a fully dry region is NaN.
    C = fill(2.0, length(x), length(y), 1)
    C[1:3, :, 1] .= NaN
    C[5, 3, 1] = NaN
    oc = regrid_ocean(C, C, x, y, px, py)
    @test all(isnan, oc.T[px .< 16, :, 1])
    @test all(oc.T[px .> 16, :, 1] .≈ 2)
    # fill_ocean!: flat below a column's water, and dry columns take the nearest one.
    A = [k == 3 ? NaN : Float64(i + 10k) for i = 1:4, j = 1:3, k = 1:3]
    A[1, :, :] .= NaN
    B = copy(A)
    fill_ocean!(A, B)
    @test all(isfinite, A)
    @test A[2, 1, 3] == A[2, 1, 2]
    @test A[1, 2, :] == A[2, 2, :]
    Laddie.OceanForcing2D(A, B, [-300.0, -200.0, -100.0])   # complete, so it constructs
end
