# Geometry ingestion: BedMachine-style mask/draft builders and mask cleaning.

@testset "Geometry ingestion: build_laddie_mask classification" begin
    # 2×3 interior domain:
    #   (1,1) h=600, bed=-500 → h_af = 600*917/1028 - 500 ≈ +35 → grounded (2)
    #   (1,2) h=400, bed=-500 → h_af = 400*917/1028 - 500 ≈ -143 → floating (3)
    #   (1,3) h=0,   bed=-200 → no ice → ocean (0)
    #   row 2: same pattern
    bed       = [-500.0  -500.0  -200.0;
                 -500.0  -500.0  -200.0]
    thickness = [ 600.0   400.0     0.0;
                  600.0   400.0     0.0]

    mask = build_laddie_mask(bed, thickness)
    @test size(mask) == (4, 5)           # (ny+2, nx+2) = (4, 5)
    # Border ring is all 1
    @test all(mask[1, :]   .== 1)
    @test all(mask[end, :] .== 1)
    @test all(mask[:, 1]   .== 1)
    @test all(mask[:, end] .== 1)
    # Interior cells
    @test mask[2, 2] == 2   # grounded
    @test mask[2, 3] == 3   # floating
    @test mask[2, 4] == 0   # ocean
    @test mask[3, 2] == 2
    @test mask[3, 3] == 3
    @test mask[3, 4] == 0

    # Ice-free cells are split by bed elevation: bedrock at or above sea level
    # is land (1), not ocean (0).  Exposed rock classified as ocean would act
    # as an open-boundary sink wherever it sits inside the domain.
    bed2       = [-500.0  -500.0  -200.0   40.0    0.0;
                  -500.0  -500.0  -200.0  900.0   -0.1]
    thickness2 = [ 600.0   400.0     0.0    0.0    0.0;
                   600.0   400.0     0.0    0.0    0.0]
    m2 = build_laddie_mask(bed2, thickness2)
    @test m2[2, 4] == 0   # ice-free, bed -200 m  → ocean
    @test m2[2, 5] == 1   # ice-free, bed  +40 m  → land
    @test m2[3, 5] == 1   # ice-free, bed +900 m  → land
    @test m2[2, 6] == 1   # ice-free, bed    0 m  → land (sea level counts as land)
    @test m2[3, 6] == 0   # ice-free, bed -0.1 m  → ocean

    # Mismatched sizes must throw
    @test_throws ArgumentError build_laddie_mask(bed, thickness[1:1, :])
end

@testset "Land vs grounded ice: rock islands are walls, not ice fronts" begin
    # Shelf filling the domain, open ocean on the last interior column, and a
    # 2x2 rock island (mask 1) punched into the middle of the shelf.
    mk = zeros(Int, 12, 22)
    mk[1, :] .= 1;  mk[end, :] .= 1;  mk[:, 1] .= 1;  mk[:, end] .= 1
    mk[2:11, 2:3]  .= 2      # grounded ice
    mk[2:11, 4:20] .= 3      # shelf
    mk[2:11, 21]   .= 0      # open ocean → real ice front at column 20
    mk[5:6, 10:11] .= 1      # rock island
    isl = CartesianIndices((5:6, 10:11))
    grid = Grid(mk, fill(-400.0, 12, 22), 2000.0, 2000.0; FT, domain_cropping = NoDomainCropping())
    m = Model(grid; forcing = ISOMIPForcing(:warm; FT))

    @test all(m.lnd[isl] .== 1)      # island is land
    @test all(m.ocn[isl] .== 0)      # and emphatically not ocean
    @test all(m.grd[isl] .== 1)      # it is a wall
    @test all(m.tmask[isl] .== 0)    # not simulated
    @test sum(m.lnd) == count(mk .== 1)   # border ring + island, nothing else

    # The island generates no ice front: at_isf fires on ocean neighbours only,
    # and the island has none.  The genuine front at column 20 still does.
    isf_here = (m.tmask .> 0) .&
               (m.ocnxm1 .+ m.ocnxp1 .+ m.ocnym1 .+ m.ocnyp1 .> 0)
    for I in isl, (di, dj) in ((-1,0),(1,0),(0,-1),(0,1))
        i, j = Tuple(I) .+ (di, dj)
        (5 <= i <= 6 && 10 <= j <= 11) && continue
        @test isf_here[i, j] == false
    end
    @test any(isf_here[2:11, 20])    # the real ice front is still detected

    # The island is a land wall, not a grounding line: grd?? indicators
    # fire around it while the gl?? ones (mask == 2 only) stay off — and the
    # land-only lnd?? ones (AbstractLandBC) fire there instead, mirroring how
    # gl?? does for grounded ice.
    g = getfield(m, :geometry)
    @test wall_faces(g).grdEv[5, 9] > 0 || wall_faces(g).grdWv[5, 12] > 0 || wall_faces(g).grdNu[7, 10] > 0 || wall_faces(g).grdSu[4, 10] > 0
    @test all(g.glNu[isl] .== 0) && all(g.glSu[isl] .== 0)
    @test all(g.glEv[isl] .== 0) && all(g.glWv[isl] .== 0)
    @test all(g.glNu .<= wall_faces(g).grdNu) && all(g.glEv .<= wall_faces(g).grdEv)
    @test g.lndEv[5, 9] > 0 || g.lndWv[5, 12] > 0 || g.lndNu[7, 10] > 0 || g.lndSu[4, 10] > 0
    @test all(g.lndNu .<= wall_faces(g).grdNu) && all(g.lndEv .<= wall_faces(g).grdEv)

    sim = Simulation(m)
    @test sim.model === m
    run!(sim; days = 0.5, verbose = false)
    @test all(isfinite, m.D.present) && all(isfinite, m.melt)
end

@testset "Geometry ingestion: ice_base_depth values" begin
    bed       = [-500.0  -500.0  -200.0]
    thickness = [ 600.0   400.0     0.0]

    z_draft = ice_base_depth(bed, thickness)
    @test size(z_draft) == (3, 5)    # (1+2, 3+2)
    # Border zeros
    @test all(z_draft[1, :] .== 0.0)
    @test all(z_draft[end, :] .== 0.0)
    @test all(z_draft[:, 1]   .== 0.0)
    @test all(z_draft[:, end] .== 0.0)
    # Grounded: z_draft = bed
    @test z_draft[2, 2] ≈ -500.0
    # Floating: z_draft = -h * rho_ice/rho_sw
    @test z_draft[2, 3] ≈ -400.0 * 917.0 / 1028.0
    # Ocean: z_draft = 0
    @test z_draft[2, 4] ≈ 0.0
end

@testset "Mask cleaning: fill_ocean_holes!, fill_shelf_holes!, fill_small_shelf_patches!" begin
    # 6×6 interior (8×8 with border ring).
    # Layout (interior rows 1-6, cols 1-6):
    #   cols 1-3  → grounded (2)
    #   cols 4-6  → shelf (3), except cell (3,3) interior = (4,4) in full = isolated ocean
    #   cell (2,5) interior = (3,6) in full = isolated shelf (surrounded by grounded)
    m = zeros(Int, 8, 8)
    m[1, :] .= 1; m[end, :] .= 1; m[:, 1] .= 1; m[:, end] .= 1
    m[2:end-1, 2:4] .= 2   # grounded block
    m[2:end-1, 5:end-1] .= 3  # shelf block

    # Plant an isolated ocean pocket inside the shelf
    m[4, 6] = 0   # one ocean cell surrounded by shelf — not reachable from border
    # Plant an isolated shelf cell surrounded by grounded ice
    m[3, 3] = 3   # shelf cell inside the grounded block

    # fill_ocean_holes!: the isolated ocean pocket at (4,6) is not connected
    # to the outer ocean; it should be reclassified as land (1) — there is no
    # ice in a water pocket, so calling it grounded ice would be wrong.
    mc = copy(m)
    n = fill_ocean_holes!(mc)
    @test n == 1
    @test mc[4, 6] == 1   # reclassified as land
    @test mc[3, 3] == 3   # shelf cell untouched

    # fill_shelf_holes!: the isolated shelf at (3,3) has no ocean neighbour
    # → reclassified; the shelf block (cols 5-6) touches ocean → kept.
    mc2 = copy(m)
    n = fill_shelf_holes!(mc2)
    @test n == 1
    @test mc2[3, 3] == 2   # reclassified
    @test mc2[3, 6] == 3   # shelf block untouched

    # fill_small_shelf_patches!: a single-cell shelf component is below min_cells=5
    # → reclassified; the main shelf block (many cells) is kept.
    mc3 = copy(m)
    n = fill_small_shelf_patches!(mc3, 5)
    @test n == 1
    @test mc3[3, 3] == 2   # single-cell component removed
    # Main shelf block has >> 5 cells → untouched (spot-check cells away from the ocean hole)
    @test mc3[2, 5] == 3
    @test mc3[5, 7] == 3

    # min_cells=1 keeps everything including single-cell patches
    mc4 = copy(m)
    mc4[3, 3] = 3
    n = fill_small_shelf_patches!(mc4, 1)
    @test n == 0
    @test mc4[3, 3] == 3

    # Return value: cells reclassified (0 when nothing to do)
    mc5 = copy(m); mc5[3, 3] = 2   # no isolated shelf
    @test fill_small_shelf_patches!(mc5, 10) == 0
end

@testset "Mask cleaning: fill_ocean_holes! on non-square masks" begin
    # The outer-ocean seeds are the two outermost rings of the array.  With
    # nx ≠ ny, testing the far rings against the wrong dimension seeds interior
    # pockets as "outer ocean" and leaves them unfilled.
    for (nx, ny) in ((12, 6), (6, 12))
        m = ones(Int, nx, ny)
        m[2:end-1, 2:end-1] .= 3
        m[2:end-1, 2] .= 0                       # open ocean along the second ring
        pocket = nx > ny ? (8, 4) : (4, 8)       # interior, beyond the short side
        m[pocket...] = 0
        n = fill_ocean_holes!(m)
        @test n == 1
        @test m[pocket...] == 1
        @test all(m[2:end-1, 2] .== 0)           # the connected ocean is kept
    end
end

@testset "Mask cleaning: fill_small_grounded_patches!" begin
    # 6×6 interior (8×8 with border ring).
    # Left grounded block (cols 2-4) touches the border → main sheet, never removed.
    # A 2-cell isolated grounded patch (rows 3-4, col 6) sits inside the shelf.
    g = zeros(Int, 8, 8)
    g[1, :] .= 1; g[end, :] .= 1; g[:, 1] .= 1; g[:, end] .= 1
    g[2:end-1, 2:4] .= 2   # main grounded block (touches border → kept)
    g[2:end-1, 5:end-1] .= 3  # shelf block
    g[3, 6] = 2             # isolated grounded cell 1
    g[4, 6] = 2             # isolated grounded cell 2 (component size = 2)

    # min_cells=5: the 2-cell isolated patch is removed → shelf; main block untouched
    gc = copy(g)
    n = fill_small_grounded_patches!(gc, 5)
    @test n == 2
    @test gc[3, 6] == 3   # reclassified to shelf
    @test gc[4, 6] == 3
    @test gc[3, 3] == 2   # main grounded block untouched

    # min_cells=1: nothing removed (component size 2 >= 1)
    gc2 = copy(g)
    @test fill_small_grounded_patches!(gc2, 1) == 0
    @test gc2[3, 6] == 2   # kept

    # min_cells=3: 2-cell patch is below threshold → removed
    gc3 = copy(g)
    @test fill_small_grounded_patches!(gc3, 3) == 2
    @test gc3[3, 6] == 3

    # Border-connected grounded cells are never removed regardless of min_cells
    gc4 = copy(g)
    @test fill_small_grounded_patches!(gc4, 100) == 2   # only the 2-cell patch
    @test gc4[3, 3] == 2   # main block kept
end

@testset "Geometry ingestion: end-to-end Model from synthetic BedMachine" begin
    # 4×8 interior: cols 1-2 grounded, cols 3-8 floating
    ny_bm, nx_bm = 4, 8
    bed_bm = fill(-500.0, ny_bm, nx_bm)
    h_bm   = zeros(ny_bm, nx_bm)
    h_bm[:, 1:2] .= 600.0   # grounded: h_af = 600*917/1028 - 500 > 0
    h_bm[:, 3:8] .= 400.0   # floating: h_af = 400*917/1028 - 500 < 0
    mask_bm = build_laddie_mask(bed_bm, h_bm)
    zb_bm   = ice_base_depth(bed_bm, h_bm)

    @test count(==(2), mask_bm) == ny_bm * 2
    @test count(==(3), mask_bm) == ny_bm * 6

    forcing = ISOMIPForcing(:warm; FT)
    params  = Params(; FT)
    m = Simulation(Model(Grid(mask_bm, zb_bm, 2000.0, 2000.0; FT); forcing, params))
    @test all(isfinite, m.model.melt)
    @test all(m.model.melt[m.model.tmask .> 0] .>= 0)
    run!(m; days = 0.5, verbose = false)
    @test all(isfinite, m.model.D.present)
    @test all(isfinite, m.model.melt)
end
